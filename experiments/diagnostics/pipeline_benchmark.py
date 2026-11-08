"""Benchmark saved AFB components without training or changing source data.

Run from the repository root with ``python -m
experiments.diagnostics.pipeline_benchmark --output-dir <new-directory>``.
Each device/mode runs in a fresh process. Timings include reading a PNG,
YOLO prediction, mask/box conversion, distance filtering, tensor conversion,
U-Net prediction, thresholding, and output transfer to CPU. Model loading,
warmup, accuracy scoring, output-file writing, and microscopy acquisition
are excluded. Stage boundaries synchronize CUDA; no stages overlap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import platform
import random
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YOLO_PATH = "artifacts/models/yolo11l-seg - True/train/weights/best.pt"
UNET_PATH = "artifacts/models/unet/with_augmentation/weights/best.pt"
CONFIG_PATH = "src/afb_segmentation/models/config.yml"
IMAGE_DIR = "artifacts/dataset/dds3-restructured/test/images"
SOURCE_PATHS = [
    "experiments/diagnostics/pipeline_benchmark.py",
    "src/afb_segmentation/utils/yolo.py",
    "src/afb_segmentation/utils/image_filtering.py",
    "src/afb_segmentation/models/architectures/unet.py",
    "src/afb_segmentation/models/evaluation/uncertainty.py",
    CONFIG_PATH,
]
STAGES = [
    "read_decode_ms",
    "yolo_predict_ms",
    "yolo_output_ms",
    "distance_filter_ms",
    "tensor_transfer_ms",
    "unet_ms",
    "output_transfer_ms",
    "end_to_end_ms",
]


def sha256(path: Path) -> str:
    """Return the SHA-256 digest without loading a whole checkpoint at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def worker(output: Path, case: str) -> None:
    """Measure one device and refinement mode in an isolated process."""
    # Historical checkpoints reference Python 3.13's pathlib submodule.
    sys.modules.setdefault("pathlib._local", pathlib)
    import cv2
    import numpy as np
    import psutil
    import torch
    import ultralytics
    import yaml
    from PIL import Image
    from ultralytics import YOLO

    from afb_segmentation.models.architectures.unet import AttentionUNetMaskPredictor
    from afb_segmentation.models.evaluation.uncertainty import mc_dropout_inference
    from afb_segmentation.utils.image_filtering import distance_based_filtering
    from afb_segmentation.utils.yolo import YoloModel

    manifest = json.loads((output / "manifest.json").read_text())
    settings = manifest["settings"]
    device_name, mode = case.split("_", 1)
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(settings["cpu_threads"])
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.backends.cudnn.benchmark = False
    cfg = yaml.safe_load((ROOT / CONFIG_PATH).read_text())["unet"]

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    started = time.perf_counter()
    detector = YOLO(str(ROOT / YOLO_PATH)).to(device)
    yolo_params = sum(p.numel() for p in detector.model.parameters())
    wrapper = YoloModel(detector, input_size=400, conf=settings["yolo_confidence"])
    checkpoint = torch.load(ROOT / UNET_PATH, map_location="cpu", weights_only=False)
    refiner = AttentionUNetMaskPredictor(
        in_channels=3,
        out_channels=1,
        dropout=cfg["dropout"],
        base_channels=64,
        depth=cfg["depth"],
        channel_factor=2,
        scale_factor=cfg["scale_factor"],
    )
    refiner.load_state_dict(checkpoint["model_state_dict"], strict=True)
    checkpoint_epoch = checkpoint.get("epoch")
    del checkpoint
    unet_params = sum(p.numel() for p in refiner.parameters())
    refiner = refiner.to(device).float().eval()
    sync()
    model_load_s = time.perf_counter() - started
    images = [ROOT / item["path"] for item in manifest["images"]]

    def infer(path: Path) -> dict:
        sync()
        marks = [time.perf_counter()]
        with Image.open(path) as source:
            image = np.array(source.convert("RGB"))
        assert image.shape == (400, 400, 3), image.shape
        marks.append(time.perf_counter())
        # These are the project's YoloModel prediction arguments with an
        # explicit device and FP32 setting, avoiding device auto-selection.
        results = detector.predict(
            source=image,
            imgsz=wrapper.output_size,
            conf=wrapper.conf,
            agnostic_nms=True,
            verbose=False,
            device=str(device),
            half=False,
            save=False,
        )
        sync()
        marks.append(time.perf_counter())
        if results[0].boxes is not None and len(results[0].boxes):
            if results[0].masks is None:
                raise RuntimeError("Detector returned boxes without segmentation masks")
        mask = wrapper.extract_masks(results)[0]
        boxes = wrapper.extract_boxes(results)[0].astype(np.int16)
        sync()
        marks.append(time.perf_counter())
        filtered = distance_based_filtering(img=image, msk=mask)
        filtered = cv2.resize(filtered, (400, 400))
        marks.append(time.perf_counter())
        tensor = (
            torch.FloatTensor(filtered / 255).permute(2, 0, 1).unsqueeze(0).to(device)
        )
        sync()
        marks.append(time.perf_counter())
        with torch.no_grad():
            if mode == "mc10":
                prediction, uncertainty = mc_dropout_inference(
                    refiner, tensor, 10, device
                )
            else:
                prediction = refiner(tensor)
                uncertainty = None
        sync()
        marks.append(time.perf_counter())
        binary = (prediction > settings["unet_threshold"]).int().cpu().numpy().squeeze()
        if uncertainty is not None:
            uncertainty = uncertainty.cpu().numpy().squeeze()
        sync()
        marks.append(time.perf_counter())
        assert binary.shape == (400, 400)
        assert np.isfinite(prediction.detach().cpu().numpy()).all()
        if uncertainty is not None:
            assert uncertainty.shape == (400, 400) and np.isfinite(uncertainty).all()
        values = {
            stage: (marks[i + 1] - marks[i]) * 1000
            for i, stage in enumerate(STAGES[:-1])
        }
        values["end_to_end_ms"] = (marks[-1] - marks[0]) * 1000
        # Inspection and checksums happen after the timed region.
        values.update(
            filename=path.name,
            predicted_boxes=len(boxes),
            foreground_pixels=int(binary.sum()),
            mask_sha256=hashlib.sha256(binary.astype(np.uint8).tobytes()).hexdigest(),
        )
        return values

    warmup_times = []
    for i in range(settings["warmup_images"]):
        result = infer(images[i % len(images)])
        warmup_times.append(result["end_to_end_ms"])
    print(f"{case}: warmup complete; first call {warmup_times[0]:.1f} ms", flush=True)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak_rss = [rss_before]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.01):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    try:
        for repetition in range(settings["repeats"]):
            order = list(images)
            random.Random(settings["seed"] + repetition).shuffle(order)
            repeat_started = time.perf_counter()
            for i, path in enumerate(order):
                row = infer(path)
                row.update(case=case, repeat=repetition + 1, index=i + 1)
                rows.append(row)
                if (i + 1) % 10 == 0:
                    print(
                        f"{case}: repeat {repetition + 1}/{settings['repeats']}, "
                        f"image {i + 1}/{len(order)}, "
                        f"latest {row['end_to_end_ms']:.1f} ms",
                        flush=True,
                    )
            print(
                f"{case}: repeat {repetition + 1} finished in "
                f"{time.perf_counter() - repeat_started:.1f} s",
                flush=True,
            )
    finally:
        stop.set()
        monitor_thread.join()
    peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    if mode == "single":
        for name in {r["filename"] for r in rows}:
            assert (
                len({r["mask_sha256"] for r in rows if r["filename"] == name}) == 1
            ), name

    def describe(values: list[float]) -> dict:
        a = np.asarray(values, dtype=float)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "p95": float(np.percentile(a, 95)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    result = {
        "case": case,
        "device": str(device),
        "mode": mode,
        "images": len(images),
        "repeats": settings["repeats"],
        "measurements": len(rows),
        "stages_ms": {stage: describe([r[stage] for r in rows]) for stage in STAGES},
        "serial_images_per_second": 1000 / np.mean([r["end_to_end_ms"] for r in rows]),
        "repeat_mean_end_to_end_ms": [
            float(np.mean([r["end_to_end_ms"] for r in rows if r["repeat"] == rep]))
            for rep in range(1, settings["repeats"] + 1)
        ],
        "process_rss_before_measurement_mib": rss_before / 1024**2,
        "process_peak_rss_sampled_mib": peak_rss[0] / 1024**2,
        "process_lifetime_high_water_rss_mib": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        / 1024,
        "cuda_peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "model_load_seconds": model_load_s,
        "warmup_end_to_end_ms": warmup_times,
        "yolo_parameters_before_inference_fusion": yolo_params,
        "yolo_parameters_after_inference_fusion": sum(
            p.numel() for p in detector.model.parameters()
        ),
        "unet_parameters": unet_params,
        "unet_checkpoint_epoch": checkpoint_epoch,
        "unet_architecture": {
            key: cfg[key] for key in ["dropout", "depth", "scale_factor"]
        },
        "python": sys.version,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "torch_cpu_threads_after_warmup": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "model_dtypes": {
            "yolo": str(next(detector.model.parameters()).dtype),
            "unet": str(next(refiner.parameters()).dtype),
        },
    }
    with (output / f"{case}_per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / f"{case}.json", result)
    print(
        f"{case}: completed; mean {result['stages_ms']['end_to_end_ms']['mean']:.1f} ms",
        flush=True,
    )


def main() -> None:
    """Record provenance, run four isolated benchmark cases, and summarize."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument(
        "--worker",
        choices=["cuda_single", "cuda_mc10", "cpu_single", "cpu_mc10"],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker:
        worker(output, args.worker)
        return
    if min(args.images, args.repeats, args.warmup, args.cpu_threads) < 1:
        parser.error("image, repeat, warmup and thread counts must be positive")
    files = sorted((ROOT / IMAGE_DIR).glob("*.png"))
    if len(files) < args.images:
        parser.error(f"Need {args.images} images; found {len(files)}")
    selected = sorted(random.Random(args.seed).sample(files, args.images))
    for name in [YOLO_PATH, UNET_PATH, *SOURCE_PATHS]:
        if not (ROOT / name).is_file():
            parser.error(f"Required file not found: {name}")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Supplementary end-to-end AFB inference cost benchmark",
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "hardware": {
            "platform": platform.platform(),
            "cpu": command_output(["lscpu"]),
            "memory": command_output(["free", "-b"]),
            "gpu_before": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                    "--format=csv",
                ]
            ),
            "load_average_before": list(os.getloadavg()),
        },
        "settings": {
            "seed": args.seed,
            "unique_images": args.images,
            "repeats": args.repeats,
            "warmup_images": args.warmup,
            "cpu_threads": args.cpu_threads,
            "batch_size": 1,
            "input_size": [400, 400],
            "yolo_inference_size": [416, 416],
            "precision": "FP32 (no autocast or explicit FP16); backend TF32 flags recorded per case",
            "yolo_confidence": 0.5,
            "unet_threshold": 0.5,
            "mc_samples": 10,
            "read_decode_included": True,
            "accuracy_scoring_included": False,
            "output_file_writes_included": False,
        },
        "checkpoints": {
            name: {"sha256": sha256(ROOT / name), "bytes": (ROOT / name).stat().st_size}
            for name in [YOLO_PATH, UNET_PATH]
        },
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_PATHS},
        "images": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in selected
        ],
        "limitations": [
            "Uses the saved augmented-data checkpoints; checkpoint identity relative to historical manuscript runs is not independently established.",
            "Standalone Pipeline cannot be imported because MaskRefiner is missing; this harness composes the existing YOLO, filtering, U-Net and MC-dropout components directly.",
            "One YOLO prediction is reused for filtering. The unusable Pipeline.__call__ contains another detector call; its literal runtime is not benchmarked.",
            "Single-image serial inference, synchronized stages, warm loaded models and warmed filesystem cache; no batch-throughput or cold-start claim.",
            "Runtime repetitions are not independent training runs or evidence of accuracy reproducibility.",
            "CPU-only uses a high-end desktop CPU with eight PyTorch threads, not a validated low-cost laboratory computer.",
            "GPU memory means PyTorch allocated/reserved memory, not total board memory or minimum supported hardware.",
            "RSS includes Python, libraries and loaded model process memory; sampled runtime RSS and lifetime OS high-water values are distinguished.",
            "Shared workstation with no exclusive GPU reservation; normal system activity may affect runtime.",
            "No whole-slide acquisition, image stitching, data transfer from a microscope, storage or clinical workflow time is measured.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS=str(args.cpu_threads),
        MKL_NUM_THREADS=str(args.cpu_threads),
        OPENBLAS_NUM_THREADS=str(args.cpu_threads),
        NUMEXPR_NUM_THREADS=str(args.cpu_threads),
    )
    cases = ["cuda_single", "cuda_mc10", "cpu_single", "cpu_mc10"]
    for case in cases:
        command = [
            sys.executable,
            "-u",
            "-m",
            "experiments.diagnostics.pipeline_benchmark",
            "--output-dir",
            str(output),
            "--worker",
            case,
        ]
        print(f"Starting {case}", flush=True)
        with (output / f"{case}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(
                    f"Benchmark {case} failed; see {output / (case + '.log')}"
                )
    summary = {
        "cases": {
            case: json.loads((output / f"{case}.json").read_text()) for case in cases
        },
        "limitations": manifest["limitations"],
    }
    write_json(output / "summary.json", summary)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["hardware"]["load_average_after"] = list(os.getloadavg())
    manifest["hardware"]["gpu_after"] = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv",
        ]
    )
    assert all(
        sha256(ROOT / item["path"]) == item["sha256"] for item in manifest["images"]
    )
    assert all(
        sha256(ROOT / name) == info["sha256"]
        for name, info in manifest["checkpoints"].items()
    )
    manifest["input_integrity_verified_after_run"] = True
    write_json(output / "manifest.json", manifest)
    print(f"Completed benchmark: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
