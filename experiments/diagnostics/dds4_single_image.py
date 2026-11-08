"""Count segmentation candidates in one DDS4 field using saved checkpoints.

Run with ``python -m experiments.diagnostics.dds4_single_image --image <bmp>
--output-dir <new-directory>``. Source images and checkpoint files are read
only. Counts are four-connected regions in the merged refinement mask, not
clinically verified bacilli or a comparison to DDS4's whole-set count.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from afb_segmentation.data_pipeline.load.dds4 import file_sha256
from afb_segmentation.models.inference.patch_refinement import PatchRefinement
from afb_segmentation.models.inference.tiled import predict_field

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Run one reproducible trial and save masks, counts, and a numbered overlay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--refinement-threshold", type=float, default=0.5)
    parser.add_argument(
        "--yolo-checkpoint",
        type=Path,
        default=ROOT / "artifacts/models/yolo11l-seg - True/train/weights/best.pt",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=ROOT / "artifacts/models/unet/with_augmentation/weights/best.pt",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "src/afb_segmentation/models/config.yml"
    )
    args = parser.parse_args()
    source = args.image.resolve(strict=True)
    output = args.output_dir.resolve()
    if output == source.parent or source.parent in output.parents:
        raise ValueError("Write results outside the source field directory")
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    before_hash = file_sha256(source)
    with Image.open(source) as image:
        array = np.array(image.convert("RGB"))
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    predictor = PatchRefinement(
        args.yolo_checkpoint,
        args.unet_checkpoint,
        args.config,
        args.device,
        confidence=args.yolo_confidence,
    )
    started = time.perf_counter()
    result = predict_field(
        array,
        predictor,
        overlap=args.overlap,
        batch_size=args.batch_size,
        threshold=args.refinement_threshold,
    )
    elapsed = time.perf_counter() - started
    if file_sha256(source) != before_hash:
        raise RuntimeError("Source image changed during inference")
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(array).save(output / "image.png")
    Image.fromarray(result["refined_mask"].astype(np.uint8) * 255).save(
        output / "refined_mask.png"
    )
    Image.fromarray(result["yolo_mask"].astype(np.uint8) * 255).save(
        output / "yolo_mask.png"
    )
    np.savez_compressed(
        output / "probabilities.npz",
        probability=result["probability"],
        coverage=result["coverage"],
    )

    labels = result["labels"]
    count = result["component_count"]
    areas = np.bincount(labels.ravel())[1:]
    centers = ndimage.center_of_mass(
        result["refined_mask"], labels, range(1, count + 1)
    )
    bounds = ndimage.find_objects(labels)
    rows = []
    for i, (center, box) in enumerate(zip(centers, bounds), start=1):
        rows.append(
            {
                "id": i,
                "area_pixels": int(areas[i - 1]),
                "centroid_x": float(center[1]),
                "centroid_y": float(center[0]),
                "x1": box[1].start,
                "y1": box[0].start,
                "x2_exclusive": box[1].stop,
                "y2_exclusive": box[0].stop,
            }
        )
    with (output / "components.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "area_pixels",
                "centroid_x",
                "centroid_y",
                "x1",
                "y1",
                "x2_exclusive",
                "y2_exclusive",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    overlay = array.copy()
    mask = result["refined_mask"]
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([0, 255, 80])).astype(
        np.uint8
    )
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 160, 0), 1)
    display = Image.fromarray(overlay)
    draw = ImageDraw.Draw(display)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    placed = []
    for row in rows:
        x, y = row["centroid_x"] + 5, max(0, row["centroid_y"] - 15)
        for _ in range(20):
            if not any(abs(x - px) < 22 and abs(y - py) < 18 for px, py in placed):
                break
            x = min(x + 20, display.width - 35)
            y = min(y + 18, display.height - 20)
        placed.append((x, y))
        if abs(x - row["centroid_x"]) > 22 or abs(y - row["centroid_y"]) > 25:
            draw.line(
                (row["centroid_x"], row["centroid_y"], x, y + 9),
                fill=(0, 110, 0),
                width=1,
            )
        draw.text(
            (x, y),
            str(row["id"]),
            fill="black",
            stroke_width=2,
            stroke_fill="white",
            font=font,
        )
    title = f"YOLO + Attention U-Net: {count} candidate regions (estimated count)"
    titled = Image.new("RGB", (display.width, display.height + 38), "white")
    titled.paste(display, (0, 38))
    ImageDraw.Draw(titled).text((10, 9), title, fill="black", font=font)
    titled.save(output / "overlay.png")

    source_paths = [
        Path(__file__),
        ROOT / "src/afb_segmentation/models/inference/patch_refinement.py",
        ROOT / "src/afb_segmentation/models/inference/tiled.py",
        ROOT / "src/afb_segmentation/models/architectures/unet.py",
        ROOT / "src/afb_segmentation/utils/image_filtering.py",
        ROOT / "src/afb_segmentation/utils/yolo.py",
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "sha256": before_hash,
            "width": array.shape[1],
            "height": array.shape[0],
            "unchanged_after_inference": True,
        },
        "settings": {
            "device": args.device,
            "tile_size": predictor.size,
            "overlap": args.overlap,
            "batch_size": args.batch_size,
            "yolo_imgsz": predictor.yolo.output_size,
            "yolo_confidence": args.yolo_confidence,
            "refinement_threshold": args.refinement_threshold,
            "merge": "arithmetic mean of probabilities in overlapping pixels before thresholding",
            "connectivity": 4,
            "area_filter": None,
            "morphology": None,
            "refinement": "single_pass_fp32_dropout_disabled",
        },
        "results": {
            "candidate_count": count,
            "detector_union_components": result["yolo_component_count"],
            "foreground_pixels": int(mask.sum()),
            "tile_count": len(result["tiles"]),
            "components_of_one_or_two_pixels": int((areas <= 2).sum()),
            "elapsed_inference_seconds_including_first_call": elapsed,
        },
        "tiles": result["tiles"],
        "checkpoints": {
            "yolo": {
                "path": str(args.yolo_checkpoint.resolve()),
                "sha256": file_sha256(args.yolo_checkpoint),
            },
            "unet": {
                "path": str(args.unet_checkpoint.resolve()),
                "sha256": file_sha256(args.unet_checkpoint),
                "saved_epoch": predictor.checkpoint_epoch,
            },
            "config": {
                "path": str(args.config.resolve()),
                "sha256": file_sha256(args.config),
            },
        },
        "code_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in source_paths},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": (
                torch.cuda.get_device_name(predictor.device)
                if predictor.device.type == "cuda"
                else None
            ),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "limitations": [
            "No per-field reference count or mask is supplied by DDS4.",
            "Connected regions are segmentation candidates; touching bacilli may merge and fragmented predictions may split.",
            "No threshold tuning, size filtering, or retraining was performed.",
            "Tiling and merging are a DDS4 adaptation, not a previously validated full-field counting method.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), **report["results"]}, indent=2))


if __name__ == "__main__":
    main()
