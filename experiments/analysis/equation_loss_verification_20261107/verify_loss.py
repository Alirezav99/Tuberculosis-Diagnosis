"""Read-only run provenance audit and isolated synthetic loss checks.

Run with the project's .venv/bin/python. No trainer, dataset, or model is run.
Only this audit directory receives new files.
"""

import ast
import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LOSS = ROOT / "src/afb_segmentation/models/loss/segmentation.py"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class WithoutDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node

    visit_ClassDef = visit_FunctionDef


def class_ast(source, name):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    return ast.dump(WithoutDocstrings().visit(node), include_attributes=False)


spec = importlib.util.spec_from_file_location("isolated_project_losses", LOSS)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
boundary = module.BoundaryLoss()
checks = []


def check(name, condition):
    assert condition, name
    checks.append(name)


def weights(target):
    # Large exponent values legitimately produce zero weights at float precision.
    with np.errstate(over="ignore", under="ignore"):
        return boundary._compute_boundary_weights(target)


def summary(array):
    a = array.detach().numpy().astype(np.float64)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "sum": float(a.sum()),
        "nonzero_pixels": int(np.count_nonzero(a)),
        "total_pixels": int(a.size),
    }


a = torch.zeros((1, 1, 400, 400), dtype=torch.float32)
a[:, :, 80:100, 80:100] = 1
b = torch.zeros_like(a)
b[:, :, 280:300, 280:300] = 1
eta = 0.08
smooth_a = a * (1 - eta) + 0.5 * eta
smooth_b = b * (1 - eta) + 0.5 * eta
hard_a, hard_b = weights(a), weights(b)
soft_a, soft_b = weights(smooth_a), weights(smooth_b)
check(
    "binary weight map changes when the object moves", not torch.equal(hard_a, hard_b)
)
check(
    "smoothed weight map is identical when the object moves",
    torch.equal(soft_a, soft_b),
)
check(
    "smoothed targets and complements have no zero pixels",
    bool(torch.all(smooth_a > 0) and torch.all(1 - smooth_a > 0)),
)
with np.errstate(over="ignore", under="ignore"):
    all_ones_distance = 2 * ndimage.distance_transform_edt(np.ones((400, 400)))
    all_ones_weights = torch.from_numpy(
        (1 / (1 + np.exp((all_ones_distance - 3) / 5))).astype(np.float32)
    ).reshape(1, 1, 400, 400)
check(
    "smoothed weights equal distance weights of all-nonzero arrays",
    torch.equal(soft_a, all_ones_weights),
)
check(
    "smoothed empty mask has the same weight map",
    torch.equal(soft_a, weights(torch.full_like(a, eta / 2))),
)
check(
    "smoothed full mask has the same weight map",
    torch.equal(soft_a, weights(torch.full_like(a, 1 - eta / 2))),
)
check(
    "unsmoothed empty/full masks use constant 0.5 weights",
    bool(
        torch.all(weights(torch.zeros_like(a)) == 0.5)
        and torch.all(weights(torch.ones_like(a)) == 0.5)
    ),
)

pred = torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float32)
target = torch.tensor([0.0, 1.0, 0.0, 1.0])
alpha, gamma = 0.7815, 2.18
focal = module.FocalLoss(alpha=alpha, gamma=gamma, reduction="none")
focal_results = {}
for name, t in [("binary", target), ("smoothed", target * (1 - eta) + eta / 2)]:
    pt = pred * t + (1 - pred) * (1 - t)
    at = alpha * t + (1 - alpha) * (1 - t)
    implemented = focal(pred, t)
    bce = -t * torch.log(pred) - (1 - t) * torch.log(1 - pred)
    explicit_bce = at * (1 - pt) ** gamma * bce
    compact_logpt = -at * (1 - pt) ** gamma * torch.log(pt)
    check(
        f"{name} focal loss equals weighted BCE expression",
        torch.allclose(implemented, explicit_bce),
    )
    check(
        f"{name} compact log(pt) equivalence",
        (
            torch.allclose(implemented, compact_logpt)
            if name == "binary"
            else not torch.allclose(implemented, compact_logpt)
        ),
    )
    focal_results[name] = {
        "implementation": implemented.tolist(),
        "weighted_bce": explicit_bce.tolist(),
        "compact_logpt": compact_logpt.tolist(),
        "max_abs_difference_from_compact_logpt": float(
            (implemented - compact_logpt).abs().max()
        ),
    }

historical = []
for revision, path in [
    ("744997c", "src/tb_diagnosis/models/losses.py"),
    ("580601f", "src/tb_diagnosis/models/losses.py"),
    ("2be86f8", "src/afb_segmentation/models/loss/segmentation.py"),
    ("HEAD", "src/afb_segmentation/models/loss/segmentation.py"),
]:
    source = git("show", f"{revision}:{path}")
    matches = {
        name: class_ast(source, name) == class_ast(LOSS.read_text(), name)
        for name in ["BoundaryLoss", "FocalLoss", "DiceLoss", "CompoundLoss"]
    }
    historical.append(
        {
            "revision": git("rev-parse", revision).strip(),
            "path": path,
            "date": git("show", "-s", "--format=%aI", revision).strip(),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "class_ast_matches_current_ignoring_docstrings": matches,
        }
    )
    check(f"{revision} loss executable AST matches current", all(matches.values()))

checkpoint_results = []
checkpoint_paths = [
    "artifacts/models/unet/with_augmentation/weights/best.pt",
    "artifacts/models/unet/with_augmentation/weights/last.pt",
    "artifacts/models/unet/with_augmentation/weights/epoch_024_last.pt",
    "artifacts/models/unet/without_augmentation/weights/best.pt",
    "artifacts/models/unet/without_augmentation/weights/last.pt",
    "artifacts/models/unet/without_augmentation/weights/epoch_030_last.pt",
    "experiments/analysis/reviewer_q4_lowload_20261030/unet_runs/boundary_true/best.pt",
    "experiments/analysis/reviewer_q4_lowload_20261030/unet_runs/boundary_false/best.pt",
]
input_paths = [
    Path(__file__).resolve(),
    LOSS,
    ROOT / "src/afb_segmentation/models/training/segmentation_epochs.py",
    ROOT / "src/afb_segmentation/models/training/unet_components.py",
    ROOT / "src/afb_segmentation/models/training/unet_trainer.py",
    ROOT / "src/afb_segmentation/models/config.yml",
    ROOT / "docs/paper/my-paper/word/Manuscript 14050514.docx",
]
for rel in checkpoint_paths:
    p = ROOT / rel
    data = torch.load(p, map_location="cpu", weights_only=True)
    metadata = {
        k: v
        for k, v in data.items()
        if k not in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict"]
    }
    row = {"path": rel, "sha256": sha256(p), "keys": list(data), "metadata": metadata}
    if "/models/unet/" in rel:
        history = p.parents[1] / "results/train_history.csv"
        records = list(csv.DictReader(history.open()))
        matching = [r for r in records if int(r["epoch"]) == data["epoch"]]
        row["history_epoch_present"] = bool(matching)
        if matching:
            row["checkpoint_validation_loss_matches_history"] = np.isclose(
                float(matching[0]["val_loss"]),
                data["val_metrics"]["loss"],
                rtol=0,
                atol=1e-12,
            ).item()
            row["checkpoint_validation_f1_matches_history"] = np.isclose(
                float(matching[0]["val_f1"]),
                data["val_metrics"]["f1"],
                rtol=0,
                atol=1e-12,
            ).item()
        input_paths.extend([history, p.parents[1] / "results/test_results.csv"])
    else:
        history = p.parent / "history.csv"
        records = list(csv.DictReader(history.open()))
        matching = [r for r in records if int(r["epoch"]) == data["epoch"]]
        row["checkpoint_validation_loss_matches_history"] = bool(
            matching and abs(float(matching[0]["val_loss"]) - data["val_loss"]) < 1e-12
        )
        check(
            f"{p.parent.name} supplementary checkpoint matches history",
            row["checkpoint_validation_loss_matches_history"],
        )
    checkpoint_results.append(row)
    input_paths.append(p)
    del data

supplement = ROOT / "experiments/analysis/reviewer_q4_lowload_20261030"
for name in [
    "run_lowload_ablation.py",
    "unet_training.json",
    "unet_runs/boundary_true/history.csv",
    "unet_runs/boundary_false/history.csv",
    "summary.json",
]:
    input_paths.append(supplement / name)

manifest = {str(p.relative_to(ROOT)): sha256(p) for p in dict.fromkeys(input_paths)}
result = {
    "purpose": "Isolated implementation verification, not a segmentation experiment or estimate of training impact",
    "git_head": git("rev-parse", "HEAD").strip(),
    "versions": {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    },
    "synthetic": {
        "shape": list(a.shape),
        "smoothing": eta,
        "target_values": torch.unique(smooth_a).tolist(),
        "binary_weights": summary(hard_a),
        "smoothed_weights": summary(soft_a),
        "binary_max_abs_difference_after_object_move": float(
            (hard_a - hard_b).abs().max()
        ),
        "smoothed_max_abs_difference_after_object_move": float(
            (soft_a - soft_b).abs().max()
        ),
        "weight_samples": {
            f"row_{r}_col_{c}": {
                "binary": float(hard_a[0, 0, r, c]),
                "smoothed": float(soft_a[0, 0, r, c]),
            }
            for r, c in [(0, 0), (80, 80), (90, 90), (200, 200), (399, 399)]
        },
        "focal": focal_results,
    },
    "checks_passed": checks,
    "historical_loss_comparisons": historical,
    "boundary_enable_change": {
        "revision": git("rev-parse", "02379ae").strip(),
        "parent_flag": False,
        "new_flag": True,
        "path": "src/afb_segmentation/models/training/unet_components.py",
        "evidence": git(
            "show",
            "02379ae",
            "--",
            "src/afb_segmentation/models/training/unet_components.py",
        ),
    },
    "checkpoints": checkpoint_results,
    "input_sha256_before": manifest,
    "original_unet_file_inventory": sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "artifacts/models/unet").rglob("*")
        if p.is_file()
    ),
    "limitations": [
        "Original checkpoints do not identify the run configuration or source commit.",
        "Git history alone cannot exclude uncommitted code used for a historical run.",
        "Supplementary script imports the project loss module without recording its source hash at training time.",
        "Synthetic results establish implementation behavior, not the effect of a correction on trained model performance.",
    ],
}
check(
    "all inspected inputs remain unchanged",
    all(sha256(ROOT / rel) == expected for rel, expected in manifest.items()),
)
result["input_sha256_after_matches_before"] = True
(OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "checks_passed": len(checks),
            "synthetic": result["synthetic"],
            "historical_loss_comparisons": historical,
            "checkpoints": checkpoint_results,
            "inputs_unchanged": True,
        },
        indent=2,
    )
)
