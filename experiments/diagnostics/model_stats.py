"""Small experiment utilities for inspecting model resource usage."""

from ultralytics import YOLO


def compute_model_stats():
    """Load selected YOLO segmentation variants and print their model info."""
    for variant in ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg"]:
        model = YOLO(f"{variant}.pt")
        info = model.info()
        print(f"{variant}: {info}")


if __name__ == "__main__":
    compute_model_stats()
