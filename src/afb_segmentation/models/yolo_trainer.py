"""Compatibility imports and command-line entry point for YOLO training.

The implementation is maintained in ``training.yolo_trainer``.
"""

from .training.yolo_trainer import train_from_scratch

__all__ = ["train_from_scratch"]

if __name__ == "__main__":
    train_from_scratch()
