# Required model download

> **Warning:** A fresh repository clone contains directory placeholders, but
> does not include the trained YOLO and Attention U-Net weights. Download the
> model archive from Google Drive before running workflows that load these
> checkpoints.

**[Download the model archive from Google Drive](https://drive.google.com/file/d/1Xuy3VppsPMXnCZeMW0zXb8-Fst7mUmRe/view?usp=sharing).**

To install the model files:

1. Download the model archive using the Google Drive link above.
2. Extract the archive into the repository's `artifacts/` directory. The archive
   contains a `models/` folder, which should populate `artifacts/models/`.
3. Check that the required checkpoints exist, including:

   - `artifacts/models/yolo11l-seg - True/train/weights/best.pt`
   - `artifacts/models/yolo11l-seg - False/train/weights/best.pt`
   - `artifacts/models/unet/with_augmentation/weights/best.pt`
   - `artifacts/models/unet/without_augmentation/weights/best.pt`

Preserve the archive's folder names and layout, including spaces. Avoid creating
an extra `artifacts/models/models/` directory. Review the checkpoint paths in
`src/afb_segmentation/models/config.yml` before running a workflow.

Downloaded weights remain local and are ignored by Git. This README is included
in the repository so the download instructions are available in fresh clones.
