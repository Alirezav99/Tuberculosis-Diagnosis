"""Move augmentation files between working dataset directories."""

import os
import re
import shutil


class AugmentManager:
    """Move augmentation files between prepared DDS3 dataset layouts."""

    @staticmethod
    def exclude_yolo_dataset_augmentation() -> None:
        """
        Move augmentation files from the YOLO dataset training folder to a separate augmentation folder.

        Moves files matching the pattern MSC{number}.png and MSC{number}.txt where number > 6000
        from the dds3-yolo/images/train and dds3-yolo/labels/train directories to
        dds3-yolo/augmentation/images and dds3-yolo/augmentation/labels respectively.

        Uses the ROOT_DATA environment variable to locate the dataset directory.
        Creates the augmentation directories if they don't exist.

        Returns
        -------
        None
            Prints a message if no augmentation files are found.
        """
        src_data = os.getenv("ROOT_DATA")
        parent_dir_name = "dds3-yolo"

        os.makedirs(
            os.path.join(src_data, parent_dir_name, "augmentation", "images"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(src_data, parent_dir_name, "augmentation", "labels"),
            exist_ok=True,
        )

        pattern_images = re.compile(r"MSC(\d+)\.png")
        pattern_labels = re.compile(r"MSC(\d+)\.txt")

        found_aug = False
        for folder in ["images", "labels"]:
            for file in os.listdir(
                os.path.join(src_data, parent_dir_name, folder, "train")
            ):
                match_images = pattern_images.fullmatch(file)
                match_labels = pattern_labels.fullmatch(file)

                if match_images or match_labels:
                    number = (
                        int(match_images.group(1))
                        if match_images
                        else int(match_labels.group(1))
                    )
                    if number > 6000:
                        found_aug = True
                        src_path = os.path.join(
                            src_data, parent_dir_name, folder, "train", file
                        )
                        dst_path = os.path.join(
                            src_data, parent_dir_name, "augmentation", folder, file
                        )
                        shutil.move(src_path, dst_path)

        if not found_aug:
            print("No augmentation files found.")

    @staticmethod
    def include_yolo_dataset_augmentation() -> None:
        """
        Copy augmentation files from the augmentation folder back to the YOLO training dataset.

        Copies files matching the pattern MSC{number}.png and MSC{number}.txt from
        dds3-yolo/augmentation/images and dds3-yolo/augmentation/labels to the corresponding
        training directories (dds3-yolo/images/train and dds3-yolo/labels/train).

        Uses the ROOT_DATA environment variable to locate the dataset directory.
        Only processes files if the ROOT_DATA directory exists.

        Returns
        -------
        None
        """
        src_data = os.getenv("ROOT_DATA")
        parent_dir_name = "dds3-yolo"

        pattern_imgs = re.compile(r"MSC(\d+)\.png")
        pattern_labels = re.compile(r"MSC(\d+)\.txt")
        augmentation_folder = os.path.join(src_data, parent_dir_name, "augmentation")

        if os.path.exists(src_data):
            for folder in ["images", "labels"]:
                for file in os.listdir(os.path.join(augmentation_folder, folder)):
                    match_imgs = pattern_imgs.fullmatch(file)
                    match_labels = pattern_labels.fullmatch(file)

                    if match_imgs or match_labels:
                        shutil.copy(
                            src=os.path.join(augmentation_folder, folder, file),
                            dst=os.path.join(
                                src_data, parent_dir_name, folder, "train", file
                            ),
                        )

    @staticmethod
    def exclude_augmentation():
        """
        Move augmented images and masks from the main training dataset to a separate augmentation folder.

        Moves files with numbers >= 6000 from dds3-restructured/train/images and
        dds3-restructured/train/masks to dds3-restructured/augmentations/train/images and
        dds3-restructured/augmentations/train/masks respectively.

        Uses the ROOT_DATA environment variable to locate the dataset directory.
        Creates the destination directories if they don't exist.
        Processes image files ending with .jpg or .png and their corresponding mask files.

        Returns
        -------
        None
        """
        src_dir = os.path.join(os.getenv("ROOT_DATA"), "dds3-restructured")
        dst_dir = os.path.join(
            os.getenv("ROOT_DATA"), "dds3-restructured", "augmentations"
        )

        os.makedirs(os.path.join(dst_dir, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, "train", "masks"), exist_ok=True)

        train_images = os.path.join(src_dir, "train", "images")
        train_labels = os.path.join(src_dir, "train", "masks")

        for img_file in os.listdir(train_images):
            if img_file.endswith((".jpg", ".png")):
                num = int(re.findall(r"\d+", img_file)[0])

                if num >= 6000:
                    img_src = os.path.join(train_images, img_file)
                    img_dst = os.path.join(dst_dir, "train", "images", img_file)
                    shutil.move(img_src, img_dst)

                    label_src = os.path.join(train_labels, img_file)
                    label_dst = os.path.join(dst_dir, "train", "masks", img_file)
                    shutil.move(label_src, label_dst)

    @staticmethod
    def include_augmentation():
        """
        Copy augmented images and masks back from the augmentation folder to the main training dataset.

        Copies files with numbers >= 6000 from dds3-restructured/augmentations/train/images and
        dds3-restructured/augmentations/train/masks back to dds3-restructured/train/images and
        dds3-restructured/train/masks respectively.

        Uses the ROOT_DATA environment variable to locate the dataset directory.
        Only processes PNG files and copies both the image and its corresponding mask file.

        Returns
        -------
        None
        """
        src_dir = os.path.join(
            os.getenv("ROOT_DATA"), "dds3-restructured", "augmentations"
        )
        dst_dir = os.path.join(os.getenv("ROOT_DATA"), "dds3-restructured")

        train_images_src = os.path.join(src_dir, "train", "images")
        train_labels_src = os.path.join(src_dir, "train", "masks")
        train_images_dst = os.path.join(dst_dir, "train", "images")
        train_labels_dst = os.path.join(dst_dir, "train", "masks")

        for img_file in os.listdir(train_images_src):
            if img_file.endswith((".png")):
                num = int(re.findall(r"\d+", img_file)[0])
                if num >= 6000:
                    shutil.copy(
                        os.path.join(train_images_src, img_file),
                        os.path.join(train_images_dst, img_file),
                    )

                    shutil.copy(
                        os.path.join(train_labels_src, img_file),
                        os.path.join(train_labels_dst, img_file),
                    )
