"""
Build YOLO-formatted dataset from VOC-style split lists.

Reads image IDs from Train.txt, Validation.txt, Test.txt in voc_train/val/test.
Copies images from dataset_dir/images/{train,val,test}/ and labels from
voc_*/labels/... into yolo_dir. Val/test exclude any IDs already in previous splits.
"""

import argparse
import os
import shutil


def project_root():
    """Return the absolute path of the repository root (parent of utils/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_paths(voc_train, voc_val, voc_test, dataset_dir, yolo_dir):
    """Build a dict of input and output paths for the YOLO dataset build."""
    return {
        "voc_train": voc_train,
        "voc_val": voc_val,
        "voc_test": voc_test,
        "dataset_dir": dataset_dir,
        "yolo_dir": yolo_dir,
    }


def ensure_dirs(*dirs):
    """Create each directory if it does not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def read_ids(list_file):
    """Read a split file and return image IDs (without extension)."""
    ids = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_name = line.replace("\\", "/").split("/")[-1].split(".")[0]
            ids.append(image_name)
    return ids


def copy_split(ids, exclude, src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
    """Copy images and labels for each id not in exclude. Returns set of copied ids."""
    copied = set()
    for image_id in ids:
        if image_id in exclude:
            continue
        image_path = os.path.join(src_images_dir, image_id + ".jpg")
        label_path = os.path.join(src_labels_dir, image_id + ".txt")
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue
        shutil.copy2(image_path, os.path.join(dst_images_dir, image_id + ".jpg"))
        shutil.copy2(label_path, os.path.join(dst_labels_dir, image_id + ".txt"))
        copied.add(image_id)
    return copied


def main(voc_train, voc_val, voc_test, dataset_dir, yolo_dir):
    """Build yolo_data/ from VOC split lists; val and test exclude IDs already in train."""
    paths = build_paths(voc_train, voc_val, voc_test, dataset_dir, yolo_dir)
    images_dir = os.path.join(paths["yolo_dir"], "images")
    labels_dir = os.path.join(paths["yolo_dir"], "labels")
    train_img = os.path.join(images_dir, "train")
    val_img = os.path.join(images_dir, "val")
    test_img = os.path.join(images_dir, "test")
    train_lbl = os.path.join(labels_dir, "train")
    val_lbl = os.path.join(labels_dir, "val")
    test_lbl = os.path.join(labels_dir, "test")

    ensure_dirs(train_img, val_img, test_img, train_lbl, val_lbl, test_lbl)

    train_ids = read_ids(os.path.join(paths["voc_train"], "Train.txt"))
    val_ids = read_ids(os.path.join(paths["voc_val"], "Validation.txt"))
    test_ids = read_ids(os.path.join(paths["voc_test"], "Test.txt"))

    train_copied = copy_split(
        train_ids,
        set(),
        os.path.join(paths["dataset_dir"], "images", "train"),
        os.path.join(paths["voc_train"], "labels", "Train", "train"),
        train_img,
        train_lbl,
    )

    val_copied = copy_split(
        val_ids,
        train_copied,
        os.path.join(paths["dataset_dir"], "images", "val"),
        os.path.join(paths["voc_val"], "labels", "Validation", "val"),
        val_img,
        val_lbl,
    )

    copy_split(
        test_ids,
        train_copied | val_copied,
        os.path.join(paths["dataset_dir"], "images", "test"),
        os.path.join(paths["voc_test"], "labels", "Test", "test"),
        test_img,
        test_lbl,
    )


if __name__ == "__main__":
    root = project_root()
    parser = argparse.ArgumentParser(description="Build yolo_data from VOC-style splits.")
    parser.add_argument("--voc-train", default=os.path.join(root, "..", "voc_train"),
                        help="Dir with Train.txt and labels/Train/train/")
    parser.add_argument("--voc-val", default=os.path.join(root, "..", "voc_val"),
                        help="Dir with Validation.txt and labels/Validation/val/")
    parser.add_argument("--voc-test", default=os.path.join(root, "..", "voc_test"),
                        help="Dir with Test.txt and labels/Test/test/")
    parser.add_argument("--dataset-dir", default=os.path.join(root, "new_data"),
                        help="Dir with images/train, images/val, images/test")
    parser.add_argument("--yolo-dir", default=os.path.join(root, "yolo_data"),
                        help="Output YOLO dataset directory")
    args = parser.parse_args()
    main(args.voc_train, args.voc_val, args.voc_test, args.dataset_dir, args.yolo_dir)
