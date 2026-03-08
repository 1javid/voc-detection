"""
Sample a small VOC2012 subset for downstream conversion to YOLO.

Reads VOC ImageSets/Main/*.txt. For each target class: up to 50 train,
up to 10 val, up to 10 test (from *_val.txt). Copies JPEGs and XMLs
into out_dir/images/{train,val,test}/ and out_dir/labels/{train,val,test}/.
"""

import argparse
import os
import shutil


def project_root():
    """Return the absolute path of the repository root (parent of utils/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_paths(voc_root, out_dir):
    """Build a dict of paths for VOC ImageSets, JPEGs, annotations, and output dirs."""
    return {
        "main_dir": os.path.join(voc_root, "ImageSets", "Main"),
        "jpeg_dir": os.path.join(voc_root, "JPEGImages"),
        "ann_dir": os.path.join(voc_root, "Annotations"),
        "out_dir": out_dir,
        "out_images_dir": os.path.join(out_dir, "images"),
        "out_labels_dir": os.path.join(out_dir, "labels"),
    }


def ensure_dirs(*dirs):
    """Create each directory if it does not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def parse_class_and_set(filename):
    """Parse e.g. car_train.txt -> (class_name, split_name)."""
    root, _ = os.path.splitext(filename)
    if "_" not in root:
        return None
    class_name, split_name = root.split("_", 1)
    return class_name, split_name


def iter_positive_ids(list_file):
    """Yield image IDs marked as positive ('1') in a VOC classification list file."""
    with open(list_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "1":
                yield parts[0]


def copy_image_and_xml(image_id, src_jpeg_dir, src_ann_dir, dst_img_dir, dst_lbl_dir):
    """Copy one JPEG and its VOC XML to the destination. Skip if either is missing."""
    src_img = os.path.join(src_jpeg_dir, image_id + ".jpg")
    src_xml = os.path.join(src_ann_dir, image_id + ".xml")
    if not os.path.exists(src_img) or not os.path.exists(src_xml):
        return
    shutil.copy2(src_img, os.path.join(dst_img_dir, image_id + ".jpg"))
    shutil.copy2(src_xml, os.path.join(dst_lbl_dir, image_id + ".xml"))


def main(voc_root, out_dir):
    """Sample VOC subset into train/val/test under out_dir. Splits are mutually exclusive."""
    paths = build_paths(voc_root, out_dir)
    train_img = os.path.join(paths["out_images_dir"], "train")
    val_img = os.path.join(paths["out_images_dir"], "val")
    test_img = os.path.join(paths["out_images_dir"], "test")
    train_lbl = os.path.join(paths["out_labels_dir"], "train")
    val_lbl = os.path.join(paths["out_labels_dir"], "val")
    test_lbl = os.path.join(paths["out_labels_dir"], "test")

    ensure_dirs(paths["out_dir"], paths["out_images_dir"], paths["out_labels_dir"],
               train_img, val_img, test_img, train_lbl, val_lbl, test_lbl)

    classes = ["car", "bicycle", "motorbike", "bus"]
    assigned_train = set()
    assigned_val = set()
    assigned_test = set()
    train_counts = {c: 0 for c in classes}
    val_counts = {c: 0 for c in classes}
    test_counts = {c: 0 for c in classes}

    for filename in os.listdir(paths["main_dir"]):
        parsed = parse_class_and_set(filename)
        if parsed is None:
            continue
        class_name, split_name = parsed
        if class_name not in classes:
            continue

        list_file = os.path.join(paths["main_dir"], filename)
        for image_id in iter_positive_ids(list_file):
            if image_id in assigned_train or image_id in assigned_val or image_id in assigned_test:
                continue

            if split_name == "train":
                if train_counts[class_name] >= 50:
                    continue
                copy_image_and_xml(image_id, paths["jpeg_dir"], paths["ann_dir"], train_img, train_lbl)
                assigned_train.add(image_id)
                train_counts[class_name] += 1

            elif split_name == "val":
                if image_id in assigned_train:
                    continue
                if val_counts[class_name] < 10 and image_id not in assigned_val and image_id not in assigned_test:
                    copy_image_and_xml(image_id, paths["jpeg_dir"], paths["ann_dir"], val_img, val_lbl)
                    assigned_val.add(image_id)
                    val_counts[class_name] += 1
                elif test_counts[class_name] < 10 and image_id not in assigned_val and image_id not in assigned_test:
                    copy_image_and_xml(image_id, paths["jpeg_dir"], paths["ann_dir"], test_img, test_lbl)
                    assigned_test.add(image_id)
                    test_counts[class_name] += 1


if __name__ == "__main__":
    root = project_root()
    parser = argparse.ArgumentParser(description="Sample VOC subset into train/val/test.")
    parser.add_argument("--voc-root", default=os.path.join(root, "..", "VOCdevkit", "VOC2012"),
                        help="VOC2012 root (ImageSets, JPEGImages, Annotations)")
    parser.add_argument("--out-dir", default=os.path.join(root, "new_data"),
                        help="Output directory for images and labels")
    args = parser.parse_args()
    main(args.voc_root, args.out_dir)
