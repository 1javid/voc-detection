"""
Generate GT vs prediction comparison images for specific test images.
Run from repository root so paths resolve correctly.

Usage:
  python utils/03_compare_gt_pred.py
"""
from pathlib import Path
import cv2
from ultralytics import YOLO

# Paths (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "yolo_data"
IMAGES_DIR = DATA_ROOT / "images" / "test"
LABELS_DIR = DATA_ROOT / "labels" / "test"
MODEL_PATH = REPO_ROOT / "notebooks" / "runs" / "detect" / "voc_detection" / "exp1" / "weights" / "best.pt"
OUT_DIR = REPO_ROOT / "notebooks" / "runs" / "detect" / "voc_detection" / "exp1_inference" / "comparisons"

CLASS_NAMES = ("car", "bus", "bicycle", "motorbike")
# BGR colors for GT boxes (green tint per class)
GT_COLORS = [
    (0, 200, 0),    # car - green
    (200, 150, 0),  # bus - teal
    (0, 200, 200),  # bicycle - yellow
    (200, 0, 200),  # motorbike - magenta
]


def load_gt_boxes(label_path: Path, img_width: int, img_height: int):
    """Load YOLO-format labels and return list of (class_id, x1, y1, x2, y2) in pixel coords."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        x1 = int((xc - w / 2) * img_width)
        y1 = int((yc - h / 2) * img_height)
        x2 = int((xc + w / 2) * img_width)
        y2 = int((yc + h / 2) * img_height)
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def draw_gt(image, boxes, thickness=2):
    """Draw ground-truth boxes on image (BGR)."""
    out = image.copy()
    for cls_id, x1, y1, x2, y2 in boxes:
        color = GT_COLORS[cls_id % len(GT_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        cv2.putText(
            out, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    return out


def main():
    """Generate side-by-side GT vs prediction images for fixed test IDs; save to OUT_DIR."""
    # Batch 0: 2008_004794, 2008_003924, 2008_003777
    # Batch 1: 2008_002384, 2008_002191, 2008_000281
    image_ids = [
        "2008_004794", "2008_003924", "2008_003777",
        "2008_002384", "2008_002191", "2008_000281",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    for im_id in image_ids:
        img_path = IMAGES_DIR / f"{im_id}.jpg"
        label_path = LABELS_DIR / f"{im_id}.txt"
        if not img_path.exists():
            print(f"Skip {im_id}: image not found")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip {im_id}: failed to load image")
            continue
        h, w = img.shape[:2]

        gt_boxes = load_gt_boxes(label_path, w, h)
        img_gt = draw_gt(img, gt_boxes)

        results = model.predict(source=str(img_path), save=False, verbose=False)
        img_pred = results[0].plot()  # BGR numpy array

        # Resize pred to same height if different (e.g. letterbox)
        if img_pred.shape[0] != h or img_pred.shape[1] != w:
            img_pred = cv2.resize(img_pred, (w, h), interpolation=cv2.INTER_LINEAR)

        # Add labels above the two panels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_gt, "Ground truth", (10, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_gt, "Ground truth", (10, 26), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_pred, "Prediction", (10, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_pred, "Prediction", (10, 26), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        combined = cv2.hconcat([img_gt, img_pred])
        out_path = OUT_DIR / f"{im_id}_compare.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
