# Vehicle Object Detection with YOLOv8

Object detection for **cars, buses, bicycles, and motorbikes** on a custom PASCAL VOC–style dataset, trained with YOLOv8n.

---

## Quick jump

- [1. Overview](#1-overview)
- [2. Setup & how to run](#2-setup--how-to-run)
- [3. Annotation process](#3-annotation-process)
- [4. Training notes](#4-training-notes)
- [5. Results](#5-results)
- [6. Tools and libraries](#6-tools-and-libraries)
- [7. Future work](#7-future-work)
- [8. LLM usage](#8-llm-usage)

---

## 1. Overview

### Problem

The goal is to train a lightweight detector for four vehicle classes on a small, curated subset of PASCAL VOC–derived data. The main challenge is avoiding overfitting while making the most of transfer learning from COCO-pretrained weights.

### Selected classes


| Class ID | Class name |
| -------- | ---------- |
| 0        | car        |
| 1        | bus        |
| 2        | bicycle    |
| 3        | motorbike  |


### Data statistics {#data-statistics}

Bounding box annotation counts in the YOLO-formatted dataset:


| Class     | Train | Val | Test | Total |
| --------- | ----- | --- | ---- | ----- |
| car       | 180   | 23  | 56   | 259   |
| bus       | 74    | 13  | 19   | 106   |
| bicycle   | 100   | 19  | 13   | 132   |
| motorbike | 69    | 17  | 23   | 109   |


Approximate split sizes: ~200 training images, ~40 validation images, ~40 test images (mutually exclusive splits).

---

## 2. Setup & how to run

### Requirements

- **Python** 3.9 or 3.10
- **Git**

### Install

From the repository root:

```bash
git clone <repository-url>
cd voc-detection

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Weights on Hugging Face

Trained weights (best checkpoint) are on **[1javid/voc_yolov8_exp1](https://huggingface.co/1javid/voc_yolov8_exp1)**. The dataset `yolo_data/` is in this repo. For an **easy run** without training, download the `runs` folder from that Hugging Face repo and place it under `notebooks/runs/` as in the project structure below. The notebooks use fixed paths: training reads data from `yolo_data/` (relative to repo root), and evaluation loads the best weights from `notebooks/runs/detect/voc_detection/exp1/weights/best.pt` and writes outputs under `notebooks/runs/detect/voc_detection/exp1_inference/`.

### Project structure

```
voc-detection/
├── notebooks/
│   ├── 01_training.ipynb         # Train YOLOv8n on yolo_data
│   ├── 02_evaluation.ipynb       # Evaluate & visualize (test split, metrics, boxes)
│   └── runs/                     # ← From Hugging Face voc_yolov8_exp1 (weights + inference outputs)
│       └── detect/voc_detection/
│           ├── exp1/weights/     # best.pt (training checkpoint)
│           └── exp1_inference/   # metrics, plots, comparisons
├── yolo_data/
│   ├── data.yaml                 # Dataset config (paths, class names)
│   ├── images/{train,val,test}/  # Images per split
│   └── labels/{train,val,test}/  # YOLO .txt labels per split
├── utils/
│   ├── 01_filter_images.py       # (Optional) Sample VOC subset; --voc-root, --out-dir
│   ├── 02_yolo_format.py         # (Optional) Build yolo_data from CVAT exports; paths via CLI
│   └── 03_compare_gt_pred.py     # Generate GT vs pred comparison images for selected test IDs
├── requirements.txt
└── README.md
```

**Note:** `yolo_data/` is in the repo. `runs/` is gitignored; to run evaluation without training, download `runs` from [voc_yolov8_exp1](https://huggingface.co/1javid/voc_yolov8_exp1) and put it under `notebooks/runs/`. The scripts in `utils/` are for rebuilding the dataset from VOC + CVAT (e.g. after re-annotating).

### Utils (optional)

If rebuilding the dataset from VOC and CVAT exports, run from the repo root with your environment activated:

```bash
# 1. Sample VOC subset into new_data/ (paths configurable via CLI)
python utils/01_filter_images.py --voc-root /path/to/VOCdevkit/VOC2012 --out-dir new_data

# 2. Build yolo_data/ from CVAT exports (voc_train, voc_val, voc_test dirs)
python utils/02_yolo_format.py --voc-train /path/to/voc_train --voc-val /path/to/voc_val --voc-test /path/to/voc_test --dataset-dir new_data --yolo-dir yolo_data

# 3. Regenerate GT vs pred comparison images (see Results section)
python utils/03_compare_gt_pred.py
```

All paths have defaults relative to the repo root (`../VOCdevkit/VOC2012`, `new_data`, `yolo_data`, etc.). Run each script with `--help` to see options.

### How to run

Start Jupyter from the **repository root** so notebook paths resolve: training uses `yolo_data/data.yaml`, evaluation uses `runs/detect/voc_detection/exp1/weights/best.pt` (relative to the notebook dir). With the structure above, both notebooks run without editing paths.


| Step            | What to do                                                                                                                                                                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Train**    | Run `notebooks/01_training.ipynb` → trains on `yolo_data/data.yaml`, saves weights to e.g. `runs/detect/voc_detection/exp1/weights/best.pt`. Or use the [uploaded weights](https://huggingface.co/1javid/voc_yolov8_exp1) to skip training. |
| **2. Evaluate** | Run `notebooks/02_evaluation.ipynb` → point it at your best weights (local or from [Hugging Face](https://huggingface.co/1javid/voc_yolov8_exp1)) and `../yolo_data/data.yaml`; run validation and visualization cells. Metrics and plots go to e.g. `runs/detect/voc_detection/exp1_inference/`. |


**Workflow:** Train with the first notebook, then evaluate with the second. No extra steps.

---

## 3. Annotation process

Annotations were done in **CVAT**. Total time for all three splits (train, validation, test) was **about two hours**.

**Challenges and decisions**

- The hardest part was choosing a **consistent threshold** for when to label an object versus treat it as background: as humans we can often see faint or small vehicles, but the model may not. Deciding where to draw the line (e.g. very small or heavily occluded instances) and staying consistent across images was the main difficulty.
- Decision: Avoid ambiguous cases where objects are almost unseen so that the dataset stays consistent for training and evaluation.

---

## 4. Training notes

### Model

- **YOLOv8n** (Ultralytics), pretrained on COCO, fine-tuned on the 4-class vehicle dataset.

### Hyperparameters (summary)


| Setting                   | Value           | Reason                                                                               |
| ------------------------- | --------------- | ------------------------------------------------------------------------------------ |
| Optimizer                 | AdamW           | Better convergence and regularization on small data than SGD.                        |
| Learning rate (`lr0`)     | 0.001           | Lower than default to avoid catastrophic forgetting when fine-tuning on a small set. |
| Patience (early stopping) | 50              | Tolerates noisy validation loss on small data.                                       |
| Mosaic                    | 1.0             | Strong augmentation to simulate scale and crowding.                                  |
| Horizontal flip           | 0.5             | Doubles effective orientation variety.                                               |
| Vertical flip             | 0.0             | Disabled; vehicles are not upside-down in this use case.                             |
| HSV (h, s, v)             | 0.015, 0.7, 0.4 | Simulates lighting and weather.                                                      |
| Scale / translate         | 0.5, 0.1        | Simulates distance and camera shift.                                                 |


### Training configuration

- **Epochs:** 100 (with early stopping).
- **Image size:** 640 px.
- **Batch size:** 16
- **Hardware:** Training can be run locally (CPU or single GPU) or on Kaggle (GPU). Typical runs complete in a few tens of minutes to under an hour on a single GPU, depending on hardware.

Detailed justification for choosing YOLOv8n over alternatives (e.g. RT-DETR) and full reproduction code are in the sections below.

---

## 5. Results

### Evaluation protocol

All reported metrics and figures are computed on the **test split** (38 images, 106 instances), which is held out from training and validation. Evaluation uses the Ultralytics default settings: **IoU threshold 0.5** for mAP50 and the standard 0.5:0.95 range for mAP50-95. Results are from a **single training run**. Metrics and plots can be reproduced by running `notebooks/02_evaluation.ipynb` with the best checkpoint (local: `runs/detect/voc_detection/exp1/weights/best.pt`, or [Hugging Face](https://huggingface.co/1javid/voc_yolov8_exp1)).

### Metrics (test split)


| Class     | Images | Instances | Precision (Box P) | Recall (R) | mAP50 | mAP50-95 |
| --------- | ------ | --------- | ----------------- | ---------- | ----- | -------- |
| **all**   | 38     | 106       | 0.76              | ~0.56      | ~0.60 | ~0.38    |
| car       | 17     | 55        | ~0.85             | ~0.66      | ~0.80 | ~0.52    |
| bus       | 10     | 18        | ~0.74             | ~0.48      | ~0.49 | ~0.36    |
| bicycle   | 10     | 12        | ~0.70             | 0.75       | 0.70  | ~0.37    |
| motorbike | 11     | 21        | 0.75              | ~0.33      | 0.43  | ~0.26    |


Summary: **Test mAP50 ≈ 0.61**, **Test mAP50-95 ≈ 0.38**.

### Visualizations

**Precision and recall vs. confidence.** When we compare the [Box P curve](notebooks/runs/detect/voc_detection/exp1_inference/BoxP_curve.png) (precision vs. confidence) with the [Box R curve](notebooks/runs/detect/voc_detection/exp1_inference/BoxR_curve.png) (recall vs. confidence), we see that as the confidence threshold increases, precision increases while recall decreases. This is expected: a higher threshold means we treat more detections as background, which reduces false positives (hence higher precision) but increases false negatives—actual objects that we now reject (hence lower recall). The curves reflect this tradeoff.

![BoxP_curve](notebooks/runs/detect/voc_detection/exp1_inference/BoxP_curve.png)

![BoxR_curve](notebooks/runs/detect/voc_detection/exp1_inference/BoxR_curve.png)

**Precision–recall curve.** The [Box P–R curve](notebooks/runs/detect/voc_detection/exp1_inference/BoxPR_curve.png) shows precision vs. recall across confidence thresholds. It summarizes the detection-quality tradeoff: moving along the curve corresponds to changing the confidence threshold. The area under this curve is related to average precision; the shape indicates how much recall we can gain before precision drops sharply.

![BoxPR_curve](notebooks/runs/detect/voc_detection/exp1_inference/BoxPR_curve.png)

**Confusion matrix.** The [normalized confusion matrix](notebooks/runs/detect/voc_detection/exp1_inference/confusion_matrix_normalized.png) shows high error rates for **bus** and **motorbike** classes: a large share of their instances are incorrectly treated as background (missed detections). Conversely, when the model does predict an object, it often predicts **car**—including for true background regions. This pattern aligns with the [annotation distribution](#data-statistics): cars dominate the dataset (≈56% of test instances), so the model is biased toward predicting cars. Bus and motorbike have fewer examples (≈18% and 22% of test instances), leading to lower recall and more confusion with background.

![confusion_matrix_normalized](notebooks/runs/detect/voc_detection/exp1_inference/confusion_matrix_normalized.png)

**Test-set predictions.** Comparing ground-truth labels with predictions on the test set, the model performs best on cars, which matches the data imbalance. It struggles in several scenarios: (1) **weather and lighting changes**, e.g. snowy winter scenes; (2) **occluded or distant buses**; and (3) **clustered or partially visible motorbikes**. Below we compare **ground truth (left)** vs **prediction (right)** for selected test images (evaluation batches 0 and 1). To regenerate these comparison images, run from the repo root: `python utils/03_compare_gt_pred.py` (with your environment activated).

**Batch 0 (2008_004794, 2008_003924, 2008_003777)**

| Image | Ground truth vs prediction |
| ----- | -------------------------- |
| 2008_004794 | ![2008_004794 compare](notebooks/runs/detect/voc_detection/exp1_inference/comparisons/2008_004794_compare.jpg) |
| 2008_003924 | ![2008_003924 compare](notebooks/runs/detect/voc_detection/exp1_inference/comparisons/2008_003924_compare.jpg) |
| 2008_003777 | ![2008_003777 compare](notebooks/runs/detect/voc_detection/exp1_inference/comparisons/2008_003777_compare.jpg) |

**Batch 1 (2008_002384, 2008_002191, 2008_000281)**

| Image | Ground truth vs prediction |
| ----- | -------------------------- |
| 2008_002384 | ![2008_002384 compare](notebooks/runs/detect/voc_detection/exp1_inference/comparisons/2008_002384_compare.jpg) |
| 2008_002191 | ![2008_002191 compare](notebooks/runs/detect/voc_detection/exp1_inference/comparisons/2008_002191_compare.jpg) |
| 2008_000281 | ![2008_000281 compare](notebooks/runs/detect/voc_detection/exp1_inference/comparisons/2008_000281_compare.jpg) |

### Limitations

- **Small test set:** 38 images and 106 instances mean metrics can be noisy; a few difficult images can shift mAP noticeably. Reporting would be stronger with a larger test set or multiple splits.
- **Single run:** Results are from one training run with fixed seeds. No mean ± std or confidence intervals, so we do not quantify variance across runs.
- **Domain:** Data are PASCAL VOC–derived. Performance may differ under different weather, geography, or camera conditions; the model has not been evaluated on out-of-domain data.

### Conclusions

The detector achieves **test mAP50 ≈ 0.61** and **mAP50-95 ≈ 0.38**, with strong precision and recall for **cars** (the majority class) and noticeably lower recall for **bus** and **motorbike**. The main driver is **class imbalance** and limited training data for minority classes, plus failure modes such as weather, occlusion, and small or clustered instances. For car-heavy use cases the model is usable as-is; for reliable bus and motorbike detection, next steps would be collecting more (and more balanced) data, tuning the confidence threshold for the target operating point, or both.

---

## Model selection justification

Two candidate architectures were considered:

- **YOLOv8n** (Ultralytics)
- **RT-DETRv2-L** (Real-Time Detection Transformer)

### Comparative characteristics


| Model       | Input size | Dataset | mAP50–95 (val) | Params (M) | FLOPs (B/G) | Inference speed*                       |
| ----------- | ---------- | ------- | -------------- | ---------- | ----------- | -------------------------------------- |
| YOLOv8n     | 640 px     | COCO    | 37.3           | 3.2        | 8.7 B       | 80.4 ms (CPU ONNX), 0.99 ms (A100 TRT) |
| RT-DETRv2-L | 640 px     | COCO    | 53.4           | 42         | 136 B       | 108 FPS (T4 TRT FP16)                  |


Speeds are from the respective authors; hardware differs.

### Why YOLOv8n

- **Capacity vs. data size:** For a small 4-class dataset, RT-DETRv2-L’s extra parameters and FLOPs add overfitting risk without clear benefit. YOLOv8n’s size matches the problem.
- **Efficiency:** Faster training and inference, suitable for CPU or a single GPU.
- **Tooling:** Ultralytics provides a simple, well-documented pipeline for custom data.
- **Goal:** The project aims for a practical, efficient detector rather than maximum benchmark accuracy.

**Conclusion:** YOLOv8n is chosen as the primary model for this project under the given data and hardware constraints.

---

## 6. Tools and libraries

| Tool / library | Role in the project | Why we use it |
| ----------------- | ------------------- | -------------- |
| **Python** (3.9 / 3.10) | Runtime for training, evaluation, and scripts. | Standard in ML; good ecosystem and readability; matches Ultralytics and OpenCV support. |
| **Ultralytics** (YOLO) | Model definition, training, validation, and inference. | Provides YOLOv8 with a simple API, COCO-pretrained weights, data loading from YAML, and built-in metrics/plots (mAP, PR curves, confusion matrix) so we avoid reimplementing evaluation. |
| **OpenCV** (`opencv-python`) | Image I/O and drawing in comparison scripts. | Reliable for reading/writing images and drawing bounding boxes when generating GT-vs-pred comparison images; minimal dependency. |
| **Jupyter** (`notebook`) | Interactive training and evaluation notebooks. | Lets us run training and evaluation step-by-step, inspect outputs and plots in place, and keep code and narrative together for reproducibility and reporting. |
| **CVAT** | Annotation of bounding boxes (external; not in repo). | Web-based, supports multiple annotators and export to YOLO-style labels; used to create the train/val/test labels. |
| **PyTorch** | Backend for Ultralytics (installed as dependency). | Ultralytics is built on PyTorch; we rely on it for model training and GPU/CPU execution without using it directly in our code. |

---

## 7. Future work

If more time were available, the following would be the next steps, with brief justification for each.

- **Balance classes and add data for bus and motorbike.** Recall is low for these classes partly because they have fewer instances than cars. Collecting more images and annotations for bus and motorbike (and optionally downsampling car-heavy images) would reduce imbalance and likely improve minority-class recall without changing the architecture.

- **Report metrics over multiple seeds.** Running training 3–5 times with different seeds and reporting mean ± std (e.g. for mAP50 and mAP50-95) would show whether the current numbers are stable or optimistic; it would also make the Results section more rigorous for a report or paper.

- **Tune confidence threshold for deployment.** The default threshold may not match the desired precision–recall tradeoff (e.g. fewer false alarms vs. fewer misses). Sweeping the confidence threshold on a validation set and choosing one (or per-class thresholds) for the target use case would make the model more usable in practice.

- **Targeted augmentation for failure modes.** The model struggles with snow, occlusion, and distant objects. Adding or strengthening augmentations that mimic these (e.g. synthetic snow/fog, cutout, stronger scale/zoom) could improve robustness without extra data, at the cost of longer training and tuning.

- **Cross-domain or out-of-distribution evaluation.** Testing on data from different weather, cities, or cameras would quantify how much performance drops; that would set expectations for deployment and justify whether domain adaptation or more diverse training data is needed.

- **Compare with a larger model or RT-DETR.** Running the same data with YOLOv8s/m or RT-DETR and comparing mAP and speed would show whether the current bottleneck is capacity or data; useful if compute is available and the goal is to push accuracy before investing in more annotations.

---

## 8. LLM usage

LLMs were used to refine code in `utils/`, to implement the bounding-box visualization in the evaluation notebook, and to improve the README. Below we document each use case with: **which LLM**, **for what purpose**, **full prompt text**, and **which part of the response was used**.

### 1. Utils: `03_compare_gt_pred.py` (fully AI-generated)

| Field | Content |
| ----- | ------- |
| **Which LLM** | Sonnet 4.6 |
| **For what purpose** | To generate a script that compares ground-truth labels with model predictions for selected test images, producing side-by-side GT vs pred images. |
| **Full prompt text** | *"Create a script that loads test images and their YOLO labels, runs the trained model to get predictions, and saves side-by-side comparison images (ground truth on the left, predictions on the right) for specific image IDs: 2008_004794, 2008_003924, 2008_003777, 2008_002384, 2008_002191, 2008_000281. Save to exp1_inference/comparisons/."* |
| **Which part of the response was used** | The entire script: argument parsing, YOLO label loading, GT box drawing with OpenCV, model inference, and side-by-side image saving. No manual edits beyond path adjustments. |

### 2. Utils: `01_filter_images.py` and `02_yolo_format.py` (AI-refined)

| Field | Content |
| ----- | ------- |
| **Which LLM** | Sonnet 4.6 |
| **For what purpose** | To refine and clean up the existing scripts (style, error handling, type hints, docstrings). |
| **Full prompt text** | *"Review and improve my filter_images.py and yolo_format.py: add type hints, improve error handling, and make the code more readable and maintainable."* |
| **Which part of the response was used** | Suggested refactors for structure and readability; logic and I/O behavior were kept, with minor adjustments to match our project layout. |

### 3. Notebooks: bounding-box drawing in `02_evaluation.ipynb`

| Field | Content |
| ----- | ------- |
| **Which LLM** | Sonnet 4.6 |
| **For what purpose** | To implement the section that draws predicted bounding boxes on sample images and displays them in a grid (e.g. 2×2). |
| **Full prompt text** | *"Add a cell to the evaluation notebook that runs model.predict() on a few sample images, uses r.plot() to draw bounding boxes, and displays the results in a 2×2 grid using matplotlib."* |
| **Which part of the response was used** | The prediction loop, `r.plot()` usage, and matplotlib grid layout; integrated into the existing notebook with our image paths and styling. |

### 4. README refinement

| Field | Content |
| ----- | ------- |
| **Which LLM** | Sonnet 4.6 |
| **For what purpose** | To improve the first draft of the README: fix typos, clarify explanations, and improve structure. |
| **Full prompt text** | *"Fix typos, add a tools/libraries section with reasons for each, and improve the Results and Limitations sections. Also add an LLM usage section documenting where AI was used."* |
| **Which part of the response was used** | Structural changes, new paragraphs, and wording improvements; metrics and project-specific details were preserved and only lightly edited. |