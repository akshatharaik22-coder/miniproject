import os
import sys
import warnings
import xml.etree.ElementTree as ET

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT, "dataset", "validation", "enhanced"
)

XML_FOLDER = os.path.join(
    PROJECT_ROOT, "dataset", "validation", "annotations"
)

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT, "results", "retrained"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT, "results", "validation_report"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Allow imports from the model directory.
MODEL_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

if MODEL_DIRECTORY not in sys.path:
    sys.path.insert(0, MODEL_DIRECTORY)

from faster_rcnn_model import get_model


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 10

# Foreground class IDs must match the model's training labels.
CLASSES = {
    "person": 1,
    "chair": 2,
    "bicycle": 3,
    "car": 4,
    "bus": 5,
    "motorbike": 6,
    "dog": 7,
    "diningtable": 8,
    "sofa": 9,
}

CLASS_NAMES = {
    class_id: class_name
    for class_name, class_id in CLASSES.items()
}

# Index 0 is reserved for the detector's background class.
# Index 10 is used only for unmatched detections/ground truth
# in our diagnostic confusion matrix.
BACKGROUND_ID = NUM_CLASSES

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50

CHECKPOINTS = [
    os.path.join(
        MODEL_FOLDER,
        f"model_epoch_{i}.pth"
    )
    for i in range(1, 6)
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("VALIDATION CONFIGURATION")
print("=" * 65)
print("Device:", DEVICE)
print("Validation images:", IMAGE_FOLDER)
print("Annotations:", XML_FOLDER)
print("Checkpoints:", MODEL_FOLDER)
print("Output folder:", OUTPUT_FOLDER)
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("IoU threshold:", IOU_THRESHOLD)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.ToTensor()


# ============================================================
# CLASS NAME NORMALIZATION
# ============================================================

def normalize_class_name(name):
    """
    Normalize XML class names to match the model's class mapping.

    Examples:
        'dining table' -> 'diningtable'
        'DiningTable'  -> 'diningtable'
        'motorbike'    -> 'motorbike'
    """

    return (
        (name or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )


# ============================================================
# VALIDATION DATASET
# ============================================================

class ValidationDataset(Dataset):

    def __init__(self):

        if not os.path.isdir(IMAGE_FOLDER):
            raise FileNotFoundError(
                f"Validation image folder not found: "
                f"{IMAGE_FOLDER}"
            )

        if not os.path.isdir(XML_FOLDER):
            raise FileNotFoundError(
                f"Validation annotation folder not found: "
                f"{XML_FOLDER}"
            )

        self.images = sorted([
            name
            for name in os.listdir(IMAGE_FOLDER)
            if name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        if not self.images:
            raise RuntimeError(
                "No validation images were found."
            )

        self.valid_images = []

        for image_name in self.images:

            xml_name = (
                os.path.splitext(image_name)[0] + ".xml"
            )

            xml_path = os.path.join(
                XML_FOLDER,
                xml_name
            )

            if not os.path.isfile(xml_path):

                warnings.warn(
                    f"Missing XML annotation for "
                    f"{image_name}; image excluded."
                )

                continue

            self.valid_images.append(image_name)

        if not self.valid_images:
            raise RuntimeError(
                "No images with matching XML annotations "
                "were found."
            )

        print("\nDataset summary")
        print("-" * 40)
        print("Images discovered:", len(self.images))
        print(
            "Images with annotations:",
            len(self.valid_images)
        )
        print(
            "Images excluded for missing annotations:",
            len(self.images) - len(self.valid_images)
        )

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, index):

        image_name = self.valid_images[index]

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_name
        )

        xml_path = os.path.join(
            XML_FOLDER,
            os.path.splitext(image_name)[0] + ".xml"
        )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        width, height = image.size

        image_tensor = transform(image)

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            raise ValueError(
                f"Invalid XML annotation: {xml_path}"
            ) from exc

        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):

            name_element = obj.find("name")
            box_element = obj.find("bndbox")

            if name_element is None or box_element is None:
                continue

            class_name = normalize_class_name(
                name_element.text
            )

            # Ignore classes not included in this detector.
            if class_name not in CLASSES:
                continue

            try:

                xmin = float(
                    box_element.findtext("xmin")
                )

                ymin = float(
                    box_element.findtext("ymin")
                )

                xmax = float(
                    box_element.findtext("xmax")
                )

                ymax = float(
                    box_element.findtext("ymax")
                )

            except (TypeError, ValueError) as exc:

                raise ValueError(
                    f"Invalid bounding-box coordinates "
                    f"in {xml_path}"
                ) from exc

            # Clip bounding boxes to the image dimensions.
            xmin = max(0.0, min(xmin, width))
            xmax = max(0.0, min(xmax, width))

            ymin = max(0.0, min(ymin, height))
            ymax = max(0.0, min(ymax, height))

            # Reject invalid bounding boxes.
            if xmax <= xmin or ymax <= ymin:

                warnings.warn(
                    f"Invalid bounding box in {xml_path}; "
                    f"object '{class_name}' skipped."
                )

                continue

            boxes.append([
                xmin,
                ymin,
                xmax,
                ymax
            ])

            labels.append(
                CLASSES[class_name]
            )

        if boxes:

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64
            )

        else:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image_tensor, target, image_name


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# INTERSECTION OVER UNION
# ============================================================

def box_iou(box1, box2):
    """
    Calculate IoU between two boxes in xyxy format.

    Each box is represented as:
        [xmin, ymin, xmax, ymax]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area1 = (
        max(0.0, box1[2] - box1[0])
        * max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0])
        * max(0.0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# PREDICTION PREPARATION
# ============================================================

def prepare_predictions(
    prediction,
    confidence_threshold=CONFIDENCE_THRESHOLD
):
    """
    Move predictions to CPU, apply the confidence threshold,
    and sort the remaining predictions by descending score.
    """

    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()

    keep = scores >= confidence_threshold

    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    order = torch.argsort(
        scores,
        descending=True
    )

    return (
        boxes[order],
        labels[order],
        scores[order]
    )


# ============================================================
# CLASS-AWARE METRIC MATCHING
# ============================================================

def calculate_class_statistics(
    prediction,
    target,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    iou_threshold=IOU_THRESHOLD
):
    """
    Calculate class-aware TP, FP, and FN counts.

    Matching rules:
      1. Predictions below the confidence threshold are removed.
      2. Predictions are processed in descending confidence order.
      3. A prediction can match only a ground-truth box of the
         same class.
      4. A ground-truth box can be matched only once per class.
      5. A matched prediction with sufficient IoU is a TP.
      6. An unmatched prediction is an FP.
      7. An unmatched ground-truth object is an FN.

    This function calculates the statistics used for
    precision, recall, and F1-score.
    """

    pred_boxes, pred_labels, _ = prepare_predictions(
        prediction,
        confidence_threshold
    )

    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = target["labels"].detach().cpu()

    class_stats = {
        class_id: {
            "TP": 0,
            "FP": 0,
            "FN": 0
        }
        for class_id in CLASS_NAMES
    }

    # Track matched ground-truth indices independently
    # for each foreground class.
    matched_gt = set()

    for pred_box, pred_label_tensor in zip(
        pred_boxes,
        pred_labels
    ):

        pred_label = int(pred_label_tensor)

        # Ignore labels outside the supported class mapping.
        if pred_label not in class_stats:
            continue

        best_iou = 0.0
        best_gt_index = -1

        # Only consider ground-truth boxes of the same class.
        for gt_index, gt_box in enumerate(gt_boxes):

            if gt_index in matched_gt:
                continue

            true_label = int(gt_labels[gt_index])

            if true_label != pred_label:
                continue

            current_iou = box_iou(
                pred_box.tolist(),
                gt_box.tolist()
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index

        if (
            best_gt_index >= 0
            and best_iou >= iou_threshold
        ):

            class_stats[pred_label]["TP"] += 1
            matched_gt.add(best_gt_index)

        else:

            # Includes incorrect-class predictions.
            class_stats[pred_label]["FP"] += 1

    # Count every unmatched ground-truth object as an FN.
    for gt_index, gt_label_tensor in enumerate(gt_labels):

        if gt_index in matched_gt:
            continue

        true_label = int(gt_label_tensor)

        if true_label in class_stats:
            class_stats[true_label]["FN"] += 1

    return class_stats


# ============================================================
# DIAGNOSTIC CONFUSION MATRIX MATCHING
# ============================================================

def calculate_confusion_matrix(
    prediction,
    target,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    iou_threshold=IOU_THRESHOLD
):
    """
    Create a diagnostic object-detection confusion matrix.

    Rows:
        Ground-truth classes.

    Columns:
        Predicted classes.

    Index 10 represents unmatched detections or missed objects.

    Unlike the class-aware metrics, this diagnostic matrix
    allows a spatially matched prediction to have a different
    class from its ground-truth object. This helps visualize
    wrong-class predictions.

    Predictions are processed by descending confidence.
    Each ground-truth object can be matched at most once.
    """

    pred_boxes, pred_labels, _ = prepare_predictions(
        prediction,
        confidence_threshold
    )

    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = target["labels"].detach().cpu()

    confusion = torch.zeros(
        (NUM_CLASSES + 1, NUM_CLASSES + 1),
        dtype=torch.int64
    )

    matched_gt = set()

    for pred_box, pred_label_tensor in zip(
        pred_boxes,
        pred_labels
    ):

        pred_label = int(pred_label_tensor)

        if pred_label not in CLASS_NAMES:
            continue

        best_iou = 0.0
        best_gt_index = -1

        # For diagnostic class-confusion analysis, search
        # across all unmatched ground-truth classes.
        for gt_index, gt_box in enumerate(gt_boxes):

            if gt_index in matched_gt:
                continue

            current_iou = box_iou(
                pred_box.tolist(),
                gt_box.tolist()
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index

        if (
            best_gt_index >= 0
            and best_iou >= iou_threshold
        ):

            true_label = int(
                gt_labels[best_gt_index]
            )

            # Record correct or incorrect class prediction.
            confusion[true_label, pred_label] += 1

            matched_gt.add(best_gt_index)

        else:

            # Unmatched prediction: background/unmatched row.
            confusion[BACKGROUND_ID, pred_label] += 1

    # Unmatched ground-truth objects are recorded in the
    # background/unmatched column.
    for gt_index, gt_label_tensor in enumerate(gt_labels):

        if gt_index in matched_gt:
            continue

        true_label = int(gt_label_tensor)

        confusion[
            true_label,
            BACKGROUND_ID
        ] += 1

    return confusion


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_metrics(class_stats):
    """
    Calculate micro-averaged precision, recall, and F1-score
    from the accumulated TP, FP, and FN counts.
    """

    total_tp = sum(
        stats["TP"]
        for stats in class_stats.values()
    )

    total_fp = sum(
        stats["FP"]
        for stats in class_stats.values()
    )

    total_fn = sum(
        stats["FN"]
        for stats in class_stats.values()
    )

    precision = safe_divide(
        total_tp,
        total_tp + total_fp
    )

    recall = safe_divide(
        total_tp,
        total_tp + total_fn
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

dataset = ValidationDataset()

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print("\nTotal validation images:", len(dataset))


# ============================================================
# EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_checkpoint(
    checkpoint_path,
    model_number
):

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("\n" + "=" * 65)
    print(f"EVALUATING MODEL {model_number}")
    print("=" * 65)

    model = get_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    # Standard TorchMetrics detection mAP calculation.
    map_metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95
        ]
    )

    accumulated_stats = {
        class_id: {
            "TP": 0,
            "FP": 0,
            "FN": 0
        }
        for class_id in CLASS_NAMES
    }

    confusion = torch.zeros(
        (NUM_CLASSES + 1, NUM_CLASSES + 1),
        dtype=torch.int64
    )

    with torch.inference_mode():

        for batch_index, (
            images,
            targets,
            names
        ) in enumerate(
            dataloader,
            start=1
        ):

            images_device = [
                image.to(DEVICE)
                for image in images
            ]

            outputs = model(images_device)

            metric_predictions = []
            metric_targets = []

            for output, target in zip(
                outputs,
                targets
            ):

                prediction_cpu = {
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu()
                }

                target_cpu = {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu()
                }

                # mAP uses confidence-ranked predictions.
                # Do not apply the fixed 0.50 confidence
                # threshold used for the other metrics.
                metric_predictions.append(
                    prediction_cpu
                )

                metric_targets.append(
                    target_cpu
                )

                # Class-aware matching for TP, FP and FN.
                image_stats = calculate_class_statistics(
                    prediction_cpu,
                    target_cpu
                )

                for class_id in CLASS_NAMES:

                    for key in ("TP", "FP", "FN"):

                        accumulated_stats[class_id][key] += (
                            image_stats[class_id][key]
                        )

                # Separate diagnostic confusion matrix.
                image_confusion = calculate_confusion_matrix(
                    prediction_cpu,
                    target_cpu
                )

                confusion += image_confusion

            map_metric.update(
                metric_predictions,
                metric_targets
            )

            if (
                batch_index % 100 == 0
                or batch_index == len(dataloader)
            ):

                print(
                    f"Model {model_number}: "
                    f"{batch_index}/{len(dataloader)} images"
                )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    overall = calculate_metrics(
        accumulated_stats
    )

    map_results = map_metric.compute()

    results = {
        "Model": model_number,
        "Checkpoint": os.path.basename(
            checkpoint_path
        ),
        "Images": len(dataset),
        "TP": overall["TP"],
        "FP": overall["FP"],
        "FN": overall["FN"],
        "Precision": overall["Precision"],
        "Recall": overall["Recall"],
        "F1": overall["F1"],
        "mAP50": float(map_results["map_50"]),
        "mAP50_95": float(map_results["map"])
    }

    # --------------------------------------------------------
    # CLASS-WISE METRICS
    # --------------------------------------------------------

    class_rows = []

    for class_id, class_name in CLASS_NAMES.items():

        stats = accumulated_stats[class_id]

        precision = safe_divide(
            stats["TP"],
            stats["TP"] + stats["FP"]
        )

        recall = safe_divide(
            stats["TP"],
            stats["TP"] + stats["FN"]
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall
        )

        class_rows.append({
            "Model": model_number,
            "Class": class_name,
            "TP": stats["TP"],
            "FP": stats["FP"],
            "FN": stats["FN"],
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    del model
    del map_metric

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nEvaluation results:")

    for key, value in results.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return results, class_rows, confusion


# ============================================================
# EVALUATE AVAILABLE CHECKPOINTS
# ============================================================

all_results = []
all_class_results = []

best_model = None
best_f1 = -1.0
best_confusion = None

for model_number, checkpoint_path in enumerate(
    CHECKPOINTS,
    start=1
):

    if not os.path.isfile(checkpoint_path):

        print(
            f"\nSkipping Model {model_number}: "
            f"checkpoint not found:"
        )

        print(checkpoint_path)

        continue

    results, class_rows, confusion = evaluate_checkpoint(
        checkpoint_path,
        model_number
    )

    all_results.append(results)
    all_class_results.extend(class_rows)

    # Select the model with the highest fixed-threshold
    # overall F1-score.
    if results["F1"] > best_f1:

        best_f1 = results["F1"]
        best_model = model_number

        best_confusion = confusion.clone()


if not all_results:

    raise RuntimeError(
        "No checkpoints were evaluated. "
        "Check MODEL_FOLDER and checkpoint filenames."
    )


# ============================================================
# SAVE CSV RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

class_df = pd.DataFrame(all_class_results)

results_csv = os.path.join(
    OUTPUT_FOLDER,
    "model_comparison.csv"
)

class_csv = os.path.join(
    OUTPUT_FOLDER,
    "class_wise_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)

class_df.to_csv(
    class_csv,
    index=False
)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df[
        [
            "Model",
            "Precision",
            "Recall",
            "F1",
            "mAP50",
            "mAP50_95"
        ]
    ].to_string(index=False)
)

print("\nBest model by fixed-threshold F1-score:", best_model)

print("\nSaved CSV:", results_csv)
print("Saved CSV:", class_csv)


# ============================================================
# GRAPH 1: PRECISION, RECALL, AND F1-SCORE
# ============================================================

plot_df = (
    results_df
    .set_index("Model")[
        ["Precision", "Recall", "F1"]
    ]
    * 100
)

ax = plot_df.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Detection Performance Comparison Across Evaluated Models"
)

ax.set_xlabel("Model checkpoint")
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 100)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "metrics_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2: mAP COMPARISON
# ============================================================

map_df = (
    results_df
    .set_index("Model")[
        ["mAP50", "mAP50_95"]
    ]
    * 100
)

ax = map_df.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Mean Average Precision Comparison Across Evaluated Models"
)

ax.set_xlabel("Model checkpoint")
ax.set_ylabel("mAP (%)")
ax.set_ylim(0, 100)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "map_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 3: CLASS-WISE F1-SCORE
# ============================================================

best_class_df = class_df[
    class_df["Model"] == best_model
].copy()

best_class_df["F1_Percentage"] = (
    best_class_df["F1"] * 100
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=best_class_df,
    x="Class",
    y="F1_Percentage"
)

plt.title(
    f"Class-Wise F1-Score of Selected Model (Model {best_model})"
)

plt.xlabel("Object class")
plt.ylabel("F1-score (%)")
plt.ylim(0, 100)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "class_wise_f1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 4: CONFUSION MATRIX
# ============================================================

class_labels = [
    CLASS_NAMES[class_id]
    for class_id in range(1, NUM_CLASSES)
]

class_labels.append("Background / Unmatched")

confusion_array = best_confusion.numpy()

# Exclude detector background index 0.
# Display the nine foreground classes and the diagnostic
# unmatched category at index 10.
display_confusion = confusion_array[1:, 1:]

plt.figure(figsize=(12, 10))

sns.heatmap(
    display_confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title(
    f"Detection Confusion Matrix of Selected Model (Model {best_model})"
)

plt.xlabel("Predicted class / unmatched detection")
plt.ylabel("Ground-truth class / unmatched object")

plt.xticks(
    rotation=35,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("Models evaluated:", len(all_results))
print("Best model by F1-score:", best_model)
print("Best F1-score:", f"{best_f1 * 100:.2f}%")
print("Output folder:", OUTPUT_FOLDER)

print("\nGenerated files:")

generated_files = [
    "model_comparison.csv",
    "class_wise_results.csv",
    "metrics_comparison.png",
    "map_comparison.png",
    "class_wise_f1.png",
    "confusion_matrix.png"
]

for filename in generated_files:

    filepath = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    print(
        f"{filename}: "
        f"{'Created' if os.path.isfile(filepath) else 'Not found'}"
    )

print("\nAll available checkpoints have been evaluated.")