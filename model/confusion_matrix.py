import os
import torch
import torchvision
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_epoch_5.pth"
)

IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "test",
    "enhanced"
)

XML_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "test",
    "annotations"
)

RESULT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results"
)

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50


# ============================================================
# CLASSES
# ============================================================

CLASSES = {
    "person": 1,
    "chair": 2,
    "bicycle": 3,
    "car": 4,
    "bus": 5,
    "motorbike": 6,
    "dog": 7,
    "diningtable": 8,
    "sofa": 9
}

CLASS_NAMES = [
    "person",
    "chair",
    "bicycle",
    "car",
    "bus",
    "motorbike",
    "dog",
    "diningtable",
    "sofa"
]

# Background = 0
NUM_CLASSES = 10


# ============================================================
# CREATE MODEL
# ============================================================

print("Using Device:", DEVICE)

print("\nLoading Faster R-CNN model...")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None
)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Epoch 5 model loaded successfully.")


# ============================================================
# IOU FUNCTION
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


# ============================================================
# READ GROUND TRUTH
# ============================================================

def read_ground_truth(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text.strip()

        if class_name not in CLASSES:
            continue

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        boxes.append(
            [xmin, ymin, xmax, ymax]
        )

        labels.append(
            CLASSES[class_name]
        )

    return boxes, labels


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=int
)


# ============================================================
# GET TEST IMAGES
# ============================================================

image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print("\nTest Images:", len(image_files))

print("\n==========================================")
print("CREATING CONFUSION MATRIX")
print("MODEL: EPOCH 5")
print("==========================================\n")


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, image_name in enumerate(image_files):

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    xml_name = os.path.splitext(image_name)[0] + ".xml"

    xml_path = os.path.join(
        XML_FOLDER,
        xml_name
    )

    if not os.path.exists(xml_path):
        continue

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        continue

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_tensor = (
        torch.from_numpy(
            image_rgb
        ).permute(2, 0, 1).float() / 255.0
    )

    image_tensor = image_tensor.to(DEVICE)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            [image_tensor]
        )[0]

    pred_boxes = (
        prediction["boxes"]
        .cpu()
        .numpy()
    )

    pred_labels = (
        prediction["labels"]
        .cpu()
        .numpy()
    )

    pred_scores = (
        prediction["scores"]
        .cpu()
        .numpy()
    )

    # Confidence filtering

    keep = pred_scores >= CONFIDENCE_THRESHOLD

    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_boxes, gt_labels = read_ground_truth(
        xml_path
    )

    matched_predictions = set()

    # --------------------------------------------------------
    # Match GT with prediction using IoU
    # --------------------------------------------------------

    for gt_box, gt_label in zip(
        gt_boxes,
        gt_labels
    ):

        best_iou = 0
        best_prediction = -1

        for pred_index, pred_box in enumerate(
            pred_boxes
        ):

            if pred_index in matched_predictions:
                continue

            iou = calculate_iou(
                gt_box,
                pred_box
            )

            if iou > best_iou:

                best_iou = iou
                best_prediction = pred_index

        # ----------------------------------------------------
        # Matched object
        # ----------------------------------------------------

        if (
            best_prediction != -1
            and best_iou >= IOU_THRESHOLD
        ):

            predicted_label = int(
                pred_labels[best_prediction]
            )

            confusion_matrix[
                gt_label,
                predicted_label
            ] += 1

            matched_predictions.add(
                best_prediction
            )

        # ----------------------------------------------------
        # Missed object
        # ----------------------------------------------------

        else:

            # Ground truth → Background
            confusion_matrix[
                gt_label,
                0
            ] += 1

    # --------------------------------------------------------
    # Remaining predictions = false positives
    # --------------------------------------------------------

    for pred_index, pred_label in enumerate(
        pred_labels
    ):

        if pred_index not in matched_predictions:

            confusion_matrix[
                0,
                int(pred_label)
            ] += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (index + 1) % 200 == 0:

        print(
            f"Processed {index + 1}/{len(image_files)}"
        )


# ============================================================
# PRINT MATRIX
# ============================================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================\n")

labels_with_background = [
    "background"
] + CLASS_NAMES

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    confusion_matrix
)


# ============================================================
# SAVE NUMPY MATRIX
# ============================================================

matrix_path = os.path.join(
    RESULT_FOLDER,
    "confusion_matrix_epoch5.npy"
)

np.save(
    matrix_path,
    confusion_matrix
)

print(
    "\nSaved:",
    matrix_path
)


# ============================================================
# DRAW CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(12, 10)
)

plt.imshow(
    confusion_matrix
)

plt.title(
    "Confusion Matrix - Faster R-CNN Epoch 5"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(NUM_CLASSES),
    labels_with_background,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    labels_with_background
)

# Add numbers inside cells

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            confusion_matrix[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

figure_path = os.path.join(
    RESULT_FOLDER,
    "confusion_matrix_epoch5.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nConfusion matrix image saved:"
)

print(
    figure_path
)

print(
    "\nCONFUSION MATRIX COMPLETED"
)