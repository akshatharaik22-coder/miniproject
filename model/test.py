import os
import glob
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torchvision.transforms import functional as F

from faster_rcnn_model import get_model


# =========================================================
# SETTINGS
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "results/model_epoch_5.pth"

IMAGE_FOLDER = "dataset/test/enhanced"
XML_FOLDER = "dataset/test/annotations"

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50


# =========================================================
# CLASSES
# =========================================================

CLASSES = {
    1: "person",
    2: "chair",
    3: "bicycle",
    4: "car",
    5: "bus",
    6: "motorbike",
    7: "dog",
    8: "diningtable",
    9: "sofa"
}


# =========================================================
# IOU
# =========================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# =========================================================
# READ XML
# =========================================================

def read_annotation(xml_path):

    root = ET.parse(xml_path).getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):

        name = obj.find("name").text.strip()

        label = None

        for class_id, class_name in CLASSES.items():

            if name == class_name:
                label = class_id
                break

        if label is None:
            continue

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        boxes.append(
            [xmin, ymin, xmax, ymax]
        )

        labels.append(label)

    return boxes, labels


# =========================================================
# LOAD MODEL
# =========================================================

print("Device:", DEVICE)

model = get_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)

model.eval()

print("Model Loaded Successfully!")


# =========================================================
# TEST IMAGES
# =========================================================

images = sorted(
    glob.glob(
        os.path.join(
            IMAGE_FOLDER,
            "*.jpg"
        )
    )
)

print("Test Images:", len(images))


# =========================================================
# RESULTS STORAGE
# =========================================================

results = {}

for class_id, class_name in CLASSES.items():

    results[class_id] = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "GT": 0
    }


total_images = 0


# =========================================================
# TEST
# =========================================================

print("\n======================================")
print("STARTING TEST")
print("======================================")


with torch.no_grad():

    for index, image_path in enumerate(images):

        image_name = os.path.basename(image_path)

        xml_name = os.path.splitext(
            image_name
        )[0] + ".xml"

        xml_path = os.path.join(
            XML_FOLDER,
            xml_name
        )

        if not os.path.exists(xml_path):
            continue

        # ---------------------------------------------
        # Ground truth
        # ---------------------------------------------

        gt_boxes, gt_labels = read_annotation(
            xml_path
        )

        for label in gt_labels:

            results[label]["GT"] += 1

        # ---------------------------------------------
        # Image
        # ---------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = F.to_tensor(
            image
        ).to(DEVICE)

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        output = model(
            [image_tensor]
        )[0]

        pred_boxes = output["boxes"].cpu().numpy()
        pred_labels = output["labels"].cpu().numpy()
        pred_scores = output["scores"].cpu().numpy()

        # ---------------------------------------------
        # Confidence filtering
        # ---------------------------------------------

        predictions = []

        for box, label, score in zip(
            pred_boxes,
            pred_labels,
            pred_scores
        ):

            if score < CONFIDENCE_THRESHOLD:
                continue

            label = int(label)

            if label not in CLASSES:
                continue

            predictions.append(
                (box, label, float(score))
            )

        # ---------------------------------------------
        # Match predictions with GT
        # ---------------------------------------------

        matched_gt = set()

        for pred_box, pred_label, pred_score in predictions:

            best_iou = 0
            best_gt = -1

            for gt_index, (
                gt_box,
                gt_label
            ) in enumerate(
                zip(gt_boxes, gt_labels)
            ):

                if gt_index in matched_gt:
                    continue

                if pred_label != gt_label:
                    continue

                iou = calculate_iou(
                    pred_box,
                    gt_box
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt = gt_index

            if (
                best_gt >= 0
                and best_iou >= IOU_THRESHOLD
            ):

                results[pred_label]["TP"] += 1

                matched_gt.add(best_gt)

            else:

                results[pred_label]["FP"] += 1

        # ---------------------------------------------
        # False negatives
        # ---------------------------------------------

        for gt_index, gt_label in enumerate(gt_labels):

            if gt_index not in matched_gt:

                results[gt_label]["FN"] += 1

        total_images += 1

        # ---------------------------------------------
        # Progress
        # ---------------------------------------------

        if total_images % 100 == 0:

            print(
                f"Tested {total_images}/{len(images)}"
            )


# =========================================================
# CLASS RESULTS
# =========================================================

print("\n======================================")
print("CLASS-WISE TEST RESULTS")
print("======================================")


total_tp = 0
total_fp = 0
total_fn = 0


for class_id, class_name in CLASSES.items():

    r = results[class_id]

    tp = r["TP"]
    fp = r["FP"]
    fn = r["FN"]
    gt = r["GT"]

    total_tp += tp
    total_fp += fp
    total_fn += fn

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
        ) / (
            precision + recall
        )
    else:
        f1 = 0

    print(
        f"\n{class_name}"
    )

    print(
        f"  Ground Truth : {gt}"
    )

    print(
        f"  TP           : {tp}"
    )

    print(
        f"  FP           : {fp}"
    )

    print(
        f"  FN           : {fn}"
    )

    print(
        f"  Precision    : {precision:.4f}"
    )

    print(
        f"  Recall       : {recall:.4f}"
    )

    print(
        f"  F1 Score     : {f1:.4f}"
    )


# =========================================================
# OVERALL RESULTS
# =========================================================

if total_tp + total_fp > 0:

    overall_precision = (
        total_tp /
        (total_tp + total_fp)
    )

else:

    overall_precision = 0


if total_tp + total_fn > 0:

    overall_recall = (
        total_tp /
        (total_tp + total_fn)
    )

else:

    overall_recall = 0


if (
    overall_precision +
    overall_recall
) > 0:

    overall_f1 = (
        2 *
        overall_precision *
        overall_recall
    ) / (
        overall_precision +
        overall_recall
    )

else:

    overall_f1 = 0


print("\n======================================")
print("OVERALL TEST RESULTS")
print("======================================")

print(
    "Images Tested:",
    total_images
)

print(
    "Total TP:",
    total_tp
)

print(
    "Total FP:",
    total_fp
)

print(
    "Total FN:",
    total_fn
)

print(
    f"Precision: {overall_precision:.4f}"
)

print(
    f"Recall:    {overall_recall:.4f}"
)

print(
    f"F1 Score:  {overall_f1:.4f}"
)

print("\n======================================")
print("TEST COMPLETED")
print("======================================")