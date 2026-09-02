import os
import glob
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torchvision.transforms import functional as F

from faster_rcnn_model import get_model


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_FOLDER = "results"

IMAGE_FOLDER = "dataset/test/enhanced"
XML_FOLDER = "dataset/test/annotations"

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50


# ============================================================
# CLASSES
# ============================================================

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


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# ============================================================
# READ XML
# ============================================================

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


# ============================================================
# GET TEST IMAGES
# ============================================================

images = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:

    images.extend(
        glob.glob(
            os.path.join(
                IMAGE_FOLDER,
                extension
            )
        )
    )

images = sorted(images)


print("==========================================")
print("CONTINUING MODEL COMPARISON")
print("==========================================")

print("Device:", DEVICE)
print("Test Images:", len(images))


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_path):

    model = get_model()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    tested = 0

    with torch.no_grad():

        for image_path in images:

            image_name = os.path.basename(
                image_path
            )

            xml_name = (
                os.path.splitext(image_name)[0]
                + ".xml"
            )

            xml_path = os.path.join(
                XML_FOLDER,
                xml_name
            )

            if not os.path.exists(xml_path):
                continue

            # ----------------------------------------
            # Ground Truth
            # ----------------------------------------

            gt_boxes, gt_labels = read_annotation(
                xml_path
            )

            # ----------------------------------------
            # Image
            # ----------------------------------------

            image = Image.open(
                image_path
            ).convert("RGB")

            image_tensor = F.to_tensor(
                image
            ).to(DEVICE)

            # ----------------------------------------
            # Prediction
            # ----------------------------------------

            output = model(
                [image_tensor]
            )[0]

            pred_boxes = (
                output["boxes"]
                .cpu()
                .numpy()
            )

            pred_labels = (
                output["labels"]
                .cpu()
                .numpy()
            )

            pred_scores = (
                output["scores"]
                .cpu()
                .numpy()
            )

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
                    (
                        box,
                        label,
                        float(score)
                    )
                )

            # ----------------------------------------
            # Match predictions with GT
            # ----------------------------------------

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

                    total_tp += 1

                    matched_gt.add(
                        best_gt
                    )

                else:

                    total_fp += 1

            # ----------------------------------------
            # False Negatives
            # ----------------------------------------

            total_fn += (
                len(gt_boxes)
                -
                len(matched_gt)
            )

            tested += 1

            if tested % 200 == 0:

                print(
                    f"    Tested {tested}/{len(images)}"
                )

    # ========================================================
    # METRICS
    # ========================================================

    if total_tp + total_fp > 0:

        precision = (
            total_tp /
            (total_tp + total_fp)
        )

    else:

        precision = 0


    if total_tp + total_fn > 0:

        recall = (
            total_tp /
            (total_tp + total_fn)
        )

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


    return (
        precision,
        recall,
        f1,
        total_tp,
        total_fp,
        total_fn
    )


# ============================================================
# CONTINUE FROM EPOCH 2
# ============================================================

all_results = []


for epoch in range(1,2):

    model_path = os.path.join(
        MODEL_FOLDER,
        f"model_epoch_{epoch}.pth"
    )

    if not os.path.exists(model_path):

        print(
            f"\nEpoch {epoch}: Model not found"
        )

        continue


    print("\n==========================================")
    print(f"TESTING MODEL EPOCH {epoch}")
    print("==========================================")


    precision, recall, f1, tp, fp, fn = evaluate_model(
        model_path
    )


    all_results.append(
        (
            epoch,
            precision,
            recall,
            f1,
            tp,
            fp,
            fn
        )
    )


    print(
        f"\nEpoch {epoch} completed."
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"TP        : {tp}"
    )

    print(
        f"FP        : {fp}"
    )

    print(
        f"FN        : {fn}"
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n==========================================")
print("EPOCH 2 - 5 COMPARISON")
print("==========================================")

print(
    "\nEpoch | Precision | Recall | F1 Score"
)

print(
    "------------------------------------------"
)


for result in all_results:

    epoch, precision, recall, f1, tp, fp, fn = result

    print(
        f"{epoch:5d} | "
        f"{precision:.4f}     | "
        f"{recall:.4f} | "
        f"{f1:.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

if all_results:

    best = max(
        all_results,
        key=lambda x: x[3]
    )

    best_epoch = best[0]
    best_precision = best[1]
    best_recall = best[2]
    best_f1 = best[3]

    print("\n==========================================")
    print("BEST MODEL FROM EPOCH 2 - 5")
    print("==========================================")

    print(
        "Best Epoch:",
        best_epoch
    )

    print(
        f"Precision: {best_precision:.4f}"
    )

    print(
        f"Recall:    {best_recall:.4f}"
    )

    print(
        f"F1 Score:  {best_f1:.4f}"
    )

    print(
        f"Model: results/model_epoch_{best_epoch}.pth"
    )


print("\n==========================================")
print("CONTINUED COMPARISON COMPLETED")
print("==========================================")