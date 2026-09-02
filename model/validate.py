import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from faster_rcnn_model import get_model


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "validation",
    "enhanced"
)

XML_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "validation",
    "annotations"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_epoch_5.pth"
)


# =========================================================
# CLASSES
# =========================================================

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

CLASS_NAMES = {
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
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor()
])


# =========================================================
# VALIDATION DATASET
# =========================================================

class ValidationDataset(Dataset):

    def __init__(self):

        self.images = sorted([
            f for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        print(
            "Validation images found:",
            len(self.images)
        )


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        original_image = image.copy()

        image = transform(image)


        # -----------------------------------------
        # XML
        # -----------------------------------------

        xml_name = os.path.splitext(
            image_name
        )[0] + ".xml"

        xml_path = os.path.join(
            XML_FOLDER,
            xml_name
        )

        tree = ET.parse(xml_path)
        root = tree.getroot()


        boxes = []
        labels = []


        for obj in root.findall("object"):

            class_name = obj.find("name").text.strip()

            if class_name not in CLASSES:
                continue


            xmin = float(
                obj.find("bndbox/xmin").text
            )

            ymin = float(
                obj.find("bndbox/ymin").text
            )

            xmax = float(
                obj.find("bndbox/xmax").text
            )

            ymax = float(
                obj.find("bndbox/ymax").text
            )


            boxes.append([
                xmin,
                ymin,
                xmax,
                ymax
            ])

            labels.append(
                CLASSES[class_name]
            )


        if len(boxes) == 0:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        else:

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64
            )


        target = {
            "boxes": boxes,
            "labels": labels
        }


        return image, target, image_name


# =========================================================
# COLLATE FUNCTION
# =========================================================

def collate_fn(batch):

    return tuple(zip(*batch))


# =========================================================
# IOU FUNCTION
# =========================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])


    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )


    intersection = (
        intersection_width *
        intersection_height
    )


    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )


    union = (
        area1 +
        area2 -
        intersection
    )


    if union <= 0:
        return 0.0


    return intersection / union


# =========================================================
# LOAD DATASET
# =========================================================

dataset = ValidationDataset()

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading Faster R-CNN model...")

model = get_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)

model.eval()

print("Model loaded successfully!")


# =========================================================
# VALIDATION
# =========================================================

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.50


total_ground_truth = 0
total_predictions = 0
total_true_positive = 0
total_false_positive = 0
total_false_negative = 0


class_results = {}

for class_id in CLASS_NAMES:

    class_results[class_id] = {
        "ground_truth": 0,
        "predictions": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0
    }


print("\n========================================")
print("STARTING VALIDATION")
print("========================================")


with torch.no_grad():

    for batch_index, (images, targets, image_names) in enumerate(
        dataloader
    ):

        images = [
            image.to(DEVICE)
            for image in images
        ]


        outputs = model(images)


        for target, output in zip(
            targets,
            outputs
        ):

            gt_boxes = target["boxes"]
            gt_labels = target["labels"]


            pred_boxes = output["boxes"].cpu()
            pred_labels = output["labels"].cpu()
            pred_scores = output["scores"].cpu()


            # -----------------------------------------
            # Apply confidence threshold
            # -----------------------------------------

            keep = pred_scores >= SCORE_THRESHOLD

            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]


            total_ground_truth += len(
                gt_labels
            )

            total_predictions += len(
                pred_labels
            )


            # -----------------------------------------
            # Per-image matching
            # -----------------------------------------

            matched_gt = set()


            for pred_index in range(
                len(pred_boxes)
            ):

                pred_box = pred_boxes[pred_index]
                pred_label = int(
                    pred_labels[pred_index]
                )


                best_iou = 0
                best_gt_index = -1


                for gt_index in range(
                    len(gt_boxes)
                ):

                    if gt_index in matched_gt:
                        continue


                    gt_label = int(
                        gt_labels[gt_index]
                    )


                    if pred_label != gt_label:
                        continue


                    iou = calculate_iou(
                        pred_box.tolist(),
                        gt_boxes[gt_index].tolist()
                    )


                    if iou > best_iou:

                        best_iou = iou
                        best_gt_index = gt_index


                # -----------------------------------------
                # TRUE POSITIVE
                # -----------------------------------------

                if (
                    best_gt_index >= 0
                    and best_iou >= IOU_THRESHOLD
                ):

                    matched_gt.add(
                        best_gt_index
                    )

                    total_true_positive += 1

                    class_results[
                        pred_label
                    ]["TP"] += 1


                # -----------------------------------------
                # FALSE POSITIVE
                # -----------------------------------------

                else:

                    total_false_positive += 1

                    if pred_label in class_results:

                        class_results[
                            pred_label
                        ]["FP"] += 1


            # -----------------------------------------
            # FALSE NEGATIVES
            # -----------------------------------------

            for gt_index in range(
                len(gt_labels)
            ):

                gt_label = int(
                    gt_labels[gt_index]
                )


                class_results[
                    gt_label
                ]["ground_truth"] += 1


                if gt_index not in matched_gt:

                    total_false_negative += 1

                    class_results[
                        gt_label
                    ]["FN"] += 1


        # -----------------------------------------
        # Progress
        # -----------------------------------------

        if (batch_index + 1) % 100 == 0:

            print(
                f"Validated {batch_index + 1} / "
                f"{len(dataset)} images"
            )


# =========================================================
# OVERALL METRICS
# =========================================================

if total_predictions > 0:

    precision = (
        total_true_positive /
        total_predictions
    )

else:

    precision = 0


if total_ground_truth > 0:

    recall = (
        total_true_positive /
        total_ground_truth
    )

else:

    recall = 0


if precision + recall > 0:

    f1_score = (
        2 *
        precision *
        recall /
        (precision + recall)
    )

else:

    f1_score = 0


# =========================================================
# RESULTS
# =========================================================

print("\n========================================")
print("VALIDATION RESULTS")
print("========================================")

print(
    "Validation Images:",
    len(dataset)
)

print(
    "Ground Truth Objects:",
    total_ground_truth
)

print(
    "Predicted Objects:",
    total_predictions
)

print(
    "True Positives:",
    total_true_positive
)

print(
    "False Positives:",
    total_false_positive
)

print(
    "False Negatives:",
    total_false_negative
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1_score:.4f}"
)


# =========================================================
# CLASS-WISE RESULTS
# =========================================================

print("\n========================================")
print("CLASS-WISE RESULTS")
print("========================================")


for class_id, class_name in CLASS_NAMES.items():

    result = class_results[class_id]

    gt = result["ground_truth"]
    tp = result["TP"]
    fp = result["FP"]
    fn = result["FN"]


    if tp + fp > 0:

        class_precision = (
            tp /
            (tp + fp)
        )

    else:

        class_precision = 0


    if tp + fn > 0:

        class_recall = (
            tp /
            (tp + fn)
        )

    else:

        class_recall = 0


    if (
        class_precision +
        class_recall
    ) > 0:

        class_f1 = (
            2 *
            class_precision *
            class_recall /
            (
                class_precision +
                class_recall
            )
        )

    else:

        class_f1 = 0


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
        f"  Precision    : {class_precision:.4f}"
    )

    print(
        f"  Recall       : {class_recall:.4f}"
    )

    print(
        f"  F1 Score     : {class_f1:.4f}"
    )


print("\n========================================")
print("VALIDATION COMPLETED")
print("========================================")