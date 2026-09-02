import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ==========================================
# Project Root
# ==========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


# ==========================================
# NEXT TRAINING DATASET
# ==========================================

IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train",
    "enhanced"
)

XML_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train",
    "annotations"
)


# ==========================================
# Training Range
# ==========================================
# First 2000 images were already used.
# Now use images 2001 - 4000.

START_INDEX = 2000
END_INDEX = 4000


# ==========================================
# Classes
# ==========================================

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


# ==========================================
# Image Transform
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor()
])


# ==========================================
# Dataset Class
# ==========================================

class VOCDataset(Dataset):

    def __init__(self):

        all_images = sorted([
            f for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Take only images 2001 - 4000
        self.images = all_images[
            START_INDEX:END_INDEX
        ]

        print("Total images available:", len(all_images))
        print(
            "Images selected for this training:",
            len(self.images)
        )

        print(
            "Training range:",
            START_INDEX + 1,
            "to",
            min(END_INDEX, len(all_images))
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

        image = transform(image)


        # ======================================
        # XML Annotation
        # ======================================

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

            labels.append(
                CLASSES[class_name]
            )


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


        # ======================================
        # Convert to Tensor
        # ======================================

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

            boxes = torch.as_tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.as_tensor(
                labels,
                dtype=torch.int64
            )


        image_id = torch.tensor([
            index
        ])


        area = (
            (boxes[:, 3] - boxes[:, 1])
            *
            (boxes[:, 2] - boxes[:, 0])
        )


        iscrowd = torch.zeros(
            (len(labels),),
            dtype=torch.int64
        )


        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": image_id,

            "area": area,

            "iscrowd": iscrowd
        }


        return image, target


# ==========================================
# Test Dataset
# ==========================================

if __name__ == "__main__":

    dataset = VOCDataset()

    print("\n===================================")
    print("Next Training Dataset")
    print("===================================")

    print(
        "Total Selected Images:",
        len(dataset)
    )


    image, target = dataset[0]


    print(
        "\nImage Shape:",
        image.shape
    )


    print(
        "\nLabels:"
    )

    print(
        target["labels"]
    )


    print(
        "\nBounding Boxes:"
    )

    print(
        target["boxes"]
    )