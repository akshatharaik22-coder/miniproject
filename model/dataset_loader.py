import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ===========================
# Folder Paths
# ===========================

IMAGE_FOLDER = "dataset/train/enhanced"
XML_FOLDER = "dataset/train/annotations"

# ===========================
# Classes
# Background = 0
# ===========================

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

# ===========================
# Image Transform
# ===========================

transform = transforms.Compose([
    transforms.ToTensor()
])


class VOCDataset(Dataset):

    def __init__(self):

        self.images = sorted(os.listdir(IMAGE_FOLDER))

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(IMAGE_FOLDER, image_name)

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        xml_name = image_name.replace(".jpg", ".xml")

        xml_path = os.path.join(XML_FOLDER, xml_name)

        tree = ET.parse(xml_path)

        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):

            class_name = obj.find("name").text

            if class_name not in CLASSES:
                continue

            labels.append(CLASSES[class_name])

            xmin = float(obj.find("bndbox/xmin").text)
            ymin = float(obj.find("bndbox/ymin").text)
            xmax = float(obj.find("bndbox/xmax").text)
            ymax = float(obj.find("bndbox/ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([index])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


# ===========================
# Testing
# ===========================

if __name__ == "__main__":

    dataset = VOCDataset()

    print("Total Images :", len(dataset))

    image, target = dataset[0]

    print("Image Shape :", image.shape)

    print("Target Keys :", target.keys())

    print("Labels :", target["labels"])

    print("Boxes :", target["boxes"])