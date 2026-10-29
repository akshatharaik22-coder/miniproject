import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

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
# CLASSES
# Background = 0
# Nine object classes = 1 to 9
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
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor()
])


# ==========================================
# DATASET CLASS
# ==========================================

class VOCDataset(Dataset):

    def __init__(self, start_index=0, end_index=2000):

        if not os.path.isdir(IMAGE_FOLDER):
            raise FileNotFoundError(
                f"Image folder not found: {IMAGE_FOLDER}"
            )

        if not os.path.isdir(XML_FOLDER):
            raise FileNotFoundError(
                f"Annotation folder not found: {XML_FOLDER}"
            )

        if start_index < 0 or end_index <= start_index:
            raise ValueError(
                "Invalid dataset range. "
                "Ensure start_index >= 0 and end_index > start_index."
            )

        # Get all images in sorted order.
        all_images = sorted([
            filename
            for filename in os.listdir(IMAGE_FOLDER)
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        total_images = len(all_images)

        if end_index > total_images:
            raise ValueError(
                f"Requested images {start_index + 1} to {end_index}, "
                f"but only {total_images} images are available."
            )

        # Select the requested group.
        self.images = all_images[start_index:end_index]

        self.start_index = start_index
        self.end_index = end_index

        print("\n" + "=" * 50)
        print("DATASET INITIALIZATION")
        print("=" * 50)
        print("Total available images:", total_images)
        print("Selected images:", len(self.images))
        print(
            "Selected image positions:",
            f"{start_index + 1} to {end_index}"
        )

        # Check that every selected image has an XML file.
        missing_xml = []

        for image_name in self.images:

            xml_name = os.path.splitext(image_name)[0] + ".xml"

            xml_path = os.path.join(
                XML_FOLDER,
                xml_name
            )

            if not os.path.isfile(xml_path):
                missing_xml.append(image_name)

        if missing_xml:
            raise FileNotFoundError(
                f"Found {len(missing_xml)} images without matching XML files.\n"
                f"Examples: {missing_xml[:10]}"
            )

        print("XML annotation check: PASSED")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_name
        )

        # Load image.
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image = transform(image)

        # Find corresponding XML annotation.
        xml_name = os.path.splitext(image_name)[0] + ".xml"

        xml_path = os.path.join(
            XML_FOLDER,
            xml_name
        )

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as error:
            raise ValueError(
                f"Invalid XML annotation: {xml_path}"
            ) from error

        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):

            name_element = obj.find("name")
            box_element = obj.find("bndbox")

            if name_element is None or box_element is None:
                continue

            if name_element.text is None:
                continue

            class_name = name_element.text.strip()

            # Ignore object classes outside the nine defined classes.
            if class_name not in CLASSES:
                continue

            coordinate_names = [
                "xmin",
                "ymin",
                "xmax",
                "ymax"
            ]

            coordinates = []

            for coordinate_name in coordinate_names:

                coordinate_element = box_element.find(
                    coordinate_name
                )

                if (
                    coordinate_element is None
                    or coordinate_element.text is None
                ):
                    raise ValueError(
                        f"Missing {coordinate_name} in {xml_path}"
                    )

                coordinates.append(
                    float(coordinate_element.text)
                )

            xmin, ymin, xmax, ymax = coordinates

            # Reject invalid or zero-area bounding boxes.
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([
                xmin,
                ymin,
                xmax,
                ymax
            ])

            labels.append(CLASSES[class_name])

        # Convert annotations to tensors.
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

        # Image IDs use the image's position in the full sorted list.
        image_id = torch.tensor(
            [self.start_index + index],
            dtype=torch.int64
        )

        area = (
            (boxes[:, 3] - boxes[:, 1])
            * (boxes[:, 2] - boxes[:, 0])
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
# DATASET TEST
# ==========================================

if __name__ == "__main__":

    # Test the first 2,000 images.
    dataset = VOCDataset(
        start_index=0,
        end_index=2000
    )

    print("\n" + "=" * 50)
    print("DATASET TEST RESULTS")
    print("=" * 50)

    print("Selected images:", len(dataset))

    image, target = dataset[0]

    print("First image shape:", image.shape)
    print("First image ID:", target["image_id"].item())
    print("Object labels:", target["labels"])
    print("Bounding boxes:", target["boxes"])

    print("\nDataset test completed successfully.")