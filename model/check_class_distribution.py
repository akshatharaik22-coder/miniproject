import os
import xml.etree.ElementTree as ET
from collections import Counter

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

XML_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train",
    "annotations"
)

CLASSES = [
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

counts = Counter()

xml_files = [
    f for f in os.listdir(XML_FOLDER)
    if f.lower().endswith(".xml")
]

print("Total annotation files:", len(xml_files))

for xml_file in xml_files:

    xml_path = os.path.join(
        XML_FOLDER,
        xml_file
    )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):

        class_name = obj.find("name").text.strip()

        if class_name in CLASSES:
            counts[class_name] += 1

print("\n===================================")
print("TRAINING DATA CLASS DISTRIBUTION")
print("===================================\n")

total = sum(counts.values())

for class_name in CLASSES:

    count = counts[class_name]

    percentage = (
        count / total * 100
        if total > 0
        else 0
    )

    print(
        f"{class_name:15s} : "
        f"{count:5d} "
        f"({percentage:.2f}%)"
    )

print("-----------------------------------")
print(f"Total objects: {total}")