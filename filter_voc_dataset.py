import os
import shutil
import xml.etree.ElementTree as ET

TARGET_CLASSES = [
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

IMAGE_DIR = "dataset/raw/JPEGImages"
ANNOTATION_DIR = "dataset/raw/Annotations"

OUTPUT_IMAGE_DIR = "dataset/train/images"
OUTPUT_LABEL_DIR = "dataset/train/annotations"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

count = 0

for xml_file in os.listdir(ANNOTATION_DIR):

    xml_path = os.path.join(ANNOTATION_DIR, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    keep = False

    for obj in root.findall("object"):

        cls = obj.find("name").text

        if cls in TARGET_CLASSES:
            keep = True
            break

    if keep:

        filename = root.find("filename").text

        img_src = os.path.join(IMAGE_DIR, filename)
        xml_src = xml_path

        img_dst = os.path.join(OUTPUT_IMAGE_DIR, filename)
        xml_dst = os.path.join(OUTPUT_LABEL_DIR, xml_file)

        shutil.copy(img_src, img_dst)
        shutil.copy(xml_src, xml_dst)

        count += 1

print("Filtered Images:", count)