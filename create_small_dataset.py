import os
import random
import shutil
import xml.etree.ElementTree as ET

# ===========================
# Original Dataset Paths
# ===========================

IMAGE_FOLDER = "dataset/train/images"
XML_FOLDER = "dataset/train/annotations"
LOWLIGHT_FOLDER = "dataset/train/lowlight"
ENHANCED_FOLDER = "dataset/train/enhanced"

# ===========================
# New Small Dataset Paths
# ===========================

NEW_IMAGE_FOLDER = "dataset/small_train/images"
NEW_XML_FOLDER = "dataset/small_train/annotations"
NEW_LOWLIGHT_FOLDER = "dataset/small_train/lowlight"
NEW_ENHANCED_FOLDER = "dataset/small_train/enhanced"

# ===========================
# Classes Needed
# ===========================

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

# ===========================
# Number of Images
# ===========================

TOTAL_IMAGES = 2000

# ===========================
# Create Folders
# ===========================

os.makedirs(NEW_IMAGE_FOLDER, exist_ok=True)
os.makedirs(NEW_XML_FOLDER, exist_ok=True)
os.makedirs(NEW_LOWLIGHT_FOLDER, exist_ok=True)
os.makedirs(NEW_ENHANCED_FOLDER, exist_ok=True)

# ===========================
# Find Valid Images
# ===========================

valid_images = []

xml_files = os.listdir(XML_FOLDER)

print("Checking XML files...")

for xml_file in xml_files:

    xml_path = os.path.join(XML_FOLDER, xml_file)

    tree = ET.parse(xml_path)

    root = tree.getroot()

    keep = False

    for obj in root.findall("object"):

        cls = obj.find("name").text

        if cls in CLASSES:
            keep = True
            break

    if keep:
        valid_images.append(xml_file.replace(".xml", ".jpg"))

print("Valid Images Found :", len(valid_images))

# ===========================
# Random Selection
# ===========================

random.shuffle(valid_images)

selected = valid_images[:TOTAL_IMAGES]

print("Selected Images :", len(selected))

# ===========================
# Copy Files
# ===========================

print("Copying Files...")

for image_name in selected:

    xml_name = image_name.replace(".jpg", ".xml")

    shutil.copy(
        os.path.join(IMAGE_FOLDER, image_name),
        os.path.join(NEW_IMAGE_FOLDER, image_name)
    )

    shutil.copy(
        os.path.join(XML_FOLDER, xml_name),
        os.path.join(NEW_XML_FOLDER, xml_name)
    )

    lowlight_path = os.path.join(LOWLIGHT_FOLDER, image_name)

    if os.path.exists(lowlight_path):
        shutil.copy(
            lowlight_path,
            os.path.join(NEW_LOWLIGHT_FOLDER, image_name)
        )

    enhanced_path = os.path.join(ENHANCED_FOLDER, image_name)

    if os.path.exists(enhanced_path):
        shutil.copy(
            enhanced_path,
            os.path.join(NEW_ENHANCED_FOLDER, image_name)
        )

print()

print("====================================")
print("Small Dataset Created Successfully")
print("Images :", len(selected))
print("====================================")