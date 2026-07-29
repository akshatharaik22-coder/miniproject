import os
import xml.etree.ElementTree as ET

# Path to XML files
ANNOTATION_PATH = "dataset/train/annotations"

# Get all XML files
xml_files = os.listdir(ANNOTATION_PATH)

print("Total XML Files:", len(xml_files))

# Read first XML file
xml_file = xml_files[0]

xml_path = os.path.join(ANNOTATION_PATH, xml_file)

tree = ET.parse(xml_path)

root = tree.getroot()

print("\nImage Name:")
print(root.find("filename").text)

print("\nImage Size")

width = root.find("size/width").text
height = root.find("size/height").text

print("Width :", width)
print("Height:", height)

print("\nObjects Found")

for obj in root.findall("object"):

    name = obj.find("name").text

    xmin = obj.find("bndbox/xmin").text
    ymin = obj.find("bndbox/ymin").text
    xmax = obj.find("bndbox/xmax").text
    ymax = obj.find("bndbox/ymax").text

    print("---------------------")
    print("Class :", name)
    print("Bounding Box")
    print("xmin:", xmin)
    print("ymin:", ymin)
    print("xmax:", xmax)
    print("ymax:", ymax)