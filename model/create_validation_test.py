import os
import shutil

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Source folders
TRAIN_ENHANCED = os.path.join(PROJECT_ROOT, "dataset", "train", "enhanced")
TRAIN_LOWLIGHT = os.path.join(PROJECT_ROOT, "dataset", "train", "lowlight")
TRAIN_XML = os.path.join(PROJECT_ROOT, "dataset", "train", "annotations")

# Destination folders
VAL_ENHANCED = os.path.join(PROJECT_ROOT, "dataset", "validation", "enhanced")
VAL_LOWLIGHT = os.path.join(PROJECT_ROOT, "dataset", "validation", "lowlight")
VAL_XML = os.path.join(PROJECT_ROOT, "dataset", "validation", "annotations")

TEST_ENHANCED = os.path.join(PROJECT_ROOT, "dataset", "test", "enhanced")
TEST_LOWLIGHT = os.path.join(PROJECT_ROOT, "dataset", "test", "lowlight")
TEST_XML = os.path.join(PROJECT_ROOT, "dataset", "test", "annotations")

# Create folders
for folder in [
    VAL_ENHANCED, VAL_LOWLIGHT, VAL_XML,
    TEST_ENHANCED, TEST_LOWLIGHT, TEST_XML
]:
    os.makedirs(folder, exist_ok=True)


# Get sorted images
images = sorted([
    f for f in os.listdir(TRAIN_ENHANCED)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])


# Already trained: 1-4000
# Validation: 4001-5283
# Test: 5284-6566

validation_images = images[4000:5283]
test_images = images[5283:6566]


def copy_dataset(image_list, enhanced_dest, lowlight_dest, xml_dest):

    copied = 0
    skipped = 0

    for image_name in image_list:

        base_name = os.path.splitext(image_name)[0]
        xml_name = base_name + ".xml"

        enhanced_src = os.path.join(TRAIN_ENHANCED, image_name)
        lowlight_src = os.path.join(TRAIN_LOWLIGHT, image_name)
        xml_src = os.path.join(TRAIN_XML, xml_name)

        # Check all three files exist
        if not (
            os.path.exists(enhanced_src)
            and os.path.exists(lowlight_src)
            and os.path.exists(xml_src)
        ):
            print("SKIPPED:", image_name)
            skipped += 1
            continue

        shutil.copy2(
            enhanced_src,
            os.path.join(enhanced_dest, image_name)
        )

        shutil.copy2(
            lowlight_src,
            os.path.join(lowlight_dest, image_name)
        )

        shutil.copy2(
            xml_src,
            os.path.join(xml_dest, xml_name)
        )

        copied += 1

    return copied, skipped


print("\nCreating VALIDATION dataset...")

val_copied, val_skipped = copy_dataset(
    validation_images,
    VAL_ENHANCED,
    VAL_LOWLIGHT,
    VAL_XML
)

print("Validation images copied:", val_copied)
print("Validation skipped:", val_skipped)


print("\nCreating TEST dataset...")

test_copied, test_skipped = copy_dataset(
    test_images,
    TEST_ENHANCED,
    TEST_LOWLIGHT,
    TEST_XML
)

print("Test images copied:", test_copied)
print("Test skipped:", test_skipped)


print("\n====================================")
print("DATASET SPLIT COMPLETED")
print("====================================")
print("Training already used: 1 - 4000")
print("Validation:", val_copied)
print("Test:", test_copied)