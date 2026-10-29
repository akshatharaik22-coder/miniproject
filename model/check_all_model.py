import os

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train",
    "enhanced"
)

# ============================================================
# EXPECTED MODEL RANGES
# ============================================================

MODEL_RANGES = {
    "Model 1": (1, 2000),
    "Model 2": (2001, 4000),
    "Model 3": (4001, 6000),
    "Model 4": (6001, 8000),
    "Model 5": (8001, 10000),
}

# ============================================================
# GET ALL IMAGES
# ============================================================

all_images = sorted(
    [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

print("=" * 60)
print("MODEL DATASET RANGE CHECK")
print("=" * 60)

print(f"Total training images available: {len(all_images)}")
print()

# ============================================================
# CHECK EACH MODEL RANGE
# ============================================================

for model_name, (start, end) in MODEL_RANGES.items():

    selected_images = all_images[start - 1:end]

    print("-" * 60)
    print(model_name)

    print(f"Expected range : {start} - {end}")
    print(f"Images selected : {len(selected_images)}")

    if selected_images:
        print(f"First image    : {selected_images[0]}")
        print(f"Last image     : {selected_images[-1]}")

    if len(selected_images) == (end - start + 1):
        print("STATUS         : OK")
    else:
        print("STATUS         : ERROR")

print()
print("=" * 60)
print("CHECK COMPLETED")
print("=" * 60)