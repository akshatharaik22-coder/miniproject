import cv2
import os

# Input folder (Low-light images)
input_folder = "dataset/train/lowlight"

# Output folder (Enhanced images)
output_folder = "dataset/train/enhanced"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if filename.endswith(".jpg"):

        img_path = os.path.join(input_folder, filename)

        image = cv2.imread(img_path)

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        # Apply CLAHE on L channel
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))

        enhanced = cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2BGR
        )

        cv2.imwrite(
            os.path.join(output_folder, filename),
            enhanced
        )

print("Enhanced images created successfully!")