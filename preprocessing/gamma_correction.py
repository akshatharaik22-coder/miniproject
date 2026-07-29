import cv2
import numpy as np
import os

input_folder = "dataset/train/images"
output_folder = "dataset/train/lowlight"

os.makedirs(output_folder, exist_ok=True)

gamma = 3.0

for filename in os.listdir(input_folder):

    if filename.endswith(".jpg"):

        img_path = os.path.join(input_folder, filename)

        image = cv2.imread(img_path)

        invGamma = 1.0 / gamma

        table = np.array(
            [((i / 255.0) ** gamma) * 255
             for i in np.arange(256)]
        ).astype("uint8")

        lowlight = cv2.LUT(image, table)

        cv2.imwrite(
            os.path.join(output_folder, filename),
            lowlight
        )

print("Low-light images created.")