import os
import sys
import time
import threading
import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F

# -----------------------------
# Project Imports
# -----------------------------

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(project_root)

from faster_rcnn_model import get_model
from model.distance_estimation import estimate_distance
from model.direction import get_direction
from model.object_counter import count_objects
from audio.smart_voice import speak_message


# -----------------------------
# Classes
# -----------------------------

CLASSES = {
    1: "Person",
    2: "Chair",
    3: "Bicycle",
    4: "Car",
    5: "Bus",
    6: "Motorbike",
    7: "Dog",
    8: "Dining Table",
    9: "Sofa"
}


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# -----------------------------
# Load Model
# -----------------------------

model = get_model()

model_path = os.path.join(
    project_root,
    "results",
    "model_epoch_5.pth"
)

print("Loading model from:", model_path)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.to(device)
model.eval()

print("Model Loaded Successfully!")


# -----------------------------
# Open Camera
# -----------------------------

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Unable to open camera.")
    exit()

print("Camera Started...")


# -----------------------------
# Variables
# -----------------------------

last_voice_time = 0
voice_speaking = False

prev_time = time.time()


# -----------------------------
# Real-Time Detection
# -----------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("Unable to read camera frame.")
        break


    # -----------------------------
    # Gamma Correction
    # -----------------------------

    gamma = 3.0

    table = np.array(
        [((i / 255.0) ** gamma) * 255
         for i in np.arange(256)]
    ).astype("uint8")

    lowlight = cv2.LUT(frame, table)


    # -----------------------------
    # CLAHE Enhancement
    # -----------------------------

    lab = cv2.cvtColor(
        lowlight,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge(
        (cl, a, b)
    )

    enhanced = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )


    # -----------------------------
    # Prediction
    # -----------------------------

    image_rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2RGB
    )

    image_tensor = F.to_tensor(
        image_rgb
    ).to(device)

    with torch.no_grad():

        prediction = model(
            [image_tensor]
        )


    boxes = prediction[0]["boxes"].cpu().numpy()
    labels = prediction[0]["labels"].cpu().numpy()
    scores = prediction[0]["scores"].cpu().numpy()


    # -----------------------------
    # Detection Variables
    # -----------------------------

    detected_objects = []
    object_counts = {}
    voice_messages = []


    # -----------------------------
    # Draw Bounding Boxes
    # -----------------------------

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        if score < 0.50:
            continue


        x1, y1, x2, y2 = map(
            int,
            box
        )


        object_name = CLASSES.get(
            label,
            "Unknown"
        )


        detected_objects.append(
            object_name
        )


        # -----------------------------
        # Distance
        # -----------------------------

        box_height = y2 - y1

        distance = estimate_distance(
            box_height
        )


        # -----------------------------
        # Direction
        # -----------------------------

        center_x = (x1 + x2) // 2

        direction = get_direction(
            center_x,
            enhanced.shape[1]
        )


        # -----------------------------
        # Voice Message
        # -----------------------------

        voice_messages.append(
            f"{object_name} on your "
            f"{direction.lower()}, approximately "
            f"{distance:.1f} meters away"
        )


        # -----------------------------
        # Draw Bounding Box
        # -----------------------------

        cv2.rectangle(
            enhanced,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # -----------------------------
        # Display Object Information
        # -----------------------------

        cv2.putText(
            enhanced,
            f"{object_name} | "
            f"{distance:.1f}m | "
            f"{direction}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # -----------------------------
    # Object Counting
    # -----------------------------

    object_counts = count_objects(
        detected_objects
    )


# -----------------------------
# Continuous Voice Alert
# -----------------------------

    current_time = time.time()

    if len(detected_objects) > 0:

        if current_time - last_voice_time >= 3:

            message = ". ".join(voice_messages)

            if not voice_speaking:

                voice_speaking = True

                def voice_alert(text):
                    global voice_speaking

                    try:
                        speak_message(text)

                    finally:
                        voice_speaking = False

                threading.Thread(
                    target=voice_alert,
                    args=(message,),
                    daemon=True
                ).start()

                print("VOICE:", message)

                last_voice_time = current_time

    else:

        # No object detected
        last_voice_time = current_time

    # -----------------------------
    # FPS
    # -----------------------------

    fps = 1 / (
        current_time - prev_time
    )

    prev_time = current_time


    cv2.putText(
        enhanced,
        f"FPS : {fps:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    # -----------------------------
    # Total Object Count
    # -----------------------------

    cv2.putText(
        enhanced,
        f"Objects : {len(detected_objects)}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    # -----------------------------
    # Individual Object Counts
    # -----------------------------

    y = 100

    for obj, count in object_counts.items():

        cv2.putText(
            enhanced,
            f"{obj}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        y += 30


    # -----------------------------
    # Show Result
    # -----------------------------

    cv2.imshow(
        "Real-Time Obstacle Detection",
        enhanced
    )


    # -----------------------------
    # Keyboard Input
    # -----------------------------

    key = cv2.waitKey(1) & 0xFF


    # -----------------------------
    # Save Screenshot
    # -----------------------------

    if key == ord("s"):

        os.makedirs(
            os.path.join(
                project_root,
                "results"
            ),
            exist_ok=True
        )

        filename = os.path.join(
            project_root,
            "results",
            f"result_{int(time.time())}.jpg"
        )

        cv2.imwrite(
            filename,
            enhanced
        )

        print(
            "Image Saved:",
            filename
        )


    # -----------------------------
    # Quit
    # -----------------------------

    if key == ord("q"):
        break


# -----------------------------
# Release Resources
# -----------------------------

camera.release()

cv2.destroyAllWindows()

print("Application Closed.")