
import os
import time

import cv2
import numpy as np
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_epoch_3.pth"
)

NUM_CLASSES = 10

CONFIDENCE_THRESHOLD = 0.50

# Images below this average grayscale brightness
# are treated as low-light images.
LOW_LIGHT_THRESHOLD = 100.0

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

DEVICE = torch.device("cpu")


# =========================================================
# MODEL INITIALIZATION
# =========================================================

def create_model():

    print("Creating Faster R-CNN architecture...")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    print("Loading trained Epoch 3 checkpoint...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    print("Epoch 3 model loaded successfully.")

    return model


# Load the model once when the API starts.
model = create_model()


# =========================================================
# IMAGE BRIGHTNESS DETECTION
# =========================================================

def calculate_brightness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mean_brightness = float(np.mean(gray))

    return mean_brightness


# =========================================================
# ORIGINAL GAMMA CORRECTION
# =========================================================

def original_gamma_correction(image, gamma=3.0):

    """
    Preserves the original project preprocessing.

    Gamma > 1 darkens the image.
    This is retained for normal-light images to preserve
    the previous inference pipeline.
    """

    table = np.array(
        [
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ],
        dtype=np.uint8
    )

    corrected = cv2.LUT(
        image,
        table
    )

    return corrected


# =========================================================
# ADAPTIVE GAMMA CORRECTION
# =========================================================

def adaptive_gamma_correction(image):

    """
    Brightens low-light images automatically.

    The gamma value is selected using image brightness.

    Gamma < 1 brightens dark images.
    """

    brightness = calculate_brightness(image)

    if brightness < 30:

        gamma = 3.0

    elif brightness < 50:

        gamma = 2.5

    elif brightness < 75:

        gamma = 2.0

    elif brightness < 100:

        gamma = 1.5

    else:

        gamma = 1.0

    inverse_gamma = 1.0 / gamma

    table = np.array(
        [
            ((i / 255.0) ** inverse_gamma) * 255
            for i in range(256)
        ],
        dtype=np.uint8
    )

    corrected = cv2.LUT(
        image,
        table
    )

    return corrected, gamma


# =========================================================
# CLAHE ENHANCEMENT
# =========================================================

def apply_clahe(image):

    """
    Contrast Limited Adaptive Histogram Equalization.

    Enhances local contrast while limiting excessive
    amplification of image noise.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    merged = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel
        )
    )

    enhanced = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# =========================================================
# AUTOMATIC IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image, return_info=False):

    """
    Automatically selects preprocessing based on brightness.

    LOW LIGHT:
        Adaptive gamma correction -> CLAHE

    NORMAL LIGHT:
        Original gamma 3.0 correction -> CLAHE

    The normal-light processing preserves the previous
    inference pipeline.
    """

    brightness = calculate_brightness(image)

    if brightness < LOW_LIGHT_THRESHOLD:

        # Low-light image enhancement.
        gamma_corrected, gamma_value = (
            adaptive_gamma_correction(image)
        )

        enhanced = apply_clahe(
            gamma_corrected
        )

        preprocessing_mode = "adaptive_low_light"

    else:

        # Preserve original normal-light pipeline.
        gamma_corrected = original_gamma_correction(
            image,
            gamma=3.0
        )

        enhanced = apply_clahe(
            gamma_corrected
        )

        gamma_value = 3.0

        preprocessing_mode = "original_normal_light"

    if return_info:

        return enhanced, {
            "mode": preprocessing_mode,
            "original_brightness": round(brightness, 2),
            "gamma": gamma_value
        }

    return enhanced


# =========================================================
# DISTANCE ESTIMATION
# =========================================================

def estimate_distance(box_height):

    if box_height <= 0:

        return 0.0

    focal_length = 700

    real_height = 1.7

    distance = (
        real_height * focal_length
    ) / box_height

    return round(distance, 2)


# =========================================================
# DIRECTION ESTIMATION
# =========================================================

def get_direction(center_x, image_width):

    if center_x < image_width / 3:

        return "Left"

    elif center_x > (2 * image_width) / 3:

        return "Right"

    return "Center"


# =========================================================
# OBJECT DETECTION
# =========================================================

def detect_objects(image):

    start_time = time.time()

    if image is None or not isinstance(image, np.ndarray):

        raise ValueError("Invalid input image.")

    if image.size == 0:

        raise ValueError("Input image is empty.")

    original_height, original_width = image.shape[:2]

    if original_width <= 0 or original_height <= 0:

        raise ValueError("Invalid image dimensions.")

    # -----------------------------------------------------
    # STEP 1: AUTOMATIC LOW-LIGHT / NORMAL-LIGHT PROCESSING
    # -----------------------------------------------------

    enhanced, preprocessing_info = preprocess_image(
        image,
        return_info=True
    )

    # -----------------------------------------------------
    # STEP 2: CONVERT IMAGE TO RGB
    # -----------------------------------------------------

    rgb_image = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2RGB
    )

    tensor_image = F.to_tensor(
        rgb_image
    ).to(DEVICE)

    # -----------------------------------------------------
    # STEP 3: FASTER R-CNN INFERENCE
    # -----------------------------------------------------

    with torch.inference_mode():

        prediction = model([
            tensor_image
        ])[0]

    boxes = prediction["boxes"].cpu().numpy()

    labels = prediction["labels"].cpu().numpy()

    scores = prediction["scores"].cpu().numpy()

    # -----------------------------------------------------
    # STEP 4: PROCESS DETECTIONS
    # -----------------------------------------------------

    detections = []

    object_counts = {}

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        if score < CONFIDENCE_THRESHOLD:

            continue

        class_id = int(label)

        if class_id not in CLASSES:

            continue

        x1, y1, x2, y2 = box.tolist()

        # Keep bounding boxes inside the image.
        x1 = max(
            0,
            min(original_width, x1)
        )

        x2 = max(
            0,
            min(original_width, x2)
        )

        y1 = max(
            0,
            min(original_height, y1)
        )

        y2 = max(
            0,
            min(original_height, y2)
        )

        box_height = max(
            0,
            y2 - y1
        )

        center_x = (
            x1 + x2
        ) / 2

        object_name = CLASSES[class_id]

        # Direction.
        direction = get_direction(
            center_x,
            original_width
        )

        # Approximate distance.
        distance = estimate_distance(
            box_height
        )

        detections.append({

            "class_id": class_id,

            "label": object_name,

            "confidence": round(
                float(score),
                4
            ),

            "box": {

                "x1": round(float(x1), 2),

                "y1": round(float(y1), 2),

                "x2": round(float(x2), 2),

                "y2": round(float(y2), 2)

            },

            "direction": direction,

            "distance": distance

        })

        object_counts[object_name] = (
            object_counts.get(object_name, 0) + 1
        )

    # -----------------------------------------------------
    # STEP 5: INFERENCE TIME
    # -----------------------------------------------------

    inference_time = round(
        time.time() - start_time,
        3
    )

    # -----------------------------------------------------
    # STEP 6: RETURN RESULTS
    # -----------------------------------------------------

    return {

        "success": True,

        "image_width": original_width,

        "image_height": original_height,

        "brightness": preprocessing_info[
            "original_brightness"
        ],

        "preprocessing": preprocessing_info[
            "mode"
        ],

        "gamma_used": preprocessing_info[
            "gamma"
        ],

        "detections": detections,

        "object_counts": object_counts,

        "total_objects": len(detections),

        "inference_time": inference_time

    }
