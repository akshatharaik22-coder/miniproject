
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from api.detection_service import detect_objects

app = FastAPI(
    title="Low-Light Obstacle Detection API",
    description="Faster R-CNN obstacle detection with gamma correction and CLAHE enhancement",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Low-Light Obstacle Detection API is running",
        "status": "success"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Detection API is ready"
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # Validate the actual image bytes instead of relying on MIME type.
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image. Could not decode the uploaded file."
            )

        result = detect_objects(image)

        return {
            "status": "success",
            "result": result
        }

    except HTTPException:
        raise
    except Exception as error:
        print("Detection error:", str(error))
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(error)}"
        )