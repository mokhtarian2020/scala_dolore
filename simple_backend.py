from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from PIL import Image
import sys
sys.path.append('.')
from pain_detector import PainDetector
import os

app = FastAPI(title="Pain Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pain detector instance
pain_detector = None

def initialize_pain_detector():
    global pain_detector
    if pain_detector is None:
        pain_detector = PainDetector(
            image_size=160,
            checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt',
            num_outputs=40
        )
    return pain_detector

def process_uploaded_image(file_content: bytes) -> np.ndarray:
    """Convert uploaded file to OpenCV image format"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_content))
        
        # Convert PIL to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def validate_image_for_face_detection(image: np.ndarray) -> bool:
    """Check if image contains a detectable face"""
    try:
        detector = initialize_pain_detector()
        landmarks = detector.face_detector(image)
        return landmarks is not None and len(landmarks) > 0
    except:
        return False

@app.get("/")
async def root():
    return {"message": "Pain Detection API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    try:
        detector = initialize_pain_detector()
        return {
            "status": "healthy",
            "model_loaded": detector is not None,
            "device": detector.device if detector else "unknown"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze-example-images")
async def analyze_example_images():
    try:
        detector = initialize_pain_detector()
        
        ref_path = "example_frames/example-reference-frame.png"
        target_path = "example_frames/example-target-frame.png"
        
        if not os.path.exists(ref_path) or not os.path.exists(target_path):
            raise HTTPException(status_code=404, detail="Example images not found")
        
        ref_image = cv2.imread(ref_path)
        target_image = cv2.imread(target_path)
        
        if ref_image is None or target_image is None:
            raise HTTPException(status_code=400, detail="Failed to load example images")
        
        detector.add_references([ref_image, ref_image, ref_image])
        pain_score = detector.predict_pain(target_image)
        
        if pain_score < 1:
            pain_level = "No Pain"
        elif pain_score < 2:
            pain_level = "Minimal Pain"
        elif pain_score < 3:
            pain_level = "Mild Pain"
        elif pain_score < 4:
            pain_level = "Moderate Pain"
        elif pain_score < 6:
            pain_level = "Moderate-Severe Pain"
        elif pain_score < 8:
            pain_level = "Severe Pain"
        else:
            pain_level = "Very Severe Pain"
        
        return {
            "success": True,
            "pain_score": float(pain_score),
            "pain_level": pain_level,
            "reference_image": "example-reference-frame.png",
            "target_image": "example-target-frame.png"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing example images: {str(e)}")

@app.post("/upload-reference")
async def upload_reference_images(
    ref1: UploadFile = File(...),
    ref2: UploadFile = File(None),
    ref3: UploadFile = File(None)
):
    """Upload reference images (neutral expressions)"""
    try:
        detector = initialize_pain_detector()
        
        reference_images = []
        
        # Process first reference image (required)
        ref1_content = await ref1.read()
        ref1_image = process_uploaded_image(ref1_content)
        
        if not validate_image_for_face_detection(ref1_image):
            raise HTTPException(status_code=400, detail="No face detected in reference image 1")
        
        reference_images.append(ref1_image)
        
        # Process additional reference images if provided
        if ref2:
            ref2_content = await ref2.read()
            ref2_image = process_uploaded_image(ref2_content)
            if validate_image_for_face_detection(ref2_image):
                reference_images.append(ref2_image)
        
        if ref3:
            ref3_content = await ref3.read()
            ref3_image = process_uploaded_image(ref3_content)
            if validate_image_for_face_detection(ref3_image):
                reference_images.append(ref3_image)
        
        # If only one reference provided, duplicate it
        while len(reference_images) < 3:
            reference_images.append(reference_images[0])
        
        # Add references to detector
        detector.add_references(reference_images)
        
        return {
            "success": True,
            "message": f"Successfully uploaded {len(reference_images)} reference image(s)",
            "references_count": len(reference_images)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing reference images: {str(e)}")

@app.post("/predict-pain")
async def predict_pain(target: UploadFile = File(...)):
    """Predict pain level in target image"""
    try:
        detector = initialize_pain_detector()
        
        if not hasattr(detector, 'ref_frames') or len(detector.ref_frames) == 0:
            raise HTTPException(status_code=400, detail="No reference images uploaded. Please upload reference images first.")
        
        # Process target image
        target_content = await target.read()
        target_image = process_uploaded_image(target_content)
        
        if not validate_image_for_face_detection(target_image):
            raise HTTPException(status_code=400, detail="No face detected in target image")
        
        # Predict pain
        pain_score = detector.predict_pain(target_image)
        
        # Interpret pain level
        if pain_score < 1:
            pain_level = "No Pain"
        elif pain_score < 2:
            pain_level = "Minimal Pain"
        elif pain_score < 3:
            pain_level = "Mild Pain"
        elif pain_score < 4:
            pain_level = "Moderate Pain"
        elif pain_score < 6:
            pain_level = "Moderate-Severe Pain"
        elif pain_score < 8:
            pain_level = "Severe Pain"
        else:
            pain_level = "Very Severe Pain"
        
        return {
            "success": True,
            "pain_score": float(pain_score),
            "pain_level": pain_level,
            "pspi_scale": "0-1: No Pain, 1-2: Minimal, 2-3: Mild, 3-4: Moderate, 4-6: Moderate-Severe, 6-8: Severe, 8+: Very Severe"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting pain: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
