"""
Enhanced Clinical Backend API for SCALA DOLORE Pain Assessment
Integrates with Welodge database and generates comprehensive PDF reports
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np
import json
import io
from typing import Optional
import logging

# Import your existing modules
from pain_detector import PainDetector
from database.welodge_connector import WelodgeConnector
from reports.pdf_generator import PainAssessmentReport

app = FastAPI(title="SCALA DOLORE Clinical API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pain_detector = PainDetector()
db_connector = WelodgeConnector()
report_generator = PainAssessmentReport()

@app.post("/api/check_patient_reference")
async def check_patient_reference(patient_id: str = Form(...)):
    """
    Check if patient has a reference image in Welodge database
    Returns status and instructions for next steps
    """
    try:
        has_reference = db_connector.has_reference_image(patient_id)
        
        return {
            "patient_id": patient_id,
            "has_reference": has_reference,
            "status": "ready" if has_reference else "needs_reference",
            "message": "Patient ready for assessment" if has_reference 
                      else "Please capture or upload reference image first",
            "next_action": "capture_target" if has_reference else "capture_reference"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/save_reference_image")
async def save_reference_image(
    patient_id: str = Form(...),
    image: UploadFile = File(...),
    patient_data: str = Form(default="{}")
):
    """
    Save or update reference image for patient
    This endpoint is called when taking/uploading reference image
    """
    try:
        # Read and validate image
        image_bytes = await image.read()
        
        # Validate image using your pain detector
        try:
            # Convert to numpy array for validation
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image format")
            
            # Validate image quality (ensure face is detected)
            result = pain_detector.verify_reference_image(img)
            if not result['valid']:
                raise HTTPException(status_code=400, 
                                  detail=f"Image quality issue: {result['message']}")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image validation failed: {str(e)}")
        
        # Parse patient data
        try:
            patient_info = json.loads(patient_data)
        except json.JSONDecodeError:
            patient_info = {}
        
        # Save to database
        db_connector.save_reference_image(
            patient_id=patient_id,
            image_data=image_bytes,
            metadata={
                "image_name": image.filename,
                "image_size": len(image_bytes),
                "patient_info": patient_info,
                "validation_result": result
            }
        )
        
        return {
            "status": "success",
            "message": "Reference image saved successfully",
            "patient_id": patient_id,
            "image_quality": result,
            "next_action": "ready_for_assessment"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save reference image: {str(e)}")

@app.post("/api/assess_pain_with_report")
async def assess_pain_with_report(
    patient_id: str = Form(...),
    target_image: UploadFile = File(...),
    patient_data: str = Form(default="{}")
):
    """
    Complete pain assessment workflow:
    1. Check for reference image
    2. Perform pain assessment
    3. Generate comprehensive PDF report
    4. Save assessment to database
    """
    try:
        # Check if reference exists
        if not db_connector.has_reference_image(patient_id):
            raise HTTPException(
                status_code=400, 
                detail="No reference image found. Please capture reference image first."
            )
        
        # Get reference image from database
        reference_bytes = db_connector.get_reference_image(patient_id)
        if not reference_bytes:
            raise HTTPException(status_code=500, detail="Failed to retrieve reference image")
        
        # Read target image
        target_bytes = await target_image.read()
        
        # Convert images for pain detection
        ref_nparr = np.frombuffer(reference_bytes, np.uint8)
        ref_img = cv2.imdecode(ref_nparr, cv2.IMREAD_COLOR)
        
        target_nparr = np.frombuffer(target_bytes, np.uint8)
        target_img = cv2.imdecode(target_nparr, cv2.IMREAD_COLOR)
        
        if ref_img is None or target_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Perform pain assessment
        pain_detector.add_references([ref_img])
        pain_score = pain_detector.predict_pain(target_img)
        
        # Parse patient data
        try:
            patient_info = json.loads(patient_data)
        except json.JSONDecodeError:
            patient_info = {"patient_id": patient_id}
        
        # Generate comprehensive PDF report
        pdf_bytes = report_generator.generate_report(
            patient_data=patient_info,
            reference_image=reference_bytes,
            target_image=target_bytes,
            pain_score=pain_score
        )
        
        # Save assessment to database
        db_connector.save_assessment(
            patient_id=patient_id,
            target_image=target_bytes,
            reference_image=reference_bytes,
            pain_score=pain_score,
            metadata={
                "target_image_name": target_image.filename,
                "assessment_type": "facial_expression_analysis",
                "model_version": "ConvNetOrdinalLateFusion",
                "patient_info": patient_info
            }
        )
        
        # Determine pain level
        def get_pain_level(score):
            if score < 1: return "No Pain"
            elif score < 3: return "Minimal Pain"  
            elif score < 5: return "Mild Pain"
            elif score < 7: return "Moderate Pain"
            elif score < 10: return "Severe Pain"
            else: return "Very Severe Pain"
        
        pain_level = get_pain_level(pain_score)
        
        return {
            "status": "success",
            "patient_id": patient_id,
            "pain_score": round(pain_score, 2),
            "pain_level": pain_level,
            "assessment_date": datetime.now().isoformat(),
            "pdf_report": base64.b64encode(pdf_bytes).decode('utf-8'),
            "recommendation": get_clinical_recommendation(pain_score),
            "scale_reference": {
                "0-1": "No Pain",
                "1-3": "Minimal Pain",
                "3-5": "Mild Pain", 
                "5-7": "Moderate Pain",
                "7-10": "Severe Pain",
                "10+": "Very Severe Pain"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@app.get("/api/download_report/{patient_id}")
async def download_report(patient_id: str, assessment_id: Optional[int] = None):
    """
    Download PDF report for specific assessment
    """
    try:
        # Here you would implement logic to retrieve stored assessment
        # and regenerate or serve the PDF report
        
        # For now, return info about the endpoint
        return {
            "message": "Report download endpoint",
            "patient_id": patient_id,
            "note": "Implementation depends on your specific storage requirements"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

def get_clinical_recommendation(score: float) -> str:
    """Get clinical recommendation based on pain score"""
    if score < 3:
        return "Continue monitoring. No immediate intervention required."
    elif score < 5:
        return "Consider non-pharmacological interventions."
    elif score < 7:
        return "Evaluate for pain management interventions."
    else:
        return "Immediate pain management assessment recommended."

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SCALA DOLORE Clinical API",
        "version": "1.0.0",
        "pain_detector": "loaded",
        "database": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)