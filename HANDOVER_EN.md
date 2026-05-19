# SCALA DOLORE — Project Handover Document (English)

**Project:** SCALA DOLORE — AI-Powered Pain Assessment System  
**Parent System:** CATO MAIOR  
**Handover Date:** May 2026  
**Written by:** Outgoing Developer  

---

## 1. Project Overview

SCALA DOLORE is a clinical AI system for **non-verbal pain assessment**, primarily targeting elderly patients with dementia who cannot self-report their pain. The system analyses facial expressions using a deep learning model to output a PSPI (Pictorial Scale of Pain Intensity) score, allowing clinicians to objectively measure and document a patient's pain level.

The project is part of the larger **CATO MAIOR** clinical platform, which is described in detail in `documents/Specifiche App Terapia Dolore.txt`.

---

## 2. Research Foundation

The AI model is based on the published academic paper:

> **"Unobtrusive Pain Monitoring in Older Adults with Dementia using Pairwise and Contrastive Training"**  
> IEEE Xplore DOI: [10.1109/...](https://ieeexplore.ieee.org/document/9298886)

The key idea: instead of classifying pain from a single image, the model compares a **target frame** (current expression) against a **reference frame** (the same patient with a neutral expression). This pairwise approach makes it more robust and patient-specific.

---

## 3. Project Structure

```
project/
├── pain_detector.py            # Core AI engine — the main class to understand first
├── clinical_backend.py         # Production FastAPI REST API (clinical use)
├── simple_backend.py           # Simplified FastAPI API (testing/dev)
├── frontend_server.py          # Lightweight HTTP server for the web UI
├── test.py                     # Basic test script using example frames
├── compare_models.py           # Compares the two pretrained checkpoints
├── detailed_analysis.py        # Detailed image/face detection analysis tool
├── test_example_images.py      # Test using example_frames/ images
├── start_backend.sh            # Shell script to launch backend (needs fix — see §8)
├── standard_face_68.npy        # CRITICAL: mean 68 landmark positions for alignment
├── requirements.txt            # Python dependencies
│
├── models/
│   └── comparative_model.py    # ConvNetOrdinalLateFusion neural network definition
│
├── face_alignment/             # Local copy of Face Alignment Network (FAN) library
│   ├── api.py                  # Main FAN API
│   ├── models.py               # FAN model definitions
│   ├── utils.py                # FAN utilities
│   └── detection/              # Face detector (S3FD, dlib, folder-based)
│
├── checkpoints/                # Pretrained model weights
│   ├── 50342566/50343918_3/model_epoch4.pt   # UNBC + UofR (40 outputs) ← PRIMARY
│   └── 59448122/59448122_3/model_epoch13.pt  # UNBC only (7 outputs)
│
├── backend/
│   └── image_processor.py      # Image preprocessing utilities
│
├── database/
│   └── welodge_connector.py    # SQLite database connector
│
├── reports/
│   └── pdf_generator.py        # PDF report generator (ReportLab)
│
├── frontend/
│   ├── clinical_interface.html # Main clinical web UI (multi-step workflow)
│   └── index.html              # Simple demo UI
│
├── example_frames/             # Example reference and target face images for testing
├── pretrained/                 # FAN pretrained weights (currently empty — see §8)
├── documents/                  # Project specifications in Italian
└── docs/images/                # Documentation images
```

---

## 4. How the AI Works — Step by Step

1. **Input**: Two images of the same patient — a *reference* (neutral face) and a *target* (face to assess).
2. **Face Detection**: The `FaceAlignment` (FAN) model with S3FD backend detects faces and extracts 68 facial landmarks.
3. **Alignment**: Each image is aligned using:
   - **Similarity Transform** (rotation/scale) aligning eyes and mouth anchor points.
   - **Piecewise Affine Transform** warping 31 key landmarks to a standard template (`standard_face_68.npy`).
4. **Preprocessing**: Grayscale conversion + CLAHE histogram normalisation → 160×160 pixel patch.
5. **Model Inference** (`ConvNetOrdinalLateFusion`):
   - Target and reference patches are passed through the same CNN backbone.
   - Feature maps are **subtracted** (target − reference), capturing expression difference.
   - Pooling + fully-connected layers → PSPI score output.
6. **Scoring**: If multiple reference frames are provided, the mean of all predictions is returned.

### Pain Scale (PSPI)

| Score | Level          | Clinical Meaning                         |
|-------|----------------|------------------------------------------|
| 0–1   | No Pain        | No visible signs of discomfort           |
| 1–3   | Minimal Pain   | Slight facial tension                    |
| 3–5   | Mild Pain      | Noticeable facial expression changes     |
| 5–7   | Moderate Pain  | Clear pain indicators, frowning, tension |
| 7–10  | Severe Pain    | Significant facial distortion            |
| 10+   | Very Severe    | Extreme distress, maximum expression     |

---

## 5. Pretrained Model Checkpoints

| File | Training Data | `num_outputs` | Notes |
|------|--------------|---------------|-------|
| `checkpoints/50342566/50343918_3/model_epoch4.pt` | UNBC-McMaster **+** University of Regina "Pain in Severe Dementia" | 40 | **Recommended for clinical use** (dementia patients) |
| `checkpoints/59448122/59448122_3/model_epoch13.pt` | UNBC-McMaster only | 7 | General healthy adults |

> **Important**: Subjects 66, 80, 97, 108, 121 from UNBC were excluded from training to avoid data leakage.

---

## 6. How to Run

### Prerequisites
- Python 3.6+
- PyTorch 1.6+ (tested with 2.x)
- CUDA 10.2+ (optional but recommended; falls back to CPU)
- All packages in `requirements.txt`

```bash
pip install -r requirements.txt
```

### Quick Test
```bash
python test.py                     # Uses UNBC + UofR model
python test.py -unbc_only          # Uses UNBC-only model
python test.py -test_framerate     # Also measures frames per second
```

### Simple Backend API (development/testing)
```bash
uvicorn simple_backend:app --host 0.0.0.0 --port 8000 --reload
```

### Full Clinical Backend API
```bash
uvicorn clinical_backend:app --host 0.0.0.0 --port 8001
```

### Frontend Web UI
```bash
python frontend_server.py          # Serves on http://localhost:3002
```

---

## 7. API Endpoints (Clinical Backend)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/check_patient_reference` | Check if patient has a reference image in the DB |
| `POST` | `/api/save_reference_image` | Upload and save a reference image for a patient |
| `POST` | `/api/assess_pain_with_report` | Full workflow: assess pain + generate PDF report |

All endpoints use `multipart/form-data`. The `patient_id` field is required on every call.

---

## 8. Known Issues and Technical Debt

### Critical / Bugs

1. **`start_backend.sh` references a missing file**: The script runs `python main.py` inside `backend/`, but `backend/main.py` does not exist. The script is currently broken. You need to either create `backend/main.py` or change the script to launch `clinical_backend.py` or `simple_backend.py` from the project root.

2. **Typo in `pain_detector.py`**: The method `verify_refenerece_image` (line ~100) is misspelled. `clinical_backend.py` calls it as `verify_reference_image`, which will raise an `AttributeError` at runtime. **Fix**: rename the method in `pain_detector.py` to `verify_reference_image`.

3. **`pretrained/` folder is empty**: FAN (Face Alignment Network) will attempt to download its pretrained weights from the internet on first run. On machines without internet access, this will fail silently. Pre-download the weights and place them in `pretrained/` and pass the path via the `fan_checkpoint` parameter of `PainDetector`.

4. **`WelodgeConnector` is SQLite, not real Welodge**: The connector in `database/welodge_connector.py` stores data in a local SQLite file (`welodge.db`). It is **not connected to the real CATO MAIOR / Welodge clinical system**. This is a stub for development. Real integration requires implementing the actual Welodge REST API or database connection.

### Security / Production Concerns

5. **CORS is fully open**: Both backends use `allow_origins=["*"]`. This must be restricted to the actual client origins before any production deployment.

6. **No authentication**: The `requirements.txt` includes `python-jose[cryptography]` (JWT library) and `aiofiles`, suggesting authentication was planned but **never implemented**. All API endpoints are completely unprotected. Any request with a valid `patient_id` can access any patient's data.

7. **Images stored as BLOBs in SQLite**: Storing binary image data in SQLite scales poorly. For production, images should be stored in a file system or object storage (e.g., S3/MinIO), with only paths in the database.

8. **No input sanitization on `patient_id`**: The `patient_id` is passed directly into SQL queries via parameterized statements (safe from injection), but there is no format validation (e.g., length, character set, existence in CatoMaior).

### Performance

9. **Single-threaded inference**: The model runs one prediction at a time. For concurrent clinical use, a job queue (e.g., Celery + Redis) should be introduced.

10. **Reference frames accumulate**: `PainDetector.add_references()` appends to a list without clearing. If the same `PainDetector` instance is reused across multiple assessments, reference frames from previous patients will pollute predictions. The instance should be reset between assessments, or re-instantiated per request.

---

## 9. What Has Been Built (Summary)

- [x] Core AI pain detection engine (`PainDetector` class)
- [x] Face alignment and landmark detection pipeline (FAN + S3FD)
- [x] Two pretrained model checkpoints (UNBC+UofR, UNBC-only)
- [x] Clinical REST API (FastAPI) with patient workflow
- [x] SQLite database stub for patient references and assessments
- [x] PDF report generation (ReportLab) with patient data, images, pain scale
- [x] Clinical web interface (multi-step HTML/JS workflow)
- [x] Image preprocessing utilities (resize, CLAHE, quality enhancement)
- [x] Model comparison and analysis utility scripts
- [x] Basic test scripts with example frames

---

## 10. What Still Needs to Be Done

### High Priority

- [ ] **Fix the `verify_reference_image` typo** in `pain_detector.py`
- [ ] **Create `backend/main.py`** or fix `start_backend.sh` to point to the correct entry point
- [ ] **Implement authentication**: JWT-based auth using `python-jose` (already in requirements). Protect all `/api/*` endpoints.
- [ ] **Restrict CORS** to known client origins
- [ ] **Real Welodge/CatoMaior integration**: Replace the SQLite stub with actual API calls to the CATO MAIOR backend. The specifications in `documents/Specifiche App Terapia Dolore.txt` describe the full integration requirements (patient registration, subscription management, DMS, notifications)

### Medium Priority

- [ ] **Pre-bundle FAN pretrained weights** in `pretrained/` for offline deployment
- [ ] **Reset reference frames between assessments** (`pain_detector.ref_frames = []` before each new patient session)
- [ ] **Move image storage** out of SQLite to file/object storage
- [ ] **Add assessment history UI**: The database records all assessments, but there is no UI to view past assessments or trend a patient's pain over time
- [ ] **Video/real-time stream support**: Currently only static image analysis is supported. Real-time webcam or video feed processing would significantly improve clinical usability
- [ ] **Error logging and monitoring**: Replace `print` statements and bare exceptions with structured logging

### Long Term (per specification documents)

- [ ] **Mobile app development**: The `documents/Specifiche App Terapia Dolore.txt` contains a full specification for a companion mobile app. Key features: patient onboarding with Italian Tax Code (Codice Fiscale) + health card (TEAM) validation, multi-channel notifications (push/SMS/email), medical report viewing, DMS integration
- [ ] **Notification system**: Push notification gateway with FCM/APNs, SMS, email; failed-message queue
- [ ] **DMS integration**: Middleware connecting CatoMaior to a Document Management System for archiving reports
- [ ] **Docker/containerisation**: Package the backend and model for reproducible deployment
- [ ] **Performance optimisation**: Consider replacing FAN/S3FD with a lighter face detector for higher FPS on lower-end hardware (currently ~9 FPS on RTX 2080 Ti)
- [ ] **Model retraining pipeline**: Allow fine-tuning on new patient populations without full retraining

---

## 11. Key Dependencies and Versions

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.10.0 | Neural network inference |
| `face-alignment` | 1.3.5 | Face detection + 68-point landmarks |
| `opencv-python` | 4.13.0.90 | Image I/O and processing |
| `scikit-image` | 0.26.0 | Piecewise affine transform |
| `fastapi` | (not pinned) | REST API framework |
| `reportlab` | ≥3.6.0 | PDF generation |
| `python-jose` | (not pinned) | JWT — planned but not yet used |

---

## 12. Contact and References

- **Original paper**: https://ieeexplore.ieee.org/document/9298886  
- **Face Alignment Network (FAN)**: https://github.com/1adrianb/face-alignment  
- **UNBC-McMaster Shoulder Pain Dataset**: Contact the original dataset maintainers for access  
- **University of Regina Pain in Severe Dementia Dataset**: Contact the original dataset maintainers for access  
- **CATO MAIOR specification**: `documents/Specifiche App Terapia Dolore.txt` (Italian)

---

*Good luck with the project. The AI core is solid and production-ready for basic use; the main work remaining is in security hardening, real clinical system integration, and the mobile app.*
