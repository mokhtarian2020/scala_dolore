#!/usr/bin/env python3

from pain_detector import PainDetector
import cv2
import os

def compare_models():
    """Compare pain detection results between different models"""
    
    # Load example images
    example_frames_dir = 'example_frames'
    reference_image_path = os.path.join(example_frames_dir, 'example-reference-frame.png')
    target_image_path = os.path.join(example_frames_dir, 'example-target-frame.png')
    
    ref_frame = cv2.imread(reference_image_path)
    target_frame = cv2.imread(target_image_path)
    
    print("🔬 PAIN DETECTION MODEL COMPARISON")
    print("="*60)
    
    # Test Model 1: Both UNBC + UofR datasets
    print("📊 Model 1: Trained on UNBC + University of Regina datasets")
    print("-"*40)
    
    pain_detector_1 = PainDetector(
        image_size=160, 
        checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', 
        num_outputs=40
    )
    
    print('Device:', pain_detector_1.device)
    pain_detector_1.add_references([ref_frame, ref_frame, ref_frame])
    pain_score_1 = pain_detector_1.predict_pain(target_frame)
    
    print(f"🎯 Pain Score: {pain_score_1:.6f}")
    
    # Interpret score 1
    if pain_score_1 < 2:
        interpretation_1 = "Very Low Pain"
    elif pain_score_1 < 4:
        interpretation_1 = "Low Pain"
    elif pain_score_1 < 6:
        interpretation_1 = "Moderate Pain"
    elif pain_score_1 < 8:
        interpretation_1 = "High Pain"
    else:
        interpretation_1 = "Very High Pain"
    
    print(f"📊 Interpretation: {interpretation_1}")
    print("="*60)
    
    # Test Model 2: UNBC only
    print("📊 Model 2: Trained on UNBC dataset only")
    print("-"*40)
    
    pain_detector_2 = PainDetector(
        image_size=160, 
        checkpoint_path='checkpoints/59448122/59448122_3/model_epoch13.pt', 
        num_outputs=7
    )
    
    print('Device:', pain_detector_2.device)
    pain_detector_2.add_references([ref_frame, ref_frame, ref_frame])
    pain_score_2 = pain_detector_2.predict_pain(target_frame)
    
    print(f"🎯 Pain Score: {pain_score_2:.6f}")
    
    # Interpret score 2
    if pain_score_2 < 2:
        interpretation_2 = "Very Low Pain"
    elif pain_score_2 < 4:
        interpretation_2 = "Low Pain"
    elif pain_score_2 < 6:
        interpretation_2 = "Moderate Pain"
    elif pain_score_2 < 8:
        interpretation_2 = "High Pain"
    else:
        interpretation_2 = "Very High Pain"
    
    print(f"📊 Interpretation: {interpretation_2}")
    print("="*60)
    
    # Compare results
    print("🔍 COMPARISON SUMMARY")
    print("-"*40)
    print(f"UNBC + UofR Model: {pain_score_1:.6f} ({interpretation_1})")
    print(f"UNBC Only Model:   {pain_score_2:.6f} ({interpretation_2})")
    
    diff = abs(pain_score_1 - pain_score_2)
    print(f"Difference:        {diff:.6f}")
    
    if diff < 0.5:
        consistency = "Very Consistent"
    elif diff < 1.0:
        consistency = "Consistent"
    elif diff < 2.0:
        consistency = "Somewhat Different"
    else:
        consistency = "Significantly Different"
    
    print(f"Model Consistency: {consistency}")
    print("="*60)

if __name__ == "__main__":
    compare_models()
