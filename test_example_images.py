#!/usr/bin/env python3

from pain_detector import PainDetector
import cv2
import os

def test_example_images():
    """Test pain detection on example images"""
    
    # Initialize pain detector (using the model trained on both datasets)
    print("Loading pain detector...")
    pain_detector = PainDetector(
        image_size=160, 
        checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', 
        num_outputs=40
    )
    
    print('Device:', pain_detector.device)
    print("="*60)
    
    # Load example images
    example_frames_dir = 'example_frames'
    reference_image_path = os.path.join(example_frames_dir, 'example-reference-frame.png')
    target_image_path = os.path.join(example_frames_dir, 'example-target-frame.png')
    
    if not os.path.exists(reference_image_path):
        print(f"Reference image not found: {reference_image_path}")
        return
    
    if not os.path.exists(target_image_path):
        print(f"Target image not found: {target_image_path}")
        return
    
    # Load images
    print(f"Loading reference image: {reference_image_path}")
    ref_frame = cv2.imread(reference_image_path)
    
    print(f"Loading target image: {target_image_path}")
    target_frame = cv2.imread(target_image_path)
    
    if ref_frame is None:
        print("Failed to load reference image")
        return
    
    if target_frame is None:
        print("Failed to load target image")
        return
    
    print(f"Reference image shape: {ref_frame.shape}")
    print(f"Target image shape: {target_frame.shape}")
    print("="*60)
    
    # Add reference frames (using the same reference multiple times as in original test)
    print("Adding reference frames...")
    try:
        pain_detector.add_references([ref_frame, ref_frame, ref_frame])
        print("✓ Reference frames added successfully")
    except Exception as e:
        print(f"✗ Error adding reference frames: {e}")
        return
    
    # Predict pain on target frame
    print("Analyzing target frame...")
    try:
        pain_score = pain_detector.predict_pain(target_frame)
        print(f"✓ Pain prediction successful")
        print("="*60)
        print(f"🔍 PAIN SCORE (PSPI): {pain_score:.6f}")
        print("="*60)
        
        # Interpret the score
        if pain_score < 2:
            interpretation = "Very Low Pain"
        elif pain_score < 4:
            interpretation = "Low Pain"
        elif pain_score < 6:
            interpretation = "Moderate Pain"
        elif pain_score < 8:
            interpretation = "High Pain"
        else:
            interpretation = "Very High Pain"
        
        print(f"📊 Interpretation: {interpretation}")
        
    except Exception as e:
        print(f"✗ Error predicting pain: {e}")
        return
    
    print("="*60)
    print("✓ Analysis completed successfully!")

if __name__ == "__main__":
    test_example_images()
