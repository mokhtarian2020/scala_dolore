#!/usr/bin/env python3

from pain_detector import PainDetector
import cv2
import os
import numpy as np

def analyze_image_details():
    """Analyze the example images in detail"""
    
    print("🖼️  IMAGE ANALYSIS REPORT")
    print("="*60)
    
    # Load example images
    example_frames_dir = 'example_frames'
    reference_image_path = os.path.join(example_frames_dir, 'example-reference-frame.png')
    target_image_path = os.path.join(example_frames_dir, 'example-target-frame.png')
    
    ref_frame = cv2.imread(reference_image_path)
    target_frame = cv2.imread(target_image_path)
    
    # Image properties
    print("📏 IMAGE PROPERTIES")
    print("-"*40)
    print(f"Reference Image:")
    print(f"  - Path: {reference_image_path}")
    print(f"  - Shape: {ref_frame.shape} (Height x Width x Channels)")
    print(f"  - Size: {ref_frame.size} pixels")
    print(f"  - Data type: {ref_frame.dtype}")
    print(f"  - File size: {os.path.getsize(reference_image_path)} bytes")
    
    print(f"\nTarget Image:")
    print(f"  - Path: {target_image_path}")
    print(f"  - Shape: {target_frame.shape} (Height x Width x Channels)")
    print(f"  - Size: {target_frame.size} pixels")
    print(f"  - Data type: {target_frame.dtype}")
    print(f"  - File size: {os.path.getsize(target_image_path)} bytes")
    
    # Basic image statistics
    print("\n📊 IMAGE STATISTICS")
    print("-"*40)
    print("Reference Image:")
    print(f"  - Mean pixel value: {np.mean(ref_frame):.2f}")
    print(f"  - Std deviation: {np.std(ref_frame):.2f}")
    print(f"  - Min pixel value: {np.min(ref_frame)}")
    print(f"  - Max pixel value: {np.max(ref_frame)}")
    
    print("\nTarget Image:")
    print(f"  - Mean pixel value: {np.mean(target_frame):.2f}")
    print(f"  - Std deviation: {np.std(target_frame):.2f}")
    print(f"  - Min pixel value: {np.min(target_frame)}")
    print(f"  - Max pixel value: {np.max(target_frame)}")
    
    print("\n🔬 FACE DETECTION ANALYSIS")
    print("-"*40)
    
    # Initialize pain detector to test face detection
    pain_detector = PainDetector(
        image_size=160, 
        checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', 
        num_outputs=40
    )
    
    # Test face detection on reference image
    print("Testing face detection on reference image...")
    try:
        landmarks_ref = pain_detector.face_detector(ref_frame)
        if landmarks_ref is not None and len(landmarks_ref) > 0:
            print(f"✓ Face detected! Found {len(landmarks_ref)} face(s)")
            print(f"  - Landmarks shape: {landmarks_ref[0].shape}")
            print(f"  - Number of landmark points: {landmarks_ref[0].shape[0]}")
        else:
            print("✗ No face detected in reference image")
    except Exception as e:
        print(f"✗ Error during face detection: {e}")
    
    # Test face detection on target image
    print("\nTesting face detection on target image...")
    try:
        landmarks_target = pain_detector.face_detector(target_frame)
        if landmarks_target is not None and len(landmarks_target) > 0:
            print(f"✓ Face detected! Found {len(landmarks_target)} face(s)")
            print(f"  - Landmarks shape: {landmarks_target[0].shape}")
            print(f"  - Number of landmark points: {landmarks_target[0].shape[0]}")
        else:
            print("✗ No face detected in target image")
    except Exception as e:
        print(f"✗ Error during face detection: {e}")
    
    print("\n🎯 PAIN ANALYSIS")
    print("-"*40)
    
    # Run pain detection
    try:
        pain_detector.add_references([ref_frame, ref_frame, ref_frame])
        pain_score = pain_detector.predict_pain(target_frame)
        
        print(f"Pain Score (PSPI): {pain_score:.6f}")
        
        # Pain level classification
        if pain_score < 1:
            level = "No Pain"
        elif pain_score < 2:
            level = "Minimal Pain"
        elif pain_score < 3:
            level = "Mild Pain"
        elif pain_score < 4:
            level = "Moderate Pain"
        elif pain_score < 6:
            level = "Moderate-Severe Pain"
        elif pain_score < 8:
            level = "Severe Pain"
        else:
            level = "Very Severe Pain"
        
        print(f"Pain Level: {level}")
        
        # PSPI scale explanation
        print("\nPSPI Scale (Prkachin and Solomon Pain Intensity):")
        print("  0-1: No Pain")
        print("  1-2: Minimal Pain") 
        print("  2-3: Mild Pain")
        print("  3-4: Moderate Pain")
        print("  4-6: Moderate-Severe Pain")
        print("  6-8: Severe Pain")
        print("  8+:  Very Severe Pain")
        
    except Exception as e:
        print(f"✗ Error during pain analysis: {e}")
    
    print("="*60)
    print("✓ Analysis completed!")

if __name__ == "__main__":
    analyze_image_details()
