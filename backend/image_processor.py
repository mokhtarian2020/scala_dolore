#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io

class ImageProcessor:
    """Utility class for processing images to be compatible with the pain detection model"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: tuple = (640, 480)) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create canvas and center the image
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    @staticmethod
    def enhance_image_quality(image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better face detection"""
        # Convert to PIL for enhancements
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    @staticmethod
    def normalize_lighting(image: np.ndarray) -> np.ndarray:
        """Normalize lighting conditions using CLAHE"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    @staticmethod
    def preprocess_for_model(image: np.ndarray, enhance_quality: bool = True) -> np.ndarray:
        """Complete preprocessing pipeline for the pain detection model"""
        processed = image.copy()
        
        # Resize to reasonable size for processing
        processed = ImageProcessor.resize_image(processed, (640, 480))
        
        # Enhance image quality if requested
        if enhance_quality:
            processed = ImageProcessor.enhance_image_quality(processed)
        
        # Normalize lighting
        processed = ImageProcessor.normalize_lighting(processed)
        
        return processed
    
    @staticmethod
    def convert_pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format"""
        # Convert to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array and then to BGR for OpenCV
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    
    @staticmethod
    def convert_opencv_to_pil(opencv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
    
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Load image from bytes and convert to OpenCV format"""
        # Load with PIL first
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = ImageProcessor.convert_pil_to_opencv(pil_image)
        
        return opencv_image
    
    @staticmethod
    def validate_image_format(image: np.ndarray) -> bool:
        """Validate that image is in correct format for processing"""
        if image is None:
            return False
        
        if len(image.shape) != 3:
            return False
        
        if image.shape[2] != 3:
            return False
        
        if image.dtype != np.uint8:
            return False
        
        return True
    
    @staticmethod
    def auto_rotate_image(image: np.ndarray) -> np.ndarray:
        """Auto-rotate image based on EXIF data (for uploaded images)"""
        # This is a simplified version - for full EXIF support, 
        # you'd need to handle this at the PIL level before conversion
        return image
    
    @staticmethod
    def crop_face_region(image: np.ndarray, face_landmarks) -> np.ndarray:
        """Crop image to focus on face region (if landmarks are available)"""
        if face_landmarks is None or len(face_landmarks) == 0:
            return image
        
        # Get bounding box from landmarks
        landmarks = face_landmarks[0]  # Take first face
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        # Calculate bounding box with some padding
        padding = 0.3  # 30% padding
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Add padding
        x_min = max(0, x_min - int(width * padding))
        y_min = max(0, y_min - int(height * padding))
        x_max = min(image.shape[1], x_max + int(width * padding))
        y_max = min(image.shape[0], y_max + int(height * padding))
        
        # Crop the image
        cropped = image[y_min:y_max, x_min:x_max]
        
        return cropped if cropped.size > 0 else image
    
    @staticmethod
    def prepare_for_pain_detection(image_bytes: bytes, enhance: bool = True) -> tuple:
        """
        Complete pipeline to prepare an uploaded image for pain detection
        
        Returns:
            tuple: (processed_image, success, error_message)
        """
        try:
            # Load image from bytes
            image = ImageProcessor.load_image_from_bytes(image_bytes)
            
            # Validate format
            if not ImageProcessor.validate_image_format(image):
                return None, False, "Invalid image format"
            
            # Preprocess for model
            processed = ImageProcessor.preprocess_for_model(image, enhance)
            
            return processed, True, ""
            
        except Exception as e:
            return None, False, f"Error processing image: {str(e)}"
    
    @staticmethod
    def get_image_stats(image: np.ndarray) -> dict:
        """Get statistics about the image"""
        if image is None:
            return {}
        
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "mean_brightness": float(np.mean(image)),
            "std_brightness": float(np.std(image)),
            "min_value": int(np.min(image)),
            "max_value": int(np.max(image)),
            "size_bytes": image.nbytes
        }
