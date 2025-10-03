#!/usr/bin/env python3

import os
import cv2
import numpy as np
import json
from PIL import Image
from google import genai
from google.genai import types

def detect_objects_in_image(image_path, target_object="tomato can"):
    """
    Detect objects in an image using Google Gemini API and return bounding boxes
    """
    try:
        # Initialize Gemini client
        client = genai.Client()
        
        # Load image
        image = Image.open(image_path)
        width, height = image.size
        
        # Create prompt for object detection
        prompt = f"Detect all prominent items in the image, especially looking for a {target_object}. " \
                "For each detected object, provide: object name, confidence score, and box_2d coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000. " \
                "Focus on finding the {target_object} and provide its bounding box coordinates. " \
                "Return the response as a JSON array with objects containing 'object', 'confidence', and 'box_2d' fields."
        
        # Configure response format
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        # Generate content using Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt],
            config=config
        )
        
        # Parse response
        print(f"Raw response: {response.text}")
        bounding_boxes = json.loads(response.text)
        
        # Convert normalized coordinates to absolute pixel coordinates
        converted_bounding_boxes = []
        for i, bounding_box in enumerate(bounding_boxes):
            print(f"Detection {i}: {bounding_box}")
            
            # Handle different possible response formats
            if "box_2d" in bounding_box:
                box_coords = bounding_box["box_2d"]
            elif "bbox" in bounding_box:
                box_coords = bounding_box["bbox"]
            elif "box" in bounding_box:
                box_coords = bounding_box["box"]
            else:
                print(f"Warning: No bounding box coordinates found in detection {i}")
                continue
            
            abs_y1 = int(box_coords[0] / 1000 * height)
            abs_x1 = int(box_coords[1] / 1000 * width)
            abs_y2 = int(box_coords[2] / 1000 * height)
            abs_x2 = int(box_coords[3] / 1000 * width)
            
            # Try different possible object name fields
            object_name = "unknown"
            if "object" in bounding_box:
                object_name = bounding_box["object"]
            elif "name" in bounding_box:
                object_name = bounding_box["name"]
            elif "label" in bounding_box:
                object_name = bounding_box["label"]
            elif "class" in bounding_box:
                object_name = bounding_box["class"]
            
            # Try different possible confidence fields
            confidence = 1.0
            if "confidence" in bounding_box:
                confidence = bounding_box["confidence"]
            elif "score" in bounding_box:
                confidence = bounding_box["score"]
            elif "prob" in bounding_box:
                confidence = bounding_box["prob"]
            
            converted_bounding_boxes.append({
                "object": object_name,
                "box": [abs_x1, abs_y1, abs_x2, abs_y2],
                "confidence": confidence
            })
        
        return converted_bounding_boxes, width, height
        
    except Exception as e:
        print(f"Error detecting objects: {e}")
        return [], 0, 0

def draw_detection_results(image_path, bounding_boxes, output_path):
    """
    Draw bounding boxes and center points on the image
    """
    try:
        # Load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return False
        
        # Draw bounding boxes and center points
        for i, detection in enumerate(bounding_boxes):
            x1, y1, x2, y2 = detection["box"]
            object_name = detection["object"]
            confidence = detection.get("confidence", 1.0)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw center point
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw label with confidence
            label = f"{object_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw center coordinates
            center_label = f"({center_x}, {center_y})"
            cv2.putText(img, center_label, (center_x + 10, center_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print(f"Detected {object_name} at center ({center_x}, {center_y}) with confidence {confidence:.2f}")
        
        # Save the result
        cv2.imwrite(output_path, img)
        print(f"Detection results saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error drawing detection results: {e}")
        return False

def test_object_detection():
    """
    Test object detection on the provided images
    """
    # Check if API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("Error: GOOGLE_API_KEY environment variable not set")
        return False
    
    # Image paths
    image_folder = "/Docker-LLM-Spot-Image/image"
    images = ["IMG_6277.jpg", "IMG_6278.jpg"]
    target_object = "pringles can"
    
    # Create output directory
    output_dir = "/Docker-LLM-Spot-Image/catkin_ws/images"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        print(f"\nProcessing {image_name}...")
        print(f"Looking for: {target_object}")
        
        # Detect objects
        bounding_boxes, width, height = detect_objects_in_image(image_path, target_object)
        
        if not bounding_boxes:
            print(f"No objects detected in {image_name}")
            continue
        
        print(f"Image size: {width} x {height}")
        print(f"Found {len(bounding_boxes)} objects:")
        
        for detection in bounding_boxes:
            print(f"  - {detection['object']}: {detection['box']} (confidence: {detection.get('confidence', 1.0):.2f})")
        
        # Create output filename
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}_detection_result.jpg")
        
        # Draw results
        if draw_detection_results(image_path, bounding_boxes, output_path):
            success_count += 1
            print(f"Successfully processed {image_name}")
        else:
            print(f"Failed to process {image_name}")
    
    print(f"\nDetection test completed. Successfully processed {success_count}/{len(images)} images.")
    return success_count > 0

def main():
    """
    Main function to run the object detection test
    """
    print("Object Detection Test")
    print("====================")
    print("This script will detect objects in static images using Google Gemini API")
    print("Target object: tomato can")
    print("Images: IMG_6277.jpg, IMG_6278.jpg")
    print()
    
    try:
        success = test_object_detection()
        if success:
            print("\nTest completed successfully!")
        else:
            print("\nTest failed or no objects detected.")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == '__main__':
    main()
