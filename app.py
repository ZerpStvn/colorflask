from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
from collections import Counter

app = Flask(__name__)

def detect_named_colors(image):
    image_np = np.array(image)
    
    # Apply a light Gaussian blur to reduce noise but retain color info
    image_np = cv2.GaussianBlur(image_np, (3, 3), 0)
    
    # Convert to HSV color space for easier color detection
    hsv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Define more precise color ranges in HSV
    color_ranges = {
        "red": [((0, 70, 50), (10, 255, 255)), ((160, 70, 50), (180, 255, 255))],  # Two ranges for red (0-10 and 160-180)
        "green": [(36, 50, 50), (89, 255, 255)],
        "blue": [(90, 50, 50), (128, 255, 255)],
        "yellow": [(15, 50, 50), (35, 255, 255)],
        "cyan": [(85, 50, 50), (95, 255, 255)],
        "magenta": [(140, 50, 50), (170, 255, 255)],
        "purple": [(125, 50, 50), (150, 255, 255)],
        "orange": [(10, 100, 100), (25, 255, 255)],
        "pink": [(150, 50, 50), (170, 255, 255)],
        "brown": [(10, 50, 50), (20, 255, 150)],
        "gray": [(0, 0, 50), (180, 50, 200)],
        "black": [(0, 0, 0), (180, 255, 50)],
        "white": [(0, 0, 200), (180, 25, 255)] 
    }
    
    detected_colors = Counter()
    
    # Calculate the total number of pixels to ignore black pixels in the percentage calculation
    total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
    non_black_pixels = total_pixels - np.sum(cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([180, 255, 50]))) // 255

    for color_name, ranges in color_ranges.items():
        if isinstance(ranges[0], tuple):
            ranges = [ranges]
        
        color_ratio_sum = 0
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
            color_ratio_sum += (np.sum(mask) / 255) / total_pixels
        
        # If the detected color occupies more than 1% of non-black pixels, consider it significant
        if color_ratio_sum > 0.01:
            detected_colors[color_name] = (color_ratio_sum / non_black_pixels) * 100

    # Sort colors by percentage
    sorted_colors = detected_colors.most_common()

    return sorted_colors

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream)  
    detected_colors = detect_named_colors(img)

    return jsonify({'colors': detected_colors})

if __name__ == '__main__':
    app.run(debug=True)
