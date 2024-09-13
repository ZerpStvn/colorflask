from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
from collections import Counter

app = Flask(__name__)

# Function to detect the most prominent colors
def detect_named_colors(image):
    image_np = np.array(image)
    
    # Apply Gaussian blur to reduce noise
    image_np = cv2.GaussianBlur(image_np, (5, 5), 0)
    
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Define a broad range of colors in HSV
    color_ranges = {
        "red": [(0, 70, 50), (10, 255, 255)],
        "dark red": [(160, 70, 50), (180, 255, 255)],
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

    # Loop through color ranges and calculate the percentage of pixels matching each color
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        color_ratio = (np.sum(mask) / 255) / (mask.shape[0] * mask.shape[1])
        
        # Only consider colors that occupy more than 1% of the image
        if color_ratio > 0.01:
            detected_colors[color_name] = color_ratio * 100  # Store percentage

    # Sort detected colors by percentage (dominance)
    sorted_colors = detected_colors.most_common()

    return sorted_colors

# Flask route to accept image upload and detect colors
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream)  # Open the image using PIL

    # Detect named colors
    detected_colors = detect_named_colors(img)

    # Return the detected colors with their percentages
    return jsonify({'colors': detected_colors})

if __name__ == '__main__':
    app.run(debug=True)
