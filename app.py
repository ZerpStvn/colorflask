from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
from collections import Counter

app = Flask(__name__)

def detect_named_colors(image):
    image_np = np.array(image)
    image_np = cv2.GaussianBlur(image_np, (3, 3), 0)
    hsv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Expanded color ranges with more categories
    color_ranges = {
        "red": [((0, 70, 50), (10, 255, 255)), ((160, 70, 50), (180, 255, 255))],
        "dark red": [((0, 100, 50), (5, 255, 100))],
        "light red": [((0, 50, 50), (10, 200, 255))],
        "green": [((36, 50, 50), (89, 255, 255))],
        "dark green": [((36, 100, 20), (89, 200, 100))],
        "light green": [((50, 50, 50), (80, 150, 255))],
        "blue": [((90, 50, 50), (128, 255, 255))],
        "dark blue": [((90, 70, 50), (120, 255, 150))],
        "light blue": [((100, 50, 100), (128, 150, 255))],
        "yellow": [((15, 50, 50), (35, 255, 255))],
        "dark yellow": [((15, 100, 50), (30, 200, 255))],
        "light yellow": [((20, 50, 100), (35, 150, 255))],
        "cyan": [((85, 50, 50), (95, 255, 255))],
        "dark cyan": [((85, 100, 50), (95, 200, 255))],
        "magenta": [((140, 50, 50), (170, 255, 255))],
        "purple": [((125, 50, 50), (150, 255, 255))],
        "dark purple": [((125, 100, 50), (150, 200, 100))],
        "light purple": [((130, 50, 100), (160, 150, 255))],
        "orange": [((10, 100, 100), (25, 255, 255))],
        "light orange": [((15, 50, 100), (30, 150, 255))],
        "brown": [((10, 50, 50), (20, 255, 150))],
        "gray": [((0, 0, 50), (180, 50, 200))],
        "dark gray": [((0, 0, 30), (180, 50, 150))],
        "light gray": [((0, 0, 150), (180, 50, 255))],
        "black": [((0, 0, 0), (180, 255, 50))],
        "white": [((0, 0, 200), (180, 25, 255))],
        "beige": [((10, 0, 200), (20, 50, 255))],
        "peach": [((10, 50, 150), (20, 150, 255))]
    }

    detected_colors = Counter()
    non_black_mask = cv2.inRange(hsv_img, np.array([0, 0, 50]), np.array([180, 255, 255]))
    non_black_pixels = cv2.countNonZero(non_black_mask)
    if non_black_pixels == 0:
        non_black_pixels = 1

    for color_name, ranges in color_ranges.items():
        color_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
            color_mask = cv2.bitwise_or(color_mask, mask)

        color_pixels = cv2.countNonZero(color_mask)
        percentage = (color_pixels / non_black_pixels) * 100
        if percentage > 0.01:
            detected_colors[color_name] = percentage

    # Print detected colors and percentages for debugging
    print("Detected colors and percentages:")
    for color, percentage in detected_colors.items():
        print(f"{color}: {percentage:.4f}%")

    sorted_colors = detected_colors.most_common()
    return sorted_colors

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img = Image.open(file.stream)
        detected_colors = detect_named_colors(img)
        return jsonify({'colors': detected_colors}), 200
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
