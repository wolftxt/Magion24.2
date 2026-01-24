import os
import cv2
import numpy as np
from glob import glob

BASE_DIR = "."
IMAGE_DIR = "images/magion"
PORT_HOLE_CROP = 0.6
CHANGE_THRESHOLD = 20

# Define colors in BGR, then we convert them to LAB for math
TARGET_COLORS_BGR = {
    "blue": [255, 0, 0],
    "green": [0, 255, 0],
    "white": [255, 255, 255],
    "brown": [42, 42, 165],
    "red": [0, 0, 255]
}


def bgr_to_lab(color_list):
    """Converts a single BGR list to a LAB pixel."""
    pixel = np.uint8([[color_list]])
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0][0]


# Pre-convert targets to LAB
TARGET_LAB = {name: bgr_to_lab(color) for name, color in TARGET_COLORS_BGR.items()}


def get_image_time(path):
    return os.path.getmtime(path)


def analyze_colors_lab(img, mask):
    # Convert image to LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Extract only pixels within the mask
    valid_pixels = img_lab[mask == 255]
    if len(valid_pixels) == 0:
        return None

    stats = {color: 0 for color in TARGET_LAB}

    # Create a matrix of target colors for vectorized distance calculation
    names = list(TARGET_LAB.keys())
    target_matrix = np.array([TARGET_LAB[name] for name in names])

    # For every pixel, find the closest target color using Euclidean distance
    # This ensures every pixel is assigned to EXACTLY one category
    for pixel in valid_pixels:
        distances = np.sqrt(np.sum((target_matrix - pixel) ** 2, axis=1))
        closest_index = np.argmin(distances)
        stats[names[closest_index]] += 1

    total = len(valid_pixels)
    return {color: (count / total) * 100 for color, count in stats.items()}


def main():
    image_paths = glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob(os.path.join(IMAGE_DIR, "*.png"))
    image_paths.sort(key=get_image_time)

    prev_gray = None
    print(f"Analyzing {len(image_paths)} images (Perceptual LAB Distance)...\n")

    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue

        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Porthole Mask
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, (w // 2, h // 2), int(min(w, h) * PORT_HOLE_CROP / 2), 255, -1)

        # Overlap Detection
        if prev_gray is None:
            final_mask = center_mask
        else:
            frame_diff = cv2.absdiff(prev_gray, gray)
            _, diff_mask = cv2.threshold(frame_diff, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)
            final_mask = cv2.bitwise_and(center_mask, diff_mask)

        results = analyze_colors_lab(img, final_mask)
        prev_gray = gray

        print(f"File: {os.path.basename(path)}")
        if results is None:
            print(" - No significant new data")
        else:
            # Sort by percentage
            for color in sorted(results, key=results.get, reverse=True):
                print(f" - {color}: {results[color]:.2f}%")

            # Sanity Check
            total_sum = sum(results.values())
            print(f" [Total: {total_sum:.1f}%]")
        print("-" * 20)


if __name__ == "__main__":
    main()