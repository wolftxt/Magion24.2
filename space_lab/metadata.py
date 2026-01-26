import os
import random
from PIL import Image
from PIL.ExifTags import TAGS


def analyze_random_image():
    # Define the relative path to the images directory
    image_folder = os.path.join("images", "magion")

    # Check if the directory exists
    if not os.path.exists(image_folder):
        print(f"Error: The directory '{image_folder}' was not found.")
        return

    # List all files in the directory and filter for common image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.webp')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

    if not images:
        print("No valid images found in the directory.")
        return

    # Pick a random image
    random_image_name = random.choice(images)
    image_path = os.path.join(image_folder, random_image_name)

    print(f"--- Analyzing Image: {random_image_name} ---")

    try:
        with Image.open(image_path) as img:
            # 1. Print Basic File Information
            print(f"Format: {img.format}")
            print(f"Size: {img.width}x{img.height}")
            print(f"Mode: {img.mode}")
            print("-" * 30)

            # 2. Extract EXIF Metadata
            exif_data = img.getexif()

            if not exif_data:
                print("No EXIF metadata found in this image.")
            else:
                print("Available Metadata:")
                for tag_id, value in exif_data.items():
                    # Get the human-readable name for the tag ID
                    tag_name = TAGS.get(tag_id, tag_id)
                    print(f"{tag_name}: {value}")

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")


if __name__ == "__main__":
    analyze_random_image()