import os
import datetime
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS


def get_image_datetime(image_path):
    """Extracts datetime object from EXIF or file modification time."""
    try:
        image = Image.open(image_path)
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "DateTimeOriginal":
                    # EXIF format is 'YYYY:MM:DD HH:MM:SS'
                    return datetime.datetime.strptime(value, '%Y:%m:%d %H:%M:%S')

        return datetime.datetime.fromtimestamp(os.path.getmtime(image_path))
    except Exception:
        return datetime.datetime.fromtimestamp(os.path.getmtime(image_path))


# 1. Define the base images directory
base_folder = Path("images")
extensions = (".jpg", ".jpeg", ".png", ".tiff")

if not base_folder.exists():
    print(f"Error: The directory '{base_folder}' does not exist.")
else:
    # 2. Iterate through every item in 'images'
    # We use .iterdir() and check if it's a directory
    subfolders = sorted([d for d in base_folder.iterdir() if d.is_dir()])

    print(f"{'Folder Name':<20} | {'Count':<6} | {'Min (s)':<10} | {'Max (s)':<10} | {'Avg (s)':<10}")
    print("-" * 65)

    for folder in subfolders:
        # Get and sort images in this specific subfolder
        image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in extensions])

        if len(image_files) < 2:
            print(f"{folder.name:<20} | {len(image_files):<6} | --- Not enough images ---")
            continue

        # 3. Extract timestamps
        timestamps = [get_image_datetime(f) for f in image_files]

        # 4. Calculate deltas
        deltas = []
        for i in range(len(timestamps) - 1):
            diff = (timestamps[i + 1] - timestamps[i]).total_seconds()
            deltas.append(diff)

        # 5. Compute Stats
        min_val = min(deltas)
        max_val = max(deltas)
        avg_val = sum(deltas) / len(deltas)

        # 6. Print Row
        print(f"{folder.name:<20} | {len(image_files):<6} | {min_val:<10.2f} | {max_val:<10.2f} | {avg_val:<10.2f}")