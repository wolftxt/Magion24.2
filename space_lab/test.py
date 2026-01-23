import os
import sys
from glob import glob
from pathlib import Path

# 1. Get the directory where test.py is located
BASE_DIR = Path(__file__).resolve().parent

# 2. Add 'program' folder to sys.path using a relative path
program_path = BASE_DIR / "program"
sys.path.append(str(program_path))

# Import your functions after the path has been added
try:
    from calculateSpeed import calculate, get_time
except ImportError:
    print("Error: Could not find calculateSpeed.py in the 'program' directory.")
    sys.exit(1)

# 3. Define the relative path to images
image_dir = BASE_DIR / "images" / "magion"


def main():
    # Gather all image files
    search_path = os.path.join(image_dir, "*")
    image_files = [f for f in glob(search_path) if os.path.isfile(f)]

    if len(image_files) < 2:
        print(f"Not enough images found in {image_dir} to perform a comparison.")
        return

    # 4. Sort images by EXIF timestamp
    try:
        image_files.sort(key=get_time)
    except Exception as e:
        print(f"EXIF sort failed ({e}), falling back to filename sort.")
        image_files.sort()

    # 5. Iterate and compare
    print(f"{'Second Image Name':<30} | {'Calculated Speed (km/s)':<20}")
    print("-" * 55)

    total = 0
    unused = 0

    for i in range(len(image_files) - 1):
        img1 = image_files[i]
        img2 = image_files[i + 1]


        speed = calculate(img1, img2)
        if speed == -1:
            unused += 1
            continue
        total += speed
        image_name = os.path.basename(img2)
        print(f"{image_name:<30} | {speed:.4f} km/s")
    images_used = len(image_files) - unused
    print(total / images_used)

if __name__ == "__main__":
    main()