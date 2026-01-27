from datetime import datetime
from exif import Image
import os
import sys
from glob import glob
from pathlib import Path
import statistics

BASE_DIR = Path(__file__).resolve().parent
program_path = BASE_DIR / "program"
sys.path.append(str(program_path))

try:
    from calculateSpeed import calculate
except ImportError:
    sys.exit(1)

PARENT_IMAGE_DIR = BASE_DIR / "images"


def get_time(image):
    with open(image, "rb") as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, "%Y:%m:%d %H:%M:%S")
    return time


def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.total_seconds()


def main():
    if not os.path.exists(PARENT_IMAGE_DIR):
        return

    # Identify subfolders within the highlights path
    # Using glob to find actual subdirectories if they exist
    subfolders = sorted([str(f) for f in PARENT_IMAGE_DIR.iterdir() if f.is_dir()])

    # Fallback if no subfolders exist and astur3 is the target itself
    if not subfolders:
        subfolders = [str(PARENT_IMAGE_DIR)]

    all_subfolder_averages = []
    all_speed_values = []

    for image_dir in subfolders:
        folder_name = os.path.basename(image_dir)
        search_path = os.path.join(image_dir, "*")
        image_files = [f for f in glob(search_path) if os.path.isfile(f)]

        print(f"\nFOLDER: {folder_name}")
        if len(image_files) < 2:
            print(f"Not enough images found in {image_dir}")
            continue

        try:
            image_files.sort(key=get_time)
        except Exception:
            image_files.sort()

        results = []
        for i in range(len(image_files) - 1):
            img1 = image_files[i]
            img2 = image_files[i + 1]
            time_difference = get_time_difference(img1, img2)
            try:
                speed, inliers = calculate(img1, img2, time_difference, 420000, 0) ## Height of ISS in m
            except Exception as e:
                print(f"Error processing {os.path.basename(img2)}: {e}")
                speed = -1
                inliers = 0

            if speed != -1:
                results.append({
                    "name": os.path.basename(img2),
                    "speed": speed,
                    "confidence": inliers
                })
                all_speed_values.append(speed)

        all_raw_speeds = [res["speed"] for res in results]
        stdev_of_all_speeds = statistics.stdev(all_raw_speeds)
        # Filter for best results (top 25%)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        top_n = max(1, len(results) // 4)
        best_results = results[:top_n]

        print(f"{'Image Name':<30} | {'Speed (km/s)':<15} | {'Inliers'}")
        print("-" * 60)

        folder_speeds = []
        for res in best_results:
            print(f"{res['name']:<30} | {res['speed']:.5g} km/s   | {res['confidence']}")
            folder_speeds.append(res["speed"])

        if folder_speeds:
            folder_avg = sum(folder_speeds) / len(folder_speeds)
            print("-" * 60)
            print(f"Final Filtered Average Speed for {folder_name}: {folder_avg:.5g} km/s")
            print(f"Standard deviation for all images in {folder_name}: {stdev_of_all_speeds:.5g} km/s")
            all_subfolder_averages.append(folder_avg)
        else:
            print("No valid matches found.")
        print("=" * 60)

    # --- Final Calculations ---
    if len(all_subfolder_averages) > 0:
        stdev_of_all_images = statistics.stdev(all_speed_values)
        grand_average = statistics.mean(all_subfolder_averages)

        # Standard deviation requires at least two data points
        if len(all_subfolder_averages) > 1:
            stdev = statistics.stdev(all_subfolder_averages)
        else:
            stdev = 0.0

        print("\n" + "#" * 60)
        print(f"SUMMARY FOR ALL SUBFOLDERS:")
        print(f"Grand Average Speed: {grand_average:.5g} km/s")
        print(f"Standard Deviation:  {stdev:.5g} km/s")
        print(f"Standard Deviation of all images:  {stdev_of_all_images:.5g} km/s")
        print("#" * 60)


if __name__ == "__main__":
    main()