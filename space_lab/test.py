import os
import sys
from glob import glob
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
program_path = BASE_DIR / "program"
sys.path.append(str(program_path))

try:
    from calculateSpeed import calculate, get_time
except ImportError:
    sys.exit(1)

PARENT_IMAGE_DIR = BASE_DIR / "images"


def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds


def main():
    if not os.path.exists(PARENT_IMAGE_DIR):
        return
    highlights_path = PARENT_IMAGE_DIR / "arthur3"
    subfolders = sorted([str(highlights_path)])

    # --- New: List to store the average speed of each subfolder ---
    all_subfolder_averages = []

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
                speed, inliers = calculate(img1, img2, time_difference, 400000)
            except:
                speed = -1
                inliers = 0

            if speed != -1:
                results.append({
                    "name": os.path.basename(img2),
                    "speed": speed,
                    "confidence": inliers
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        top_n = max(1, len(results) // 4)
        best_results = results[:top_n]

        print(f"{'Image Name':<30} | {'Speed (km/s)':<15} | {'Inliers'}")
        print("-" * 60)

        folder_speeds = []
        for res in best_results:
            print(f"{res['name']:<30} | {res['speed']:.4f} km/s   | {res['confidence']}")
            folder_speeds.append(res["speed"])

        if folder_speeds:
            folder_avg = sum(folder_speeds) / len(folder_speeds)
            print("-" * 60)
            print(f"Final Filtered Average Speed for {folder_name}: {folder_avg:.4f} km/s")

            # --- New: Add this folder's average to our grand total list ---
            all_subfolder_averages.append(folder_avg)
        else:
            print("No valid matches found.")
        print("=" * 60)

    # --- New: Calculate and display the final grand average ---
    if all_subfolder_averages:
        grand_average = sum(all_subfolder_averages) / len(all_subfolder_averages)
        print("\n" + "#" * 60)
        print(f"GRAND TOTAL AVERAGE SPEED ACROSS ALL FOLDERS: {grand_average:.4f} km/s")
        print("#" * 60)


if __name__ == "__main__":
    main()