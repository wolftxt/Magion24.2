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


def main():
    if not os.path.exists(PARENT_IMAGE_DIR):
        return

    subfolders = sorted([f.path for f in os.scandir(PARENT_IMAGE_DIR) if f.is_dir()])

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
            speed, inliers = calculate(img1, img2)
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

        final_speeds = []
        for res in best_results:
            print(f"{res['name']:<30} | {res['speed']:.4f} km/s   | {res['confidence']}")
            final_speeds.append(res["speed"])

        if final_speeds:
            avg_speed = sum(final_speeds) / len(final_speeds)
            print("-" * 60)
            print(f"Final Filtered Average Speed for {folder_name}: {avg_speed:.4f} km/s")
        else:
            print("No valid matches found.")
        print("=" * 60)


if __name__ == "__main__":
    main()