from datetime import datetime, timezone
import cv2
from exif import Image
import os
import sys
import time
import requests
import json
from glob import glob
from pathlib import Path
import statistics
import math

BASE_DIR = Path(__file__).resolve().parent
program_path = BASE_DIR / "program"
sys.path.append(str(program_path))

try:
    import calculateSpeed
except ImportError:
    sys.exit(1)

PARENT_IMAGE_DIR = BASE_DIR / "images"
CACHE_FILE = BASE_DIR / "iss_cache.json"


# --- Added: Historical API Lookup ---
def get_historical_iss_position(dt):
    """Queries the 'Where the ISS at?' API for historical position at img2 time."""
    timestamp = int(dt.timestamp())
    ts_key = str(timestamp)

    # Load existing cache
    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            cache = {}

    # Check if value is in cache
    if ts_key in cache:
        return cache[ts_key]["lat"], cache[ts_key]["alt"]

    url = f"https://api.wheretheiss.at/v1/satellites/25544/positions?timestamps={timestamp}&units=kilometers"
    try:
        # Respect API rate limit (1 request per second)
        time.sleep(1.1)
        response = requests.get(url, timeout=10)
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            pos = data[0]
            lat = float(pos['latitude'])
            alt_m = float(pos['altitude']) * 1000  # Convert km to meters

            # Save to cache
            cache[ts_key] = {"lat": lat, "alt": alt_m}
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

            return lat, alt_m
    except Exception as e:
        print(f"API Error for timestamp {timestamp}: {e}")
    return None, None


def get_time(image):
    with open(image, "rb") as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        # Treated as UTC for the API call
        time_obj = datetime.strptime(time_str, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return time_obj


def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.total_seconds()


def main():
    if not os.path.exists(PARENT_IMAGE_DIR):
        return

    subfolders = sorted([str(f) for f in PARENT_IMAGE_DIR.iterdir() if f.is_dir()])
    if not subfolders:
        subfolders = [str(PARENT_IMAGE_DIR)]

    all_subfolder_averages = []
    all_speed_values = []
    all_expected_stdevs = []  # Track errors globally

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

        half_of_image_count = min(21, len(image_files))
        for i in range(half_of_image_count):
            image = cv2.imread(image_files[i], 0)
            if i == 0:
                shape = image.shape
                calculateSpeed.initiate_stability_mask(half_of_image_count, shape[0], shape[1])
            calculateSpeed.add_to_mask(image, False)

        results = []
        folder_expected_stdevs = []  # Track errors per folder

        def process_image_pair(image_1, image_2):
            time_difference = get_time_difference(image_1, image_2)
            if time_difference < 12 or time_difference > 30:
                return
            try:
                iss_latitude, iss_altitude = get_historical_iss_position(get_time(image_1))
                if iss_latitude is not None:
                    speed, inliers = calculateSpeed.calculate(image_1, image_2, time_difference, iss_altitude, iss_latitude)
                else:
                    raise ValueError("Could not retrieve ISS data from API")

            except Exception as e:
                print(f"Error processing {os.path.basename(image_1)}: {e}")
                speed = -1
                inliers = 0

            if speed != -1:
                expected_err = speed * (math.sqrt(1 / 6) / time_difference)
                results.append({
                    "name": os.path.basename(image_1),
                    "speed": speed,
                    "confidence": inliers,
                    "expected_err": expected_err
                })
                all_speed_values.append(speed)

        for i in range(half_of_image_count, len(image_files), 1):
            if i - half_of_image_count < half_of_image_count:
                img1 = image_files[i - half_of_image_count]
                img2 = image_files[i - half_of_image_count + 1]
                process_image_pair(img1, img2)

            img1 = image_files[i - 1]
            img2 = image_files[i]
            process_image_pair(img1, img2)

        if not results:
            print("No valid matches found.")
            continue

        all_raw_speeds = [res["speed"] for res in results]
        stdev_of_all_speeds = statistics.stdev(all_raw_speeds) if len(all_raw_speeds) > 1 else 0

        # Collect expected errors for folder statistics
        for res in results:
            folder_expected_stdevs.append(res["expected_err"])
            all_expected_stdevs.append(res["expected_err"])

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
            # Calculate folder expected stdev (average of the errors)
            expected_folder_stdev = sum(folder_expected_stdevs) / len(folder_expected_stdevs)

            print("-" * 60)
            print(f"Final Filtered Average Speed for {folder_name}: {folder_avg:.5g} km/s")
            print(f"Standard deviation for all images in {folder_name}: {stdev_of_all_speeds:.5g} km/s")
            print(f"Expected standard deviation: {expected_folder_stdev:.5g} km/s")
            all_subfolder_averages.append(folder_avg)
        print("=" * 60)

    # --- Final Calculations ---
    if len(all_subfolder_averages) > 0:
        stdev_of_all_images = statistics.stdev(all_speed_values) if len(all_speed_values) > 1 else 0
        grand_average = statistics.mean(all_subfolder_averages)
        stdev = statistics.stdev(all_subfolder_averages) if len(all_subfolder_averages) > 1 else 0.0
        # Final expected stdev for all images
        grand_expected_stdev = sum(all_expected_stdevs) / len(all_expected_stdevs) if all_expected_stdevs else 0

        print("\n" + "#" * 60)
        print(f"SUMMARY FOR ALL SUBFOLDERS:")
        print(f"Grand Average Speed: {grand_average:.5g} km/s")
        print(f"Standard Deviation:  {stdev:.5g} km/s")
        print(f"Standard Deviation of all images:  {stdev_of_all_images:.5g} km/s")
        print(f"Expected standard deviation for all images: {grand_expected_stdev:.5g} km/s")
        print("#" * 60)


if __name__ == "__main__":
    main()