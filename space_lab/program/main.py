import math

import cv2
from picamzero import Camera
from astro_pi_orbit import ISS
import time

from writeResult import write_result
import calculateSpeed

TIME_INTERVAL = 13
IMAGE_COUNT = 42
MAX_TIME_ELAPSED = 570  # 9.5 minutes

start_time = time.perf_counter()

def capture_images():
    cam = Camera()
    results = []
    images = []
    timestamps = []

    def process_image_pair(img1, img2, i):
        time_diff = timestamps[i] - timestamps[i - 1]

        try:
            iss_altitude = ISS().coordinates().elevation.m
            iss_latitude = ISS().coordinates().latitude.degrees
            speed, inliers = calculateSpeed.calculate(img1, img2, time_diff, iss_altitude, math.radians(iss_latitude))

            results.append({
                "speed": speed,
                "confidence": inliers
            })
        except Exception as e:
            print(f"Calculation error at index {i}: {e}")

    half_of_image_count = IMAGE_COUNT // 2
    calculateSpeed.initiate_stability_mask(half_of_image_count, 3040, 4056)

    for i in range(half_of_image_count):
        if time.perf_counter() - start_time > MAX_TIME_ELAPSED:
            print("Time limit reached. Breaking loop.")
            break
        cycle_start = time.perf_counter()

        image_path = f"image{i}.jpg"
        cam.take_photo(image_path)
        capture_end = time.perf_counter()

        images.append(image_path)
        timestamps.append(capture_end)

        calculateSpeed.add_to_mask(cv2.imread(images[i], 0), True)

        elapsed_in_cycle = time.perf_counter() - cycle_start
        sleep_time = max(0, TIME_INTERVAL - elapsed_in_cycle)

        if elapsed_in_cycle > TIME_INTERVAL:
            print(f"WARNING: Cycle {i} exceeded TIME_INTERVAL! Timing may be drifting.")

        time.sleep(sleep_time)

    for i in range(half_of_image_count, IMAGE_COUNT, 1):
        if time.perf_counter() - start_time > MAX_TIME_ELAPSED:
            print("Time limit reached. Breaking loop.")
            break

        cycle_start = time.perf_counter()

        if i - half_of_image_count < half_of_image_count:
            img1 = images[i - half_of_image_count]
            img2 = images[i - half_of_image_count + 1]
            process_image_pair(img1, img2, i - half_of_image_count + 1)

        image_path = f"image{i}.jpg"
        cam.take_photo(image_path)
        capture_end = time.perf_counter()

        images.append(image_path)
        timestamps.append(capture_end)

        img1 = images[i - 1]
        img2 = images[i]
        process_image_pair(img1, img2, i)

        elapsed_in_cycle = time.perf_counter() - cycle_start
        sleep_time = max(0, TIME_INTERVAL - elapsed_in_cycle)

        if elapsed_in_cycle > TIME_INTERVAL:
            print(f"WARNING: Cycle {i} exceeded TIME_INTERVAL! Timing may be drifting.")

        time.sleep(sleep_time)

    if not results:
        print("No results collected.")
        return 0

    results.sort(key=lambda x: x["confidence"], reverse=True)
    top_n = max(1, len(results) // 4)
    best_results = results[:top_n]

    final_speeds = [res["speed"] for res in best_results]
    avg_speed = sum(final_speeds) / len(final_speeds)

    print(f"Final Average Speed: {avg_speed:.4f}")

    del cam
    return avg_speed


def main():
    try:
        speed = capture_images()
        write_result(speed)
    except Exception as e:
        print(f"Fatal error: {e}")
        write_result(0)


if __name__ == "__main__":
    main()