from picamzero import Camera
import time
import os

from writeResult import write_result
from calculateSpeed import calculate

# Configuration
TIME_INTERVAL = 2
IMAGE_COUNT = 42
MAX_TIME_ELAPSED = 570  # 9.5 minutes

start_time = time.time()


def capture_images():
    cam = Camera()
    results = []
    images = []
    timestamps = []

    for i in range(IMAGE_COUNT):
        # 1. Check for time limit
        if time.time() - start_time > MAX_TIME_ELAPSED:
            break

        cycle_start = time.time()

        # 2. Capture image and record precise timestamp
        image_path = f"image{i}.jpg"
        cam.take_photo(image_path)
        images.append(image_path)
        timestamps.append(time.time())

        # 3. If we have at least two images, calculate speed for the interval
        if i > 0:
            img1 = images[i - 1]
            img2 = images[i]
            # Time difference between the two actual capture moments
            time_diff = timestamps[i] - timestamps[i - 1]

            try:
                # Expecting: speed, inliers = calculate(path1, path2, seconds)
                speed, inliers = calculate(img1, img2, time_diff)

                if speed != -1:
                    results.append({
                        "speed": speed,
                        "confidence": inliers
                    })
            except Exception as e:
                print(f"Calculation error at index {i}: {e}")

        # 4. Precise sleep to maintain interval
        elapsed_in_cycle = time.time() - cycle_start
        sleep_time = max(0, TIME_INTERVAL - elapsed_in_cycle)
        time.sleep(sleep_time)

    # --- Filtering Logic (The "Sorting System") ---
    if not results:
        return 0

    # Sort by confidence (inliers) descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Take the top 25% (minimum of 1)
    top_n = max(1, len(results) // 4)
    best_results = results[:top_n]

    # Calculate final average speed
    final_speeds = [res["speed"] for res in best_results]
    avg_speed = sum(final_speeds) / len(final_speeds)

    del cam
    return avg_speed


def main():
    try:
        speed = capture_images()
        write_result(speed)
    except Exception as e:
        print(f"Fatal error: {e}")
        # Ensure we write something so the mission doesn't fail
        write_result(0)


if __name__ == "__main__":
    main()