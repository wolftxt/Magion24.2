from picamzero import Camera
from astro_pi_orbit import ISS
import time


from writeResult import write_result
from calculateSpeed import calculate

# Configuration
TIME_INTERVAL = 13
IMAGE_COUNT = 42
MAX_TIME_ELAPSED = 570  # 9.5 minutes

start_time = time.time()


def capture_images():
    cam = Camera()
    results = []
    images = []
    timestamps = []

    for i in range(IMAGE_COUNT):
        if time.time() - start_time > MAX_TIME_ELAPSED:
            print("Time limit reached. Breaking loop.")
            break

        cycle_start = time.time()

        # 1. Capture timing
        image_path = f"image{i}.jpg"
        capture_start = time.time()
        cam.take_photo(image_path)
        capture_end = time.time()

        images.append(image_path)
        timestamps.append(capture_end)

        # 2. Calculation timing
        if i > 0:
            img1 = images[i - 1]
            img2 = images[i]
            time_diff = timestamps[i] - timestamps[i - 1]

            calc_start = time.time()
            try:
                height = ISS().coordinates().elevation.cm
                speed, inliers = calculate(img1, img2, time_diff, height)
                calc_end = time.time()

                results.append({
                    "speed": speed,
                    "confidence": inliers
                })
            except Exception as e:
                print(f"Calculation error at index {i}: {e}")

        # 3. Sleep timing
        elapsed_in_cycle = time.time() - cycle_start
        sleep_time = max(0, TIME_INTERVAL - elapsed_in_cycle)

        if elapsed_in_cycle > TIME_INTERVAL:
            print(f"WARNING: Cycle {i} exceeded TIME_INTERVAL! Timing may be drifting.")

        time.sleep(sleep_time)

    # --- Filtering Logic ---
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