from picamzero import Camera
import time

from writeResult import write_result
from space_lab.program.calculateSpeed import calculate

TIME_INTERVAL = 13
IMAGE_COUNT = 42
MAX_TIME_ELAPSED = 570  # 30 Seconds before the 10 minute mark to ensure the time limit.

start_time = time.time()


# Captures a picture every 13 seconds. In between, it calculates speed based on the captured images.
def capture_images():
    cam = Camera()

    speed = 0
    images = [0] * IMAGE_COUNT

    for i in range(IMAGE_COUNT):
        # Makes sure that the program will end before the 10 minute limit.
        if time.time() - start_time > MAX_TIME_ELAPSED:
            speed /= i
            return speed
        cycle_time = time.time()
        images[i] = cam.take_photo("image" + str(i) + ".jpg")
        if i > 0:
            speed += calculate(images[i - 1], images[i])
        elapsed_time = time.time() - cycle_time
        sleep_time = max(0, TIME_INTERVAL - elapsed_time)
        time.sleep(sleep_time)

    speed /= IMAGE_COUNT - 1
    del cam
    return speed


def main():
    speed = capture_images()
    write_result(speed)


if __name__ == "__main__":
    main()

