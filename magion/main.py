from picamzero import Camera
import time

from writeResult import write_result
from calculateSpeed import calculate

TIME_INTERVAL = 13
IMAGE_COUNT = 42

cam = Camera()

start_time = time.time()
images = [0] * IMAGE_COUNT

speed = 0
out_of_time = False

for i in range(IMAGE_COUNT):
    if time.time() - start_time > 570:
        speed /= i
        out_of_time = True
        break
    cycle_time = time.time()
    images[i] = cam.take_photo("image" + str(i) + ".jpg")
    if i > 0:
        speed += calculate(images[i - 1], images[i])
    elapsed_time = time.time() - cycle_time
    sleep_time = max(0, TIME_INTERVAL - elapsed_time)
    time.sleep(sleep_time)

if not out_of_time:
    speed /= IMAGE_COUNT - 1

del cam
write_result(speed)
