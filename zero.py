from sense_hat import SenseHat
from time import sleep
from time import time
from math import sqrt
import random

start_time = time()

sense = SenseHat()
sense.set_rotation(270, False)
sense.set_imu_config(False, False, True)

sense.color.gain = 60 
sense.color.integration_cycles = 64 

sleep_time = 0.5
scroll_speed = 0.01

d = (0, 0, 139)   # dark blue
a = (0, 191, 255) # sky blue
c = (0, 255, 255) # cyan
w = (255, 255, 255) # white
y = (255, 255, 0)   # yellow
o = (255, 165, 0)   # orange
r = (255, 0, 0)     # red
l = (127, 0, 255)   # purple
t = (0, 100, 0)     # dark green
r = (255, 0, 0)     # red
b = (0, 0, 0)       # black
s = (255, 255, 255) # will change during the program
falling1 = [
    b, b, b, l, s, s, l, b,
    b, b, b, b, l, l, b, b,
    b, b, b, b, b, t, t, b,
    b, b, t, t, b, t, b, b,
    b, b, b, t, t, b, b, b,
    b, b, b, b, t, t, b, b,
    b, b, b, b, t, t, b, b,
    s, s, s, s, s, s, s, s
]
falling2 = [
    b, b, b, b, b, b, b, b,
    b, b, b, b, b, b, l, s,
    b, b, b, b, b, b, l, l,
    b, b, b, t, b, t, t, b,
    b, b, b, b, t, t, b, b,
    b, b, b, b, t, b, b, b,
    b, b, b, b, t, b, b, b,
    s, s, s, s, s, s, s, s
]
falling3 = [
    b, b, b, b, b, b, b, b,
    b, b, b, b, b, b, b, b,
    b, b, b, b, b, b, b, b,
    b, b, b, b, b, b, b, b,
    b, b, b, b, b, b, b, b,
    b, b, b, b, t, t, t, t,
    b, b, b, t, t, t, t, t,
    s, s, s, s, s, s, s, s
]
fire = [
    b, r, w, w, w, w, r, b,
    b, b, o, o, o, o, b, b,
    b, b, b, y, y, b, y, y,
    y, y, b, y, y, y, b, b,
    b, b, o, o, o, b, b, b,
    b, b, b, r, r, b, b, b,
    b, b, b, r, r, b, b, b,
    w, w, w, w, w, w, w, w
]
ice = [
    b, a, w, w, w, w, a, b,
    b, b, c, c, c, c, b, b,
    b, b, b, d, d, b, d, d,
    d, d, b, d, d, d, b, b,
    b, b, c, c, c, b, b, b,
    b, b, b, a, a, b, b, b,
    b, b, b, a, a, b, b, b,
    w, w, w, w, w, w, w, w
]


shake_count = 0
mode = True
orientation = 1


while True:
    shakeXYZ = sense.get_accelerometer_raw()
    x = shakeXYZ["x"]
    y = shakeXYZ["y"]
    z = shakeXYZ["z"]
    shake_vector = sqrt(x**2 + y**2 + z**2)    
    shake_count = shake_count + 1 if shake_vector > 3 else 0
    
    rgb = sense.color
    average = (rgb.red + rgb.green + rgb.blue)/3
    brightness = 4 - (average * 8 / 255)
    brightness = 0 if brightness < 0 else brightness
    s = (rgb.red, rgb.green, rgb.blue)
  

    mode = not mode
    if mode:
        image = [
            b, l, s, s, s, s, l, b,
            b, b, l, l, l, l, b, b,
            b, b, b, t, t, b, t, t,
            t, t, b, t, t, t, b, b,
            b, b, t, t, t, b, b, b,
            b, b, b, t, t, b, b, b,
            b, b, b, t, t, b, b, b,
            s, s, s, s, s, s, s, s
        ]
    else:
        image = [
            b, l, s, s, s, s, l, b,
            b, b, l, l, l, l, b, b,
            t, t, b, t, t, b, b, b,
            b, b, t, t, t, b, t, t,
            b, b, b, t, t, t, b, b,
            b, b, b, t, t, b, b, b,
            b, b, b, t, t, b, b, b,
            s, s, s, s, s, s, s, s
        ]
    for j in range(int(brightness)):
        image[j * 8:j * 8 + 8] = 8 * [b]
    if shake_count >= 3:
        shake_count = 0
        image = falling1
        sense.set_pixels(image[::orientation])
        sleep(sleep_time)
        image = falling2
        sense.set_pixels(image[::orientation])
        sleep(sleep_time)
        image = falling3
        sense.set_pixels(image[::orientation])
        sleep(sleep_time)
        sense.show_message("You died", text_colour=b, back_colour=r, scroll_speed=scroll_speed)

    if sense.get_temperature() > 30:
        image = fire
        sense.set_pixels(image[::orientation])
        sleep(sleep_time)
        sense.show_message("You died", text_colour=b, back_colour=r, scroll_speed=scroll_speed)

    if sense.get_temperature() < 20:
        image = ice
        sense.set_pixels(image[::orientation])
        sleep(sleep_time)
        sense.show_message("You died", text_colour=b, back_colour=a, scroll_speed=scroll_speed)
    
    if sense.get_humidity() > 80:
        for i in range(5):
            image[random.randint(0, 63)] = a
    sense.set_pixels(image[::orientation])
    sleep(sleep_time)
    if time() - start_time > 25:
        break
