#>3mb, 10min, main.py, vygenerovat result.txt (5 cifer), 250mb, 42 photo
import math
from datetime import datetime
from exif import Image
import cv2

import matplotlib.pyplot as plt

# jen pridavam to co jsem v kodu pouzivala predtim jak napad :D

from time import sleep

#vubec nevim jak importovat knihovnu pro kameru !!!!!!!!!
from picamzero import Camera
#from picamera import PiCamera
#import numpy as np
#import os

image_1 = r"C:\Users\TEST\Documents\GitHub\Magion24.2\image1.jpg"
image_2 = r"C:\Users\TEST\Documents\GitHub\Magion24.2\image2.jpg"

'''
#jeste je potreba prejmenovat slozku ale idk jak se ma jmenovat
#vytvoreni slozky pro ukladani fotek
folder_name = 'photos'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


# Inicializace kamery
# v teto casti je i nastaveni infrared snimani ale nevim jak a kde se to prrepina :D
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 1
camera.sensor_mode = 3
camera.exposure_mode = 'off'
camera.awb_mode = 'off'
camera.awb_gains = (1.5, 1.5)

# Hlavní smyčka programu
# predelat snimkovani tak aby to casove vychazelo !!!!!!!!!!!!!!!!!!!!!!!
for i in range(3 * 60 * 60):
    # Získání snímku z kamery a převod do formátu numpy array
    #idk jestli je to spravny format !!!!!!!!!!

    image = np.empty((480, 640, 3), dtype=np.uint8)
    camera.capture(image, 'rgb')

#nevim jestli toto chceme ale hodim to sem taky :D
    # Detekce noci a oblačnosti pomocí jednoduchých pravidel
    is_night = np.mean(image) < 50
    is_cloudy = np.mean(image[:, :, 1]) < 50

    # Uložení snímku, pokud nebylo detekováno noční nebo zatažené počasí
    if not is_night and not is_cloudy:
        filename = f'{folder_name}/ndvi_{i:06}.jpg'
        ndvi_image = (ndvi_image * 255).astype(np.uint8)
        camera.annotate_text = filename
        camera.capture(filename, format='jpeg', quality=90, thumbnail=None)  
'''

# Pauza mezi snímky
sleep(1)

# Ukončení kamery
#camera.close()

def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time


def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds


def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv


def calculate_features(image_1_cv, image_2_cv, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')


def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2


def get_distances(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    distances = []
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        distances.append(distance)
    return distances


def calculate_mean_distance(coordinates_1, coordinates_2):
    dists = get_distances(coordinates_1, coordinates_2)
    return sum(dists) / len(dists)


def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed


time_difference = get_time_difference(image_1, image_2) # Get time difference between images
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # Get keypoints and descriptors
matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors

display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)

coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

GSD = 12648
GSD = 10648
speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)

print(speed)
plt.hist(get_distances(coordinates_1, coordinates_2), bins=64)
plt.show()


# jeste dokoncit vypocet rychlosti pomoci to norm metody 
# vypocet pomoci newtonova zakona
# pridat celkovy prumer rychlosti
# ukladani do vytvoreneho noveho souboru (jak se ma jmenovat, protoze main je soubor kodu)
# ukladani fotografii do slozky (taky nevim jak se ma jmenovat a zda je to libovolne)
# 
#    www
#  \(o_o)/ 
#   |___|       Lenča
#  /_____\
#    | |

#    www
#  \(o_o)/
#   |___|        Davča
#   |___|
#    | |

#    www
#  \(o_o)/
#   |___|        Peťka
#  /_____\
#    | |