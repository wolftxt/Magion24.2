import math
from datetime import datetime

from exif import Image
import cv2


def get_time(image):
    with open(image, "rb") as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, "%Y:%m:%d %H:%M:%S")
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


def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures=feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = keypoints_1[image_1_idx].pt
        (x2, y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
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


# Filters out all values that aren't within 10% of the median and returns the average of the rest.
def calculate_mean_distance(coordinates_1, coordinates_2):
    dists = get_distances(coordinates_1, coordinates_2)
    dists.sort()
    median = dists[len(dists) // 2]
    new_dists = []
    for dist in dists:
        if median * 0.9 < dist < median * 1.1:
            new_dists.append(dist)
    return sum(new_dists) / len(new_dists)


def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed


def calculate(image_1, image_2):
    time_difference = get_time_difference(image_1, image_2)
    global image_1_cv
    global image_2_cv
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(
        image_1_cv, image_2_cv, 2000
    )
    matches = calculate_matches(descriptors_1, descriptors_2)

    coordinates_1, coordinates_2 = find_matching_coordinates(
        keypoints_1, keypoints_2, matches
    )
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    GSD = 10648
    speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)
    return speed
