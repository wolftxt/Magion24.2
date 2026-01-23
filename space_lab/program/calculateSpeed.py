import math
import numpy as np
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

    # Create a black mask the same size as your image
    mask = np.zeros(image_1.shape, dtype=np.uint8)

    # Draw a white circle in the middle (where the ground is)
    # We use 40% of the height as the radius to stay away from the window edges
    height, width = image_1.shape
    cv2.circle(mask, (width // 2, height // 2), int(height * 0.4), 255, -1)

    # Tell ORB to ONLY look inside that white circle
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, mask)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, mask)

    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
    # Use Brute Force but turn off crossCheck to allow knnMatch
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Find the 2 best matches for every point
    raw_matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    good_matches = []
    for m, n in raw_matches:
        # Lowe's Ratio Test: is the best match much better than the second best?
        if m.distance < 0.80 * n.distance:
            good_matches.append(m)

    # Sort them so the very best ones are at the top
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    return good_matches


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


def calculate_mean_distance(coordinates_1, coordinates_2):
    dists = get_distances(coordinates_1, coordinates_2)
    return sum(dists) / len(dists)


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
    if len(coordinates_1) == 0:
        return -1
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    GSD = 13020
    speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)

    h1, w1 = image_1_cv.shape
    h2, w2 = image_2_cv.shape
    output_visual = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    output_visual[:h1, :w1] = image_1_cv
    output_visual[:h2, w1:w1 + w2] = image_2_cv
    output_visual = cv2.cvtColor(output_visual, cv2.COLOR_GRAY2BGR)  # Convert to color to draw colored lines

    # 2. Manually draw thick lines for the top 50 matches
    for m in matches[:100]:
        # Get the coordinates
        pt1 = (int(keypoints_1[m.queryIdx].pt[0]), int(keypoints_1[m.queryIdx].pt[1]))
        pt2 = (int(keypoints_2[m.trainIdx].pt[0] + w1), int(keypoints_2[m.trainIdx].pt[1]))

        # Draw the line with thickness=3
        cv2.line(output_visual, pt1, pt2, (0, 255, 0), 3)
        # Draw a circle at the joints
        cv2.circle(output_visual, pt1, 5, (0, 0, 255), -1)
        cv2.circle(output_visual, pt2, 5, (0, 0, 255), -1)
    #if speed < 4 or speed > 9:
    #cv2.imshow("Feature Matches", output_visual)
    #cv2.waitKey(0)  # This keeps the window open until you press a key
    #cv2.destroyAllWindows()
    return speed
