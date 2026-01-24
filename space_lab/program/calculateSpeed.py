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

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv


def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures=feature_number)
    height, width = image_1.shape

    mask = None

    # Check corners to see if we need a mask (Threshold 15 for noise)
    corners = [image_1[0, 0], image_1[0, width - 1],
               image_1[height - 1, 0], image_1[height - 1, width - 1]]

    if all(pixel < 15 for pixel in corners):
        # 1. Create a binary version of the image:
        # Anything not black becomes white (255)
        _, thresh = cv2.threshold(image_1, 15, 255, cv2.THRESH_BINARY)

        # 2. Find the bounding box of all white pixels
        # This locates the 'porthole' regardless of what's inside it
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # 3. Calculate center and radius based on the bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            # Use the smaller dimension to ensure the circle stays inside the box
            radius = int(min(w, h) / 2 * 0.95)

            # 4. Final check: Only create mask if radius is valid
            if radius > 10:
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # 5. EXECUTION: If mask is invalid/too small, orb will just use the whole image
    try:
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, mask)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, mask)
    except Exception as e:
        print(f"Mask error, falling back to full image: {e}")
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)

    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    good_matches = []
    for match in raw_matches:
        # Check if we actually found two matches for this descriptor
        if len(match) == 2:
            m, n = match
            if m.distance < 0.80 * n.distance:
                good_matches.append(m)

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
    ground_distance_m = (feature_distance * GSD) / 100

    EARTH_RADIUS = 6371000
    ISS_ALTITUDE = 420000

    scale_factor = (EARTH_RADIUS + ISS_ALTITUDE) / EARTH_RADIUS

    orbital_distance_m = ground_distance_m * scale_factor

    speed_kmps = (orbital_distance_m / time_difference) / 1000

    return speed_kmps


def calculate(image_1, image_2, time_difference):
    img1_cv, img2_cv = convert_to_cv(image_1, image_2)

    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(img1_cv, img2_cv, 2000)

    matches = calculate_matches(descriptors_1, descriptors_2)

    # Apply RANSAC
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # M is the matrix, mask identifies inliers
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inliers_count = np.sum(mask)
    matches_mask = mask.flatten().tolist()
    ransac_matches = [m for i, m in enumerate(matches) if matches_mask[i] == 1]

    # Calculate speed using only RANSAC inliers
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, ransac_matches)
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

    GSD = 13020
    speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)

    return speed, inliers_count