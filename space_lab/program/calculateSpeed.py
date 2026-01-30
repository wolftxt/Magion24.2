import math
import os
import numpy as np
import cv2


def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

def shift_mask(mask, shift_x, shift_y):
    rows, cols = mask.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_mask = cv2.warpAffine(mask, M, (cols, rows))
    return shifted_mask

def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures=feature_number)
    height, width = image_1.shape

    diff = cv2.absdiff(image_1, image_2)
    _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

    # First coarse search
    kp_c1, des_c1 = orb.detectAndCompute(image_1, motion_mask)
    kp_c2, des_c2 = orb.detectAndCompute(image_2, motion_mask)

    shift_x, shift_y = 0, 0
    if des_c1 is not None and des_c2 is not None:
        bf_coarse = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        coarse_matches = bf_coarse.match(des_c1, des_c2)
        if len(coarse_matches) > 5:
            dxs = [kp_c2[m.trainIdx].pt[0] - kp_c1[m.queryIdx].pt[0] for m in coarse_matches]
            dys = [kp_c2[m.trainIdx].pt[1] - kp_c1[m.queryIdx].pt[1] for m in coarse_matches]
            shift_x, shift_y = np.median(dxs), np.median(dys)

    shifted_mask_1 = shift_mask(motion_mask, -shift_x, -shift_y)
    shifted_mask_2 = shift_mask(motion_mask, shift_x, shift_y)

    final_mask_1 = cv2.bitwise_and(motion_mask, shifted_mask_1)
    final_mask_2 = cv2.bitwise_and(motion_mask, shifted_mask_2)

    try:
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, final_mask_1)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, final_mask_2)
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


def calculate_mean_distance(coordinates_1, coordinates_2):
    if not coordinates_1:
        return 0

    sum_dx = 0
    sum_dy = 0
    count = len(coordinates_1)

    for i in range(count):
        dx = coordinates_2[i][0] - coordinates_1[i][0]
        dy = coordinates_2[i][1] - coordinates_1[i][1]
        if math.fabs(dx) < 2 and math.fabs(dy) < 2:
            print("Something went VERY WRONG, the distances are 0.")
            count -= 1
            continue
        sum_dx += dx
        sum_dy += dy

    mean_dx = sum_dx / count
    mean_dy = sum_dy / count

    return math.hypot(mean_dx, mean_dy)

def get_earth_radius(latitude):
    a = 6378137  # earth_equator_radius
    b = 6356752  # earth_polar_radius
    a_cos = a * math.cos(latitude)
    b_sin = b * math.sin(latitude)

    numerator = (a_cos * a) ** 2 + (b_sin * b) ** 2
    denominator = a_cos ** 2 + b_sin ** 2
    return math.sqrt(numerator / denominator)


def calculate_speed_in_kmps(feature_distance, gsd, time_difference, iss_altitude, latitude):
    inclination = math.radians(51.64)
    lat_rad = math.radians(latitude)
    d_r = 463.8 * math.cos(lat_rad) * time_difference
    cos_beta = math.cos(inclination) / math.cos(lat_rad)
    cos_beta = min(1.0, max(-1.0, cos_beta))

    d_g = (feature_distance * gsd)
    d_g_and_r = math.sqrt(d_g ** 2 + d_r ** 2 + 2 * d_g * d_r * cos_beta)

    earth_radius = get_earth_radius(latitude)

    # Small inefficiency in the assumption that earth is a perfect sphere
    angle = 2 * math.asin(d_g_and_r / earth_radius / 2)

    arc_distance = angle * (earth_radius + iss_altitude)
    speed_in_mps = arc_distance / time_difference

    return speed_in_mps / 1000


def calculate(image_1, image_2, time_difference, iss_altitude, latitude):
    img1_cv, img2_cv = convert_to_cv(image_1, image_2)

    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(img1_cv, img2_cv, 1000)

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

    image_width_px = 4056
    focal_length_mm = 5.0
    sensor_width_mm = 6.287
    GSD = (iss_altitude * sensor_width_mm) / (focal_length_mm * image_width_px)
    speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference, iss_altitude, latitude)

    print(f"speed: {speed:.5g} km/h, inliers: {inliers_count}, image name: {os.path.basename(image_2)}")

    """h1, w1 = img1_cv.shape
    h2, w2 = img2_cv.shape
    output_visual = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    output_visual[:h1, :w1] = img1_cv
    output_visual[:h2, w1:w1 + w2] = img2_cv
    output_visual = cv2.cvtColor(output_visual, cv2.COLOR_GRAY2BGR)

    for m in ransac_matches:
        pt1 = (int(keypoints_1[m.queryIdx].pt[0]), int(keypoints_1[m.queryIdx].pt[1]))
        pt2 = (int(keypoints_2[m.trainIdx].pt[0] + w1), int(keypoints_2[m.trainIdx].pt[1]))
        cv2.line(output_visual, pt1, pt2, (0, 255, 0), 3)
        cv2.circle(output_visual, pt1, 5, (0, 0, 255), -1)
        cv2.circle(output_visual, pt2, 5, (0, 0, 255), -1)

    cv2.imshow("Feature Matches", output_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    return speed, inliers_count