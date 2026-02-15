import math
import numpy as np
import cv2

import camera_distortion

frame_stack = None
edge_stack = None
size = 21
current_index = 0
current_stability_mask = None
x_pixel_shift_per_second = []
y_pixel_shift_per_second = []

def initiate_stability_mask(length, img_height, img_width):
    global size, frame_stack, edge_stack, current_index, current_stability_mask, x_pixel_shift_per_second, y_pixel_shift_per_second
    size = length
    current_index = 0
    current_stability_mask = None
    x_pixel_shift_per_second = []
    y_pixel_shift_per_second = []

    frame_stack = np.empty((size, img_height, img_width), dtype=np.uint8)
    edge_stack = np.empty((size, img_height, img_width), dtype=np.float32)

def add_to_mask(image, sort):
    global frame_stack, edge_stack, current_index, current_stability_mask

    frame_stack[current_index] = image.astype(np.uint8)

    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)
    edge_mag = cv2.magnitude(grad_x, grad_y)
    edge_stack[current_index] = edge_mag

    current_index += 1

    if 1 < current_index < size and sort:
        frame_stack[:current_index].sort(axis=0)
        edge_stack[:current_index].sort(axis=0)

    if current_index == size:
        frame_stack[:current_index].sort(axis=0)
        edge_stack[:current_index].sort(axis=0)
        current_stability_mask = get_stability_mask()

def get_stability_mask():
    global current_stability_mask

    if current_stability_mask is not None:
        return current_stability_mask

    if current_index < size:
        return None

    median_idx = size // 2
    edge_variance = edge_stack[median_idx]
    median_intensity = frame_stack[median_idx]

    edge_threshold = 200
    color_threshold = 100

    mask_condition = (edge_variance < edge_threshold) & (median_intensity >= color_threshold)
    stability_mask = np.where(mask_condition, 255, 0).astype(np.uint8)

    return delete_small_dots(stability_mask)

def delete_small_dots(mask, min_area=1000):
    inverted_mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            cv2.drawContours(inverted_mask, [cnt], -1, 0, -1)
    return cv2.bitwise_not(inverted_mask)


def shift_mask(mask, shift_x, shift_y):
    rows, cols = mask.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_mask = cv2.warpAffine(mask, M, (cols, rows))
    return shifted_mask

def grid_calculate_features(image, mask, feature_number=2000, grid_size=(2, 2)):
    h, w = image.shape
    rows, cols = grid_size
    features_per_cell = feature_number // (rows * cols)

    orb = cv2.ORB_create(nfeatures=feature_number,
                         scaleFactor=1.2,
                         nlevels=8,
                         edgeThreshold=40,
                         patchSize=40,
                         fastThreshold=5
                         )
    all_kp = []
    all_des = []

    for r in range(rows):
        for c in range(cols):
            y1, y2 = (r * h // rows), ((r + 1) * h // rows)
            x1, x2 = (c * w // cols), ((c + 1) * w // cols)

            cell = image[y1:y2, x1:x2]
            mask_cell = mask[y1:y2, x1:x2]
            kp, des = orb.detectAndCompute(cell, mask_cell)

            if kp:
                # Offset keypoint coordinates back to global image space
                for k in kp:
                    k.pt = (k.pt[0] + x1, k.pt[1] + y1)
                all_kp.extend(kp)
                all_des.append(des)

                # Re-stack descriptors into a single numpy array
    import numpy as np
    descriptors = np.vstack(all_des) if all_des else None
    return all_kp, descriptors


def find_pixel_shift(keypoints_1, keypoints_2, matches):
    dxs = [keypoints_2[m.trainIdx].pt[0] - keypoints_1[m.queryIdx].pt[0] for m in matches]
    dys = [keypoints_2[m.trainIdx].pt[1] - keypoints_1[m.queryIdx].pt[1] for m in matches]
    shift_x, shift_y = np.median(dxs), np.median(dys)
    return shift_x, shift_y

def calculate_features(image_1, image_2, feature_number, time_difference):
    motion_mask = get_stability_mask()

    shift_x = np.median(x_pixel_shift_per_second) * time_difference if len(x_pixel_shift_per_second) != 0 else 0
    shift_y = np.median(y_pixel_shift_per_second) * time_difference if len(y_pixel_shift_per_second) != 0 else 0

    shifted_mask_1 = shift_mask(motion_mask, -shift_x, -shift_y)
    shifted_mask_2 = shift_mask(motion_mask, shift_x, shift_y)

    final_mask_1 = cv2.bitwise_and(motion_mask, shifted_mask_1)
    final_mask_2 = cv2.bitwise_and(motion_mask, shifted_mask_2)

    keypoints_1, descriptors_1 = grid_calculate_features(image_1, final_mask_1, feature_number)
    keypoints_2, descriptors_2 = grid_calculate_features(image_2, final_mask_2, feature_number)

    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_and_filter_matches(keypoints_1, keypoints_2, descriptors_1, descriptors_2):
    bf_coarse = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_coarse.match(descriptors_1, descriptors_2)
    filtered_matches = []
    for match in matches:
        x1, y1 = keypoints_1[match.queryIdx].pt
        x2, y2 = keypoints_2[match.trainIdx].pt
        if x2 - x1 < 2 and y2 - y1 < 2:
            # print("Something went wrong, pixel keypoint difference is 0")
            continue
        filtered_matches.append(match)
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


def calculate_mean_distance(coordinates_1, coordinates_2, h, latitude, GSD):
    if not coordinates_1:
        return 0

    count = len(coordinates_1)
    distance_angles = np.empty(count, dtype=np.float64)

    nadir = camera_distortion.get_nadir()
    r = get_earth_radius(latitude)
    for i in range(count):
        dx = nadir[0] - coordinates_1[i][0]
        dy = nadir[1] - coordinates_1[i][1]
        angle_1 = math.atan2(dy, dx)
        pixel_distance_1 = math.hypot(dx, dy)
        alpha = math.atan2(pixel_distance_1 * GSD, h)
        theta = math.asin(math.sin(alpha) * (r + h) / r) - alpha

        dx_2 = nadir[0] - coordinates_2[i][0]
        dy_2 = nadir[1] - coordinates_2[i][1]
        angle_2 = math.atan2(dy_2, dx_2)
        pixel_distance_2 = math.hypot(dx_2, dy_2)
        alpha_2 = math.atan2(pixel_distance_2 * GSD, h)
        theta_2 = math.asin(math.sin(alpha_2) * (r + h) / r) - alpha_2

        angle = math.fabs(angle_2 - angle_1)
        distance_angles[i] = math.acos(math.cos(theta) * math.cos(theta_2) + math.sin(theta) * math.sin(theta_2) * math.cos(angle))

    return np.mean(distance_angles)

def get_earth_radius(latitude):
    a = 6378137  # earth_equator_radius
    b = 6356752  # earth_polar_radius
    a_cos = a * math.cos(latitude)
    b_sin = b * math.sin(latitude)

    numerator = (a_cos * a) ** 2 + (b_sin * b) ** 2
    denominator = a_cos ** 2 + b_sin ** 2
    return math.sqrt(numerator / denominator)


def calculate_speed_in_kmps(distance_angle, time_difference, iss_altitude, latitude):
    inclination = math.radians(51.64)
    earth_radius = get_earth_radius(latitude)

    seconds_in_a_day = 86164.09
    earth_rotation_degrees = 2 * math.pi / seconds_in_a_day * math.cos(latitude)
    d_r = earth_rotation_degrees * time_difference
    # Ternary to avoid math domain error
    azimuth = 1 if math.cos(inclination) / math.cos(latitude) > 1 else math.cos(inclination) / math.cos(latitude)
    angle = math.pi / 2 + math.asin(azimuth)

    d_g = distance_angle
    d_g_and_r = math.acos(math.cos(d_g) * math.cos(d_r) + math.sin(d_g) * math.sin(d_r) * math.cos(angle))

    orbit_distance = d_g_and_r * (earth_radius + iss_altitude)
    speed_in_mps = orbit_distance / time_difference
    return speed_in_mps / 1000


def calculate(image_1, image_2, time_difference, iss_altitude, latitude):
    image_1 = camera_distortion.undistort_image(image_1)
    image_2 = camera_distortion.undistort_image(image_2)
    w, h = camera_distortion.get_dimensions()
    if image_1.shape[0] != h or image_1.shape[1] != w:
        return -1, 0
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1, image_2, 1000, time_difference)

    matches = calculate_and_filter_matches(keypoints_1, keypoints_2, descriptors_1, descriptors_2)

    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    inliers_count = np.sum(mask)
    matches_mask = mask.flatten().tolist()
    ransac_matches = [m for i, m in enumerate(matches) if matches_mask[i] == 1]

    shift_x, shift_y = find_pixel_shift(keypoints_1, keypoints_2, ransac_matches)
    x_pixel_shift_per_second.append(shift_x / time_difference)
    y_pixel_shift_per_second.append(shift_y / time_difference)

    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, ransac_matches)

    GSD = iss_altitude / camera_distortion.get_effective_f_px()
    average_arc_angle = calculate_mean_distance(coordinates_1, coordinates_2, iss_altitude, latitude, GSD)
    speed = calculate_speed_in_kmps(average_arc_angle, time_difference, iss_altitude, latitude)

    print(f"speed: {speed:.5g} km/s, inliers: {inliers_count}")

    """h1, w1 = image_1.shape
    h2, w2 = image_2.shape
    output_visual = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    output_visual[:h1, :w1] = image_1
    output_visual[:h2, w1:w1 + w2] = image_2
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