import math
import numpy as np

import cv2

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

    if all(pixel < 40 for pixel in corners):
        # 1. Create a binary version of the image:
        # Anything not black becomes white (255)
        _, thresh = cv2.threshold(image_1, 80, 255, cv2.THRESH_BINARY)

        # 2. Find the bounding box of all white pixels
        # This locates the 'porthole' regardless of what's inside it
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # 3. Calculate center and radius based on the bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            # Use the smaller dimension to ensure the circle stays inside the box
            radius = int(min(w, h) / 2 * 0.9)

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

def calculate_speed_in_kmps(feature_distance, gsd, time_difference, iss_altitude, latitude):
    inclination = math.radians(51.6)
    lat_rad = math.radians(latitude)
    d_r = 463.8 * math.cos(lat_rad)
    cos_beta = math.cos(inclination) / math.cos(lat_rad)
    cos_beta = min(1.0, max(-1.0, cos_beta))

    d_g = (feature_distance * gsd)
    d_g_and_r = math.sqrt(d_g ** 2 + d_r ** 2 + 2 * d_g * d_r * cos_beta)

    earth_radius = 6371000

    angle = 2 * math.asin(d_g_and_r/earth_radius/2)

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

    print(f"speed: {speed:.4f} km/h, inliers: {inliers_count}")

    """h1, w1 = img1_cv.shape
    h2, w2 = img2_cv.shape
    output_visual = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    output_visual[:h1, :w1] = img1_cv
    output_visual[:h2, w1:w1 + w2] = img2_cv
    output_visual = cv2.cvtColor(output_visual, cv2.COLOR_GRAY2BGR)  # Convert to color to draw colored lines

    # 2. Manually draw thick lines for the top 50 matches
    for m in ransac_matches:
        # Get the coordinates
        pt1 = (int(keypoints_1[m.queryIdx].pt[0]), int(keypoints_1[m.queryIdx].pt[1]))
        pt2 = (int(keypoints_2[m.trainIdx].pt[0] + w1), int(keypoints_2[m.trainIdx].pt[1]))

        # Draw the line with thickness=3
        cv2.line(output_visual, pt1, pt2, (0, 255, 0), 3)
        # Draw a circle at the joints
        cv2.circle(output_visual, pt1, 5, (0, 0, 255), -1)
        cv2.circle(output_visual, pt2, 5, (0, 0, 255), -1)
    # if speed < 4 or speed > 9:
    cv2.imshow("Feature Matches", output_visual)
    cv2.waitKey(0)  # This keeps the window open until you press a key
    cv2.destroyAllWindows()"""
    return speed, inliers_count