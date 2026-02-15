import numpy as np
import cv2
import glob

CHECKERBOARD = (6, 9)
SQUARE_SIZE = 35.0  # Measured in millimeters

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []  # 3d point in real world space (mm)
imgpoints = []  # 2d points in image plane (pixels)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

path = input("Input path to folder with images: ")
images = glob.glob(path + "/*.jpg")

if not images:
    print("Error: No images found. Please check your file paths.")
    exit()

print(f"Found {len(images)} images. Processing...")

valid_images = 0
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        valid_images += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        print(f"✅ Pattern found in: {fname}")
    else:
        print(f"❌ Pattern NOT found in: {fname}")

if valid_images > 0:
    print("\nCalculating calibration... please wait.")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    avg_error = total_error / len(objpoints)

    print("\n" + "=" * 30)
    print("CALIBRATION RESULTS")
    print("=" * 30)
    print(f"Images used for calibration: {valid_images}")
    print(f"Average Re-projection Error: {avg_error:.4f} pixels")
    print("\nIntrinsic Matrix (K):")
    print(mtx)
    print("\nDistortion Coeffs (D):")
    print(dist)

    # Distance to board in first image to double-check the calibration quality
    distance_to_board = np.linalg.norm(tvecs[0])
    print(f"\nDistance to board in first image: {distance_to_board / 10:.2f} cm")
    print("=" * 30)

else:
    print("Could not find the checkerboard in any images. Check lighting and CHECKERBOARD dimensions.")