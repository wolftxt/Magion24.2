import cv2
import numpy as np

focal_length_mm = 5.0
sensor_width_mm = 6.287
w_px, h_px = 4056, 3040
f_px = (focal_length_mm * w_px) / sensor_width_mm

K = np.array([[f_px, 0, w_px / 2],
                  [0, f_px, h_px / 2],
                  [0, 0, 1]], dtype=np.float64)

D = np.array([[-0.50159923, 0.31835896, 0.002986, 0.00156591, -0.13630313]], dtype=np.float64)

new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_px, h_px), 0, (w_px, h_px))
x, y, w_roi, h_roi = roi

def undistort_image(img_path):
    image = cv2.imread(img_path, 0)

    undistorted_img = cv2.undistort(image, K, D, None, new_camera_mtx)
    undistorted_img = undistorted_img[y:y + h_roi, x:x + w_roi]

    return undistorted_img

def get_nadir():
    return new_camera_mtx[0, 2] - x, new_camera_mtx[1, 2] - y

def get_dimensions():
    return w_roi, h_roi

def get_effective_f_px():
    return float(new_camera_mtx[0, 0])