from time import sleep
from picamera import PiCamera
import numpy as np
import os

# Nastavení názvu složky pro ukládání fotografií
folder_name = 'photos'

# Vytvoření složky pro ukládání fotografií, pokud ještě neexistuje
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Inicializace kamery a nastavení režimu infračerveného snímání
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 1
camera.sensor_mode = 3
camera.exposure_mode = 'off'
camera.awb_mode = 'off'
camera.awb_gains = (1.5, 1.5)

# Hlavní smyčka programu
for i in range(3 * 60 * 60):
    # Získání snímku z kamery a převod do formátu numpy array
    image = np.empty((480, 640, 3), dtype=np.uint8)
    camera.capture(image, 'rgb')
    
    # Převod RGB snímku na NDVI snímek a normalizace na rozsah 0 až 1
    ndvi_image = np.zeros_like(image[:, :, 0], dtype=float)
    valid_pixels = np.logical_and(image[:, :, 1] > 0, image[:, :, 0] > 0)
    ndvi_image[valid_pixels] = (image[:, :, 1][valid_pixels].astype(float) - image[:, :, 0][valid_pixels]) / \
                           (image[:, :, 1][valid_pixels].astype(float) + image[:, :, 0][valid_pixels])

    
    # Detekce noci a oblačnosti pomocí jednoduchých pravidel
    is_night = np.mean(image) < 50
    is_cloudy = np.mean(image[:, :, 1]) < 50
    
    # Uložení snímku, pokud nebylo detekováno noční nebo zatažené počasí
    if not is_night and not is_cloudy:
        filename = f'{folder_name}/ndvi_{i:06}.jpg'
        ndvi_image = (ndvi_image * 255).astype(np.uint8)
        camera.annotate_text = filename
        camera.capture(filename, format='jpeg', quality=90, thumbnail=None)
        
    # Pauza mezi snímky
    sleep(1)
    
# Ukončení kamery
camera.close()
