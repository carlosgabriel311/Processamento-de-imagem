import cv2
import numpy as np

def pos_processamento_mascara(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  
    # Convert the grayscale image to 8-bit unsigned format
    gray = cv2.convertScaleAbs(gray)
    img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel_size = (5, 5)

    # Crie o kernel para a operação de abertura
    kernel = np.ones(kernel_size, np.uint8)

    # Realize a abertura (dilatação seguida de erosão)
    opened_image = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

    return opened_image
