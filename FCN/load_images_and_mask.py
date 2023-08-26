import os

import cv2
import numpy as np


def load_images_and_masks(images_path, masks_path, input_size):
    image_list = []
    mask_list = []

    for filename in os.listdir(images_path):
        img_path = os.path.join(images_path, filename)
        if img_path.endswith(('.jpg', '.jpeg', '.png')):
            # Carregue a imagem no formato BGR usando o OpenCV
            img = cv2.imread(img_path)
            # Verifique se a imagem foi carregada corretamente
            if img is not None:
                # Redimensione a imagem para o tamanho desejado
                img = cv2.resize(img, input_size)

                # Construa o caminho para a máscara correspondente
                mask_filename = filename.split('.')[0] + '.png'
                mask_path = os.path.join(masks_path, mask_filename)

                # Carregue a máscara no formato grayscale
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Verifique se a máscara foi carregada corretamente
                if mask is not None:
                    # Redimensione a máscara para o tamanho desejado
                    mask = cv2.resize(mask, input_size)

                    # Normalização das imagens e máscaras (0-1)
                    img = img / 255.0
                    mask = mask / 255.0

                    image_list.append(img)
                    mask_list.append(mask)

    return np.array(image_list), np.array(mask_list)