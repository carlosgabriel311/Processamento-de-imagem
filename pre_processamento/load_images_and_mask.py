import os
import cv2
import numpy as np
from pre_processamento.pre_processamento_colorido import pre_processamento_colorido
from pre_processamento.pre_processamento_mascara import pre_processamento_mascara


def load_images_and_masks(images_path, masks_path):
    image_list = []
    mask_list = []

    for filename in os.listdir(images_path):
        img_path = os.path.join(images_path, filename)
        if img_path.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(img_path)
            if img is not None:

                img = pre_processamento_colorido(img)

                # Construa o caminho para a máscara correspondente
                mask_filename = filename
                mask_path = os.path.join(masks_path, mask_filename)

                # Carregue a máscara
                mask = cv2.imread(mask_path)
                # Verifique se a máscara foi carregada corretamente
                if mask is not None:
                    mask = pre_processamento_mascara(mask)
                    image_list.append(img)
                    mask_list.append(mask)

    return np.array(image_list), np.array(mask_list)