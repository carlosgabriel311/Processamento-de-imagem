import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
from pre_processamento.load_images_and_mask import load_images_and_masks
from pos_processamento.pos_processamento_mascara import pos_processamento_mascara

def test_fcn():
    # Defina o caminho para suas imagens de treinamento e máscaras de segmentação
    test_images_path = 'imagens\\test'
    test_masks_path = 'imagens\\test_ann'

    # Tamanho desejado para as imagens de entrada
    input_size = (256, 256)

    x_test, y_test = load_images_and_masks(test_images_path, test_masks_path)

    saved_model_path = 'fcn'
    loaded_model = keras.models.load_model(saved_model_path)
    y_pred = loaded_model.predict(x_test)

    index = 10  # Índice da imagem de teste
    mask = pos_processamento_mascara(y_pred[index])
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(x_test[index])
    plt.title('Imagem de Teste')
    plt.subplot(132)
    plt.imshow(y_test[index].squeeze(), cmap='gray')
    plt.title('Máscara Real')
    plt.subplot(133)
    plt.imshow(mask, cmap='gray')
    plt.title('Máscara Prevista')
    plt.savefig(f'mascara_prevista{index}.png')
    cv2.imwrite(f'mascara_isolada{index}.png', mask)