import cv2
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
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
    y_pred_thresholded = (y_pred > 0.5).astype(int)
    # Calcular as métricas
    accuracy = accuracy_score(y_test.flatten(), y_pred_thresholded.flatten())
    precision = precision_score(y_test.flatten(), y_pred_thresholded.flatten())
    recall = recall_score(y_test.flatten(), y_pred_thresholded.flatten())
    f1 = f1_score(y_test.flatten(), y_pred_thresholded.flatten())
    iou = jaccard_score(y_test.flatten(), y_pred_thresholded.flatten())

# Imprimir as métricas
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")

    index = 26  # Índice da imagem de teste
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