from tensorflow import keras
import matplotlib.pyplot as plt

from FCN.load_images_and_mask import load_images_and_masks


# Defina o caminho para suas imagens de treinamento e máscaras de segmentação
test_images_path = 'C:\\Users\\carlo\\Documents\\Processamento-de-imagem\\apples\\test'
test_masks_path = 'C:\\Users\\carlo\\Documents\\Processamento-de-imagem\\apples\\test_ann'

# Tamanho desejado para as imagens de entrada
input_size = (256, 256)

x_test, y_test = load_images_and_masks(test_images_path, test_masks_path, input_size)

saved_model_path = 'C:\\Users\\carlo\\Documents\\Processamento-de-imagem'
loaded_model = keras.models.load_model(saved_model_path)
y_pred = loaded_model.predict(x_test)

index = 2  # Índice da imagem de teste
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(x_test[index])
plt.title('Imagem de Teste')
plt.subplot(132)
plt.imshow(y_test[index].squeeze(), cmap='gray')
plt.title('Máscara Real')
plt.subplot(133)
plt.imshow(y_pred[index].squeeze(), cmap='gray')
plt.title('Máscara Prevista')
plt.savefig('teste.png')