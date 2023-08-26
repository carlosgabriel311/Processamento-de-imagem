from  pre_processamento_cinza import pre_processamento_cinza
import cv2
import os
import numpy as np

# Pasta contendo suas imagens de treinamento originais
input_folder = 'C:\\Users\\carlo\\Documents\\Processamento-de-imagem\\imagens\\train_ann'

# Pasta para salvar as imagens redimensionadas
output_folder = 'C:\\Users\\carlo\\Documents\\Processamento-de-imagem\\imagens_pre_processadas_cinza\\train_ann'

# Certifique-se de que a pasta de saída exista
os.makedirs(output_folder, exist_ok=True)

# Tamanho desejado para redimensionamento (por exemplo, 256x256)

# Listar todos os arquivos na pasta de entrada
image_files = os.listdir(input_folder)

for image_file in image_files:
    # Construa o caminho completo para a imagem de entrada
    input_path = os.path.join(input_folder, image_file)

    # Carregue a imagem usando o OpenCV
    image = cv2.imread(input_path)

    # Redimensione a imagem para o tamanho desejado
    img = cv2.resize(image, (256, 256))

    # Construa o caminho completo para a imagem redimensionada na pasta de saída
    output_path = os.path.join(output_folder, image_file)

    # Salve a imagem redimensionada
    cv2.imwrite(output_path, img)

# Agora, 'output_folder' contém todas as imagens de treinamento redimensionadas
