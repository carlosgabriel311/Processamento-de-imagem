import cv2
import numpy as np

# Carregue a imagem
imagem = cv2.imread('C:\\Users\\carlo\\Documents\\Processamento-de-imagem\\imagens\\train\\salad_0.png')

# Converta a imagem para tons de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar uma transformação morfológica (abertura) para eliminar ruídos
kernel = np.ones((3, 3), np.uint8)
abertura = cv2.morphologyEx(imagem_cinza, cv2.MORPH_OPEN, kernel, iterations=2)

# Calcule o gradiente da imagem (gradiente de Sobel)
gradiente_x = cv2.Sobel(abertura, cv2.CV_64F, 1, 0, ksize=5)
gradiente_y = cv2.Sobel(abertura, cv2.CV_64F, 0, 1, ksize=5)
gradiente = cv2.subtract(gradiente_x, gradiente_y)
gradiente = cv2.convertScaleAbs(gradiente)

# Limiarize o gradiente para obter uma imagem binária
_, binarizada = cv2.threshold(gradiente, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Realize a transformação de distância para encontrar a região dos marcadores
dist_transform = cv2.distanceTransform(binarizada, cv2.DIST_L2, 5)
_, marcadores = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
marcadores = np.uint8(marcadores)
marcadores = cv2.connectedComponents(marcadores)[1]

# Aplique a segmentação Watershed
cv2.watershed(imagem, marcadores)


imagem[marcadores == -1] = [0, 0, 255]


# Exiba a imagem segmentada
cv2.imshow('Imagem Segmentada', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

