import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage import io

# Carregar a imagem
image = io.imread('imagens\\test\\apple_100.png')

# Transformar a imagem em um array de pixels
image = np.array(image, dtype=np.float64) / 255

# Redimensionar a imagem para ser uma matriz 2D de pixels
w, h, d = tuple(image.shape)
image_array = np.reshape(image, (w * h, d))

# Número de clusters desejados
n_clusters = 5

# Inicializar o modelo KMeans
kmeans = KMeans(n_clusters=n_clusters)

# Aplicar o algoritmo KMeans aos dados
kmeans.fit(image_array)

# Obter rótulos de cluster para cada pixel
labels = kmeans.labels_

# Obter os centróides dos clusters
cluster_centers = kmeans.cluster_centers_

# Redefinir a imagem com base nos centróides
segmented_image = cluster_centers[labels].reshape(w, h, d)

# Mostrar a imagem original e a imagem segmentada


cv2.imshow('Imagem Segmentada', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

