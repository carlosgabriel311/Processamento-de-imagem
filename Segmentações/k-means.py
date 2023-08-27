import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage import io

# Carregar a imagem
image = io.imread('apples\\test\\apple_100.png')

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
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagem Segmentada")
plt.imshow(segmented_image)
plt.axis('off')

plt.show()
