import cv2
from matplotlib import pyplot as plt
import numpy as np

def isoseg(image, num_clusters, max_iterations=100):
    # Converter a imagem para tons de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Inicializar os centroides aleatoriamente
    centroids = np.random.randint(0, 256, size=num_clusters)
    
    for _ in range(max_iterations):
        # Atribuir cada pixel ao cluster mais próximo
        labels = np.argmin(np.abs(gray_image[:, :, np.newaxis] - centroids), axis=2)
        
        # Atualizar os centroides para a média dos valores dos pixels em cada cluster
        new_centroids = np.array([gray_image[labels == i].mean() for i in range(num_clusters)])
        
        # Verificar a convergência
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    # Atribuir uma cor a cada cluster
    colors = np.array([np.random.randint(0, 256, size=3) for _ in range(num_clusters)])
    
    # Criar uma imagem segmentada com base nas cores dos clusters
    segmented_image = colors[labels]
    
    return segmented_image

# Carregar a imagem
image = cv2.imread('C:\\Users\\carlo\\Documents\\Processamento-de-imagem\\imagens\\train\\salad_1.png')

# Definir o número de clusters desejado
num_clusters = 2

# Segmentar a imagem
segmented_image = isoseg(image, num_clusters)

# Mostrar a imagem segmentada
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(segmented_image)
plt.title('Imagem')
plt.savefig('isoseg.png')
