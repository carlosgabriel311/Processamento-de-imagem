import cv2

# Carregue as duas imagens
imagem = cv2.imread('C:\\Users\\carlo\\Documents\\Processamento_imagem\\imagens\\test\\bean_101.png')
mask = cv2.imread('C:\\Users\\carlo\\Documents\\Processamento_imagem\\mascara_isolada26.png')

mask = cv2.resize(mask, (imagem.shape[1], imagem.shape[0]))

# Aplica um lógica and em cada pixel da imagem
imagem_multiplicada = cv2.bitwise_and(mask, imagem)

# Salve a imagem resultante
cv2.imwrite('imagem_resultante.png', imagem_multiplicada)


