import cv2

def pre_processamento_mascara(img):

    #Redimensionar imagem
    img_resize = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)

    #Converter para cinza
    imgGray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    _, img_bin = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalização das mascaras
    new_img = img_bin / 255.0

    return new_img

    
