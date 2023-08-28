import cv2

def pre_processamento_colorido(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Redimensionar imagem
    img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

    #Aumentar contraste da imagem
    img_contraste = img_resize *1.3
    
    #Filtrar imagem
    imgGray_gaus_3x3 = cv2.GaussianBlur(img_contraste, (3,3), 0)

    # Normalização das imagens
    new_img = imgGray_gaus_3x3 / 255.0
                    
    return new_img