import cv2

def pre_processamento_cinza(img):

    #Redimensionar imagem
    img_resize = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)

    #Converter para cinza
    imgGray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    #Equalizar histograma
    imgEq = cv2.equalizeHist(imgGray)
    
    #Filtrar imagem equalizada
    imgGray_gaus_3x3 = cv2.GaussianBlur(imgEq, (3,3), 0)

    return imgGray_gaus_3x3

    
