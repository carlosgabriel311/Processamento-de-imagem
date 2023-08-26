from tensorflow import keras

from FCN.load_images_and_mask import load_images_and_masks

# Defina o caminho para suas imagens de treinamento e máscaras de segmentação
train_images_path = 'C:\\Users\\carlo\\Documents\\Processamento-de-imagem\\apples\\train'
train_masks_path = 'C:\\Users\\carlo\\Documents\\Processamento-de-imagem\\apples\\train_ann'

# Tamanho desejado para as imagens de entrada
input_size = (256, 256)


# Carregue as imagens de treinamento e máscaras de segmentação
X_train, y_train = load_images_and_masks(train_images_path, train_masks_path, input_size)


# Crie o modelo FCN
model = keras.Sequential([
    keras.layers.Input(shape=(input_size[0], input_size[1], 3)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
    keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')
])

# Compile o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treine o modelo
model.fit(X_train, y_train, epochs=2)

saved_model_path = 'C:\\Users\\carlo\\Documents\\Processamento-de-imagem'

# Salve o modelo em disco
model.save(saved_model_path)
