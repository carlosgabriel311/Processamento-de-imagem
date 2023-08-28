from matplotlib import pyplot as plt
from tensorflow import keras

from pre_processamento.load_images_and_mask import load_images_and_masks

def modelo(input_shape):
    inputs = keras.layers.Input(input_shape)
    
    # Encoder
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Decoder
    up1 = keras.layers.UpSampling2D(size=(2, 2))(pool1)
    concat1 = keras.layers.concatenate([conv1, up1], axis=-1)
    conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)
    
    # Output layer
    output = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv2)
    
    model = keras.models.Model(inputs=inputs, outputs=output)
    return model

def unet():
    # Defina o caminho para suas imagens de treinamento e máscaras de segmentação
    train_images_path = 'imagens\\train'
    train_masks_path = 'imagens\\train_ann'
    val_images_path = 'imagens\\val'
    val_masks_path = 'imagens\\val_ann'

    X_train, y_train = load_images_and_masks(train_images_path, train_masks_path)
    X_val, y_val = load_images_and_masks(val_images_path, val_masks_path)


    model = modelo(input_shape=(256, 256, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

    saved_model_path = 'unet'
    # Salve o modelo em disco
    model.save(saved_model_path)
