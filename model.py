import keras
from keras import layers

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        layers.experimental.preprocessing.RandomRotation(0.1)
    ]
)
def LeNet_like(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=64, kernel_size=(8, 8), activation='relu', strides=1, padding='same')(x)

    x = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu', strides=1, padding='same')(x)

    x = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu', strides=1, padding='same')(x)
    outputs = x
    return keras.Model(inputs, outputs)

def AlexNet_like(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=128, kernel_size=(4, 4), activation='relu', strides=1, padding='same')(x)

    x = layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu', strides=1, padding='same')(x)

    x = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu', strides=1, padding='same')(x)

    outputs = x   
    return keras.Model(inputs, outputs)

def AlexNet_like(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=512, kernel_size=(4, 4), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=512, kernel_size=(1, 1), activation='relu', strides=1, padding='same')(x)
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu', strides=1, padding='same')(x)

    outputs = x   
    return keras.Model(inputs, outputs)

