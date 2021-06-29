import keras
from keras import layers
from keras import optimizers

data_augmentation = keras.Sequential(
    [
        layers.preprocessing.image_preprocessing.RandomFlip('horizontal'),
        layers.preprocessing.image_preprocessing.RandomRotation(0.1)
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
    outputs = x[0][0][0] 
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

    outputs = x[0][0][0]    
    return keras.Model(inputs, outputs)

def VggNet_like(input_shape):
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

    outputs = x[0][0][0]   
    return keras.Model(inputs, outputs)

MODELS = {
    'lenet':{
        "model":LeNet_like,
        "init_weights":None
    },
    'alexnet':{
        "model":AlexNet_like,
        "init_weights":None
    },
    'vggnet':{
        "model":VggNet_like,
        "init_weights":None
    }
}

if __name__ == '__main__':
    pass