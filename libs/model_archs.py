import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

def vgg16D(input_shape, nb_classes):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
    return models.Sequential(
        [
            # 1st Conv Block
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 2nd Conv Block
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 3rd Conv Block
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 4th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 5th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.4),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(units=512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=nb_classes, activation='softmax')
        ],
        name='vgg16D',
    )


def vgg16C(input_shape, nb_classes):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
    return models.Sequential(
        [
            # 1st Conv Block
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.2),
            
            # 2nd Conv Block
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.3),
            
            # 3rd Conv Block
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.4),
            
            # 4th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.4),
            
            # 5th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.4),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(units=1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(units=1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(units=nb_classes, activation='softmax')
        ],
        name='vgg16C',
    )


def vgg16B(input_shape, nb_classes):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
    return models.Sequential(
        [
            # 1st Conv Block
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 2nd Conv Block
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 3rd Conv Block
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 4th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 5th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(units=1024, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(units=1024, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(units=nb_classes, activation='softmax')
        ],
        name='vgg16B',
    )


def vgg16A(input_shape, nb_classes):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
    return models.Sequential(
        [
            # 1st Conv Block
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 2nd Conv Block
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 3rd Conv Block
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # # 4th Conv Block
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            # # 5th Conv Block
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(units=1024, activation='relu'),
            layers.Dropout(0.2),
            # layers.Dense(units=4096, activation='relu'),
            layers.Dense(units=nb_classes, activation='softmax')
        ],
        name='vgg16A',
    )



def lenetA(input_shape, nb_classes):
    return models.Sequential(
        [
            layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                input_shape=input_shape,
            ),
            layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(1, 1),
                padding='valid',
            ),
            layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                padding='valid',
            ),
            layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid',
            ),
            layers.Conv2D(
                filters=120,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                padding='valid',
            ),
            layers.Flatten(),
            layers.Dense(84, activation='relu'),
            layers.Dense(nb_classes, activation='softmax'),
        ],
        name='lenetA',
    )

def fcA(input_shape, nb_classes):
    return models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Flatten(),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(nb_classes, activation='softmax'),
            ],
            name='fcA',
    )