import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

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