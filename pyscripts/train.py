import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

from utils import prepare_mnist


def fully_connected_net(input_shape, nb_classes):
    return models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Flatten(),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(nb_classes, activation='softmax'),
            ],
            name='fc_net',
    )


def lenet(input_shape, nb_classes):
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


def main():
    # global seed
    tf.random.set_seed(42)
    np.random.seed(42)

    batch_size = 32
    n_epochs = 10

    (x_train, y_train), (x_test, y_test) = prepare_mnist()

    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    n_classes = y_train.shape[1]

    #model = lenet(input_shape, n_classes)
    model = fully_connected_net(input_shape, n_classes)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'],
    )
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs,
            validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    print(score)
    model.save(f'{model.name}.h5')
    model.summary()


if __name__ == '__main__':
    main()
