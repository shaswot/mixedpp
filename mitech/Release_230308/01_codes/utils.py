import numpy as np
from tensorflow import keras


def prepare_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    n_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=n_classes)

    return (x_train, y_train), (x_test, y_test)
