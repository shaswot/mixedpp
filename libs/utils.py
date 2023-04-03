import numpy as np
from tensorflow import keras

from .mixed_precision import SAFE_FLOAT_RANGE


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

def prepare_fashion():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    n_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=n_classes)

    return (x_train, y_train), (x_test, y_test)

def get_kernel_shape(layer):
    assert isinstance(layer, keras.layers.Dense)
    return layer.get_weights()[0].shape


def quantizer_float_range_is_safe(model):
    '''
    Check if the global SAFE_FLOAT_RANGE (used as the range for Symmetric
    Linear Quantizer) is large enough to cover the weight range of given model.
    Return True if it does.
    '''
    all_weights = (w for layer in model.layers for w in layer.get_weights())
    max_weight_val = max(np.max(np.abs(w)) for w in all_weights)

    return max_weight_val <= SAFE_FLOAT_RANGE
