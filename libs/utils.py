import numpy as np
from tensorflow import keras

from .mixed_precision import SAFE_FLOAT_RANGE

def prepare_cifar100():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    n_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=n_classes)
    
    # data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon = 1e-06,  # epsilon for ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range = 0.,  # set range for random shear
    zoom_range = 0.,  # set range for random zoom
    channel_shift_range = 0.,  # set range for random channel shifts
    fill_mode ='nearest',  # set mode for filling points outside the input boundaries
    cval = 0.,  # value used for fill_mode = "constant"
    rescale = None,  # set rescaling factor (applied before any other transformation)
    preprocessing_function = None,  # set function that will be applied on each input
    data_format = None,  # image data format, either "channels_first" or "channels_last"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    
    return (x_train, y_train), (x_test, y_test)

def prepare_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    n_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=n_classes)
    
    # data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon = 1e-06,  # epsilon for ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range = 0.,  # set range for random shear
    zoom_range = 0.,  # set range for random zoom
    channel_shift_range = 0.,  # set range for random channel shifts
    fill_mode ='nearest',  # set mode for filling points outside the input boundaries
    cval = 0.,  # value used for fill_mode = "constant"
    rescale = None,  # set rescaling factor (applied before any other transformation)
    preprocessing_function = None,  # set function that will be applied on each input
    data_format = None,  # image data format, either "channels_first" or "channels_last"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    
    return (x_train, y_train), (x_test, y_test)

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
