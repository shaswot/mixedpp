import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

from .mp_layers import MPDense
from .utils import get_kernel_shape


def replace_dense_with_mp_dense(mp_mat, orig_model, layer_idx):
    '''
    Replace a normal Dense layer in a Model instance with a corresponding
    mixed-precision layer. Return a new Model instance.
    '''

    assert type(orig_model) is Sequential

    input_shape = orig_model.layers[0].input_shape
    chosen_layer = orig_model.get_layer(index=layer_idx)
    if not isinstance(chosen_layer, Dense):
        raise ValueError(f"Chosen layer (idx {layer_idx}) is not a FC layer")

    w_shape = get_kernel_shape(chosen_layer)
    if mp_mat.shape != w_shape:
        raise ValueError(f"Incompatible mixed-precision matrix. Expect matrix "
                         f"of size {w_shape} but receive size {mp_mat.shape}")

    mp_dense = MPDense.from_normal_dense(chosen_layer, mp_mat)
    layer_list = list(orig_model.layers)
    layer_list[layer_idx] = mp_dense
    new_model = Sequential(layer_list)

    new_model.build(input_shape)
    new_model.compile(
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy'],
    )

    return new_model


def accuracy_on_dataset(model, dataset):
    return model.evaluate(
        *dataset,
        verbose=0,
        return_dict=True
    ).get("accuracy", 0.0)


def mp_accuracy(mp_mat, model, layer_idx, dataset):
    new_model = replace_dense_with_mp_dense(mp_mat, model, layer_idx)
    return accuracy_on_dataset(new_model, dataset)
