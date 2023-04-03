import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

from utils import prepare_mnist
from mp_mat import create_random_mpmat, mpmat_from_csv, mpmat_to_csv
from mp_layers import MPDense


MODEL_FILE = "fc_net.h5"
MPMAT_FILE = "mpmat.csv"
LAYER_IDX = -1       # index of FC layer chosen to be in mixed-precision
                     # -1 means the last layer


def main():
    if os.path.exists(MODEL_FILE):
        orig_model = tf.keras.models.load_model(MODEL_FILE)
    else:
        raise RuntimeError(f"Model file {MODEL_FILE} does not exists. "
                           "Run `train.py` first!")

    input_shape = orig_model.layers[0].input_shape
    chosen_layer = orig_model.get_layer(index=LAYER_IDX)
    if not isinstance(chosen_layer, Dense):
        raise ValueError(f"Chosen layer (idx {LAYER_IDX}) is not a FC layer")

    W, _ = chosen_layer.get_weights()
    
    if os.path.exists(MPMAT_FILE):
        mp_mat = mpmat_from_csv(MPMAT_FILE)
        assert mp_mat.shape == W.shape
    else:
        print(f"Mixed-precision matrix file not found -> randomly create one")
        mp_mat = create_random_mpmat(size=W.shape)
        mpmat_to_csv(mp_mat, MPMAT_FILE)

    new_mp_dense = MPDense.from_normal_dense(chosen_layer, mp_mat)
    layer_list = list(orig_model.layers)
    layer_list[LAYER_IDX] = new_mp_dense
    new_model = keras.models.Sequential(layer_list)

    new_model.build(input_shape)
    new_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'],
    )

    # make sure new model has the same weight as the original model
    for l1, l2 in zip(orig_model.layers, new_model.layers):
        assert len(l1.get_weights()) == len(l2.get_weights())

        for w1, w2 in zip(l1.get_weights(), l2.get_weights()):
            np.testing.assert_array_equal(w1, w2)

    _, (x_test, y_test) = prepare_mnist()

    orig_loss, orig_acc = orig_model.evaluate(x_test, y_test)
    new_loss, new_acc = new_model.evaluate(x_test, y_test)
    
    print(f"Original model: loss: {orig_loss}, acc: {orig_acc}")
    print(f"Mixed-precision model: loss: {new_loss}, acc: {new_acc}")
    


if __name__ == "__main__":
    main()
