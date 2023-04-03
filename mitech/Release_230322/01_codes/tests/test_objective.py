import os
import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf

from mp_mat import mpmat_from_csv
from objective import replace_dense_with_mp_dense

MODEL_FILE = "../fc_net.h5"
MPMAT_FILE = "../mpmat.csv"
assert os.path.exists(MODEL_FILE)
assert os.path.exists(MPMAT_FILE)

def test_weights_are_identical():
    orig_model = tf.keras.models.load_model(MODEL_FILE)
    new_model = replace_dense_with_mp_dense(
        mp_mat=mpmat_from_csv(MPMAT_FILE),
        orig_model=orig_model,
        layer_idx=-1,
    )

    assert len(orig_model.layers) == len(new_model.layers)
    for l1, l2 in zip(orig_model.layers, new_model.layers):
        assert len(l1.get_weights()) == len(l2.get_weights())
        for w1, w2 in zip(l1.get_weights(), l2.get_weights()):
            np.testing.assert_array_equal(w1, w2)


def main():
    test_weights_are_identical()


if __name__ == "__main__":
    main()
