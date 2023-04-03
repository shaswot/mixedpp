import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np

from mixed_precision import SUPPORTED_PRECISIONS, ReducedPrecision
from mp_mat import create_random_mpmat, mpmat_from_csv, STR_TYPE
from mp_layers import create_mask
from mp_layers import MPDense


def test_creating_mask():
    c = mpmat_from_csv('test.csv')
    p = c[0, 0]
    m = create_mask(tf.convert_to_tensor(c), p)
    print(p)
    print(c)
    print(m)
    print(m.dtype)


def test_calling_mp_dense():
    batch_size = 4
    input_size = 2
    mp_mat = mpmat_from_csv('test.csv')

    layer = MPDense(mp_mat)
    x = np.random.random(size=(batch_size, input_size)).astype(np.float32)
    y = layer(x)
    print(x, x.dtype)
    print(y)


def test_mp_dense_with_fp32():
    # add FP32 to the list of supported precision, for this test only
    global SUPPORTED_PRECISIONS
    SUPPORTED_PRECISIONS['FP32'] = ReducedPrecision(
        tf_type=tf.dtypes.float32,
        bitlength=32,
        quant_fn=tf.identity,
        dequant_fn=tf.identity,
    )

    batch_size = 32
    input_size = 8
    units = 4
    mp_mat = np.full(shape=(input_size, units), fill_value='FP32', dtype=STR_TYPE)

    layer1 = tf.keras.layers.Dense(units=units)
    layer2 = MPDense(mp_mat)

    # run the layer first time to set up
    dummy = np.random.random(size=(batch_size, input_size)).astype(np.float32)
    layer1(dummy)
    layer2(dummy)

    assert len(layer1.get_weights()) == len(layer2.get_weights()) == 2
    layer2.set_weights(layer1.get_weights())    # make sure two layers share the same weights

    l1_w, l1_b = layer1.get_weights()
    l2_w, l2_b = layer2.get_weights()
    np.testing.assert_array_equal(l1_w, l2_w)
    np.testing.assert_array_equal(l1_b, l2_b)

    x = np.random.random(size=(batch_size, input_size)).astype(np.float32)
    y1 = layer1(x)
    y2 = layer2(x)

    np.testing.assert_array_equal(y1.numpy(), y2.numpy())


def test_create_from_normal_dense():
    batch_size = 2
    input_size, units = 4, 5
    mp_mat = create_random_mpmat((input_size, units))
    x = np.random.random((batch_size, input_size))

    dense = tf.keras.layers.Dense(units)
    dense(x)

    assert len(dense.get_weights()) == 2
    w0, b0 = dense.get_weights()

    mp_dense = MPDense.from_normal_dense(dense, mp_mat)
    w, b = mp_dense.get_weights()

    np.testing.assert_array_equal(w0, w)
    np.testing.assert_array_equal(b0, b)



def main():
    #test_creating_mask()
    #test_calling_mp_dense()
    test_mp_dense_with_fp32()
    test_create_from_normal_dense()


if __name__ == "__main__":
    main()
