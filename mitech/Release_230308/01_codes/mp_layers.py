import numpy as np
import tensorflow as tf
from tensorflow import keras

from mixed_precision import get_tf_type, get_quant_fn, get_dequant_fn


def create_mask(mp_mat, prec):
    return tf.cast(tf.where(mp_mat == prec, 1, 0), dtype=get_tf_type(prec))


class MPDense(keras.layers.Layer):
    
    def __init__(self, mp_mat, **kwargs):
        rank = len(mp_mat.shape)
        if rank != 2:
            raise ValueError(f"Invalid rank: {rank}")

        super().__init__(**kwargs)

        self.input_size, self.units = mp_mat.shape
        self.mp_mat = tf.convert_to_tensor(mp_mat)
        self.reduced_precs = [str(i) for i in np.unique(mp_mat)]


    def build(self, input_shape):
        if input_shape[-1] != self.input_size:
            raise ValueError(f"Expected input shape of {self.input_size},"
                             f"got {input_shape[-1]}")

        self.w = self.add_weight(
            shape=(self.input_size, self.units),
            initializer="random_normal",
            trainable=True,
            dtype=tf.dtypes.float32,
        )

        self.b = self.add_weight(
            shape=(self.units, ), 
            initializer="random_normal", 
            trainable=True,
            dtype=tf.dtypes.float32,
        )
        
        self.masks = { p: create_mask(self.mp_mat, p)
                      for p in self.reduced_precs}

    def _matmul_in_int(self, x, prec):
        assert prec.startswith('INT')

        quantize = get_quant_fn(prec)
        dequantize = get_dequant_fn(prec)
        mask = self.masks.get(prec)

        quant_x = quantize(x)
        quant_w = tf.math.multiply(quantize(self.w), mask)

        # tf.matmul only support int32 and int64, 
        # so cast to int32, do multiplication in int32,
        # and then cast back to `prec`
        quant_x_32 = tf.cast(quant_x, dtype=tf.dtypes.int32)
        quant_w_32 = tf.cast(quant_w, dtype=tf.dtypes.int32)

        output_32 = tf.matmul(quant_x_32, quant_w_32)
        output = tf.cast(output_32, dtype=get_tf_type(prec))
        return dequantize(output)

    def _matmul_in_reduced_prec(self, x, prec):
        assert x.dtype is tf.dtypes.float32

        if prec.startswith('INT'):
            # int types are handled differently
            return self._matmul_in_int(x, prec)

        quantize = get_quant_fn(prec)
        dequantize = get_dequant_fn(prec)
        mask = self.masks.get(prec)

        quant_x = quantize(x)
        quant_w = tf.math.multiply(quantize(self.w), mask)
        output = tf.matmul(quant_x, quant_w)
        return dequantize(output)

    def call(self, input):
        return tf.add_n(
            [self._matmul_in_reduced_prec(input, prec) for prec in self.reduced_precs]
        ) + self.b


    @classmethod
    def from_normal_dense(cls, dense_layer, mp_mat):
        assert len(dense_layer.get_weights()) == 2
        w, b = dense_layer.get_weights()

        if w.shape != mp_mat.shape:
            raise ValueError(f"Expect mixed-precision matrix of shape {w.shape}, "
                             f"got shape {mp_mat.shape}")

        mp_dense = cls(mp_mat)

        # running dummy input through to set up layer
        mp_dense(np.random.random((1, w.shape[0])))

        mp_dense.set_weights((w, b))
        assert len(mp_dense.get_weights()) == 2
        return mp_dense
