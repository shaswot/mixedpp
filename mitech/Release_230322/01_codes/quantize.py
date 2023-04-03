import tensorflow as tf


def FP32_to_TF32(x):
    '''
    Convert FP32 tensor to TF32 by turning off 13 least-significant bits
    '''

    x = tf.bitcast(x, tf.dtypes.uint32)
    x = tf.bitwise.bitwise_and(x, tf.constant(0xFFFFE000, dtype=tf.dtypes.uint32))
    x = tf.bitcast(x, tf.dtypes.float32)
    return x


# basically the same process
TF32_to_FP32 = FP32_to_TF32


# with FP16 and BF16, trust `tf.cast` to do the right thing
def FP32_to_FP16(x):
    return tf.cast(x, dtype=tf.dtypes.float16)


def FP16_to_FP32(x):
    return tf.cast(x, dtype=tf.dtypes.float32)


def FP32_to_BF16(x):
    return tf.cast(x, dtype=tf.dtypes.bfloat16)


def BF16_to_FP32(x):
    return tf.cast(x, dtype=tf.dtypes.float32)

class SymmetricLinearQuantizer:
    def __init__(self, num_int_bits, float_range):
        if num_int_bits == 16:
            self._tf_type = tf.dtypes.int16
        elif num_int_bits == 8:
            self._tf_type = tf.dtypes.int8
        else:
            raise ValueError(f"Unsupported bid width: {num_int_bits}")

        self._ub = 2 ** (num_int_bits - 1) - 1
        self._lb = - 2 ** (num_int_bits - 1)
        self._scale_factor = (2 ** num_int_bits - 1) / (2 * float_range)


    def quantize(self, x):
        x = x * self._scale_factor
        x = tf.math.round(x)
        x = tf.clip_by_value(x, self._lb, self._ub)
        return tf.cast(x, self._tf_type)

    def dequantize(self, x):
        return tf.cast(x, dtype=tf.dtypes.float32) / self._scale_factor
