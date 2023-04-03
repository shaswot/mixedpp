import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np

from quantize import FP32_to_TF32, TF32_to_FP32
from quantize import FP32_to_FP16, FP16_to_FP32
from quantize import FP32_to_BF16, BF16_to_FP32


def test_FP32_to_TF32():
    '''
    Test example taken from: https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_tensor_float_32_execution
    '''

    x = tf.fill((2, 2), 1.0001)
    y = tf.fill((2, 2), 1.)
    assert x.dtype is tf.dtypes.float32
    assert y.dtype is tf.dtypes.float32

    output = tf.linalg.matmul(FP32_to_TF32(x), FP32_to_TF32(y))

    np.testing.assert_array_equal(
		output.numpy(),
		np.array([[2., 2.], [2., 2.]])
	)


def main():
    test_FP32_to_TF32()


if __name__ == '__main__':
    main()
