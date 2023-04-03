import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np

from quantize import FP32_to_TF32, TF32_to_FP32
from quantize import FP32_to_FP16, FP16_to_FP32
from quantize import FP32_to_BF16, BF16_to_FP32
from quantize import SymmetricLinearQuantizer


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


def test_symm_linear_quantizer():
    num_bits = 8
    float_range = 1.0
    lb, ub = - 2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1
    quantizer = SymmetricLinearQuantizer(num_bits, float_range)

    x = tf.constant([-1.001, -1.0, -0.75, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.001])
    assert x.dtype is tf.dtypes.float32

    y = quantizer.quantize(x)
    x_new = quantizer.dequantize(y)

    assert np.all((lb <= y.numpy()) & (y.numpy() <= ub))
    np.testing.assert_allclose(x.numpy(), x_new.numpy(), rtol=1e-3, atol=1e-2)


def main():
    test_FP32_to_TF32()
    test_symm_linear_quantizer()


if __name__ == '__main__':
    main()
