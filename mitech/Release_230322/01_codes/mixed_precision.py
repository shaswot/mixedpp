from collections import namedtuple

import tensorflow as tf

from quantize import FP32_to_TF32, TF32_to_FP32
from quantize import FP32_to_FP16, FP16_to_FP32
from quantize import FP32_to_BF16, BF16_to_FP32
from quantize import SymmetricLinearQuantizer


# TODO: determine this value programmatically
# the chosen float range that is large enough to *possibly* cover the entire
# weight value of NN
SAFE_FLOAT_RANGE = 1.5


ReducedPrecision = namedtuple(
    'ReducePrecision',
    ['tf_type', 'bitlength', 'quant_fn', 'dequant_fn'],
)

FP16 = ReducedPrecision(
    tf_type=tf.dtypes.float16,
    bitlength=16,
    quant_fn=FP32_to_FP16,
    dequant_fn=FP16_to_FP32,
)

TF32 = ReducedPrecision(
    tf_type=tf.dtypes.float32,  # use FP32 as underlying representation
    bitlength=19,
    quant_fn=FP32_to_TF32,
    dequant_fn=TF32_to_FP32,
)

BF16 = ReducedPrecision(
    tf_type=tf.dtypes.bfloat16,
    bitlength=16,
    quant_fn=FP32_to_BF16,
    dequant_fn=BF16_to_FP32,
)

_int16_quantizer = SymmetricLinearQuantizer(num_int_bits=16,
                                            float_range=SAFE_FLOAT_RANGE)
INT16 = ReducedPrecision(
    tf_type=tf.dtypes.int16,
    bitlength=16,
    quant_fn=_int16_quantizer.quantize,
    dequant_fn=_int16_quantizer.dequantize,
)

_int8_quantizer = SymmetricLinearQuantizer(num_int_bits=8,
                                           float_range=SAFE_FLOAT_RANGE)
INT8 = ReducedPrecision(
    tf_type=tf.dtypes.int8,
    bitlength=8,
    quant_fn=_int8_quantizer.quantize,
    dequant_fn=_int16_quantizer.dequantize,
)

SUPPORTED_PRECISIONS = {
    'FP16': FP16,
    'BF16': BF16,
    'TF32': TF32,
    'INT16': INT16,
    'INT8': INT8,
}


def get_tf_type(type_):
    return SUPPORTED_PRECISIONS[type_].tf_type


def get_bitlength(type_):
    return SUPPORTED_PRECISIONS[type_].bitlength


def get_quant_fn(type_):
    return SUPPORTED_PRECISIONS[type_].quant_fn


def get_dequant_fn(type_):
    return SUPPORTED_PRECISIONS[type_].dequant_fn
