from collections import namedtuple

import tensorflow as tf

from quantize import FP32_to_TF32, TF32_to_FP32
from quantize import FP32_to_FP16, FP16_to_FP32
from quantize import FP32_to_BF16, BF16_to_FP32


ReducedPrecision = namedtuple(
    'ReducePrecision',
    ['tf_type', 'quant_fn', 'dequant_fn'],
)

FP16 = ReducedPrecision(
    tf_type=tf.dtypes.float16,
    quant_fn=FP32_to_FP16,
    dequant_fn=FP16_to_FP32,
)

TF32 = ReducedPrecision(
    tf_type=tf.dtypes.float32,  # use FP32 as underlying representation
    quant_fn=FP32_to_TF32,
    dequant_fn=TF32_to_FP32,
)

BF16 = ReducedPrecision(
    tf_type=tf.dtypes.bfloat16,
    quant_fn=FP32_to_BF16,
    dequant_fn=BF16_to_FP32,
)


# TODO: support INT{8,16}
SUPPORTED_PRECISIONS = {
    'FP16': FP16,
    'BF16': BF16,
    'TF32': TF32,
}


def get_tf_type(type_):
    return SUPPORTED_PRECISIONS[type_].tf_type


def get_quant_fn(type_):
    return SUPPORTED_PRECISIONS[type_].quant_fn


def get_dequant_fn(type_):
    return SUPPORTED_PRECISIONS[type_].dequant_fn
