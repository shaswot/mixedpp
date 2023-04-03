from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from tensorflow.keras.models import load_model, Sequential

from mixed_precision import SUPPORTED_PRECISIONS
from utils import prepare_mnist, get_kernel_shape

MIN_BITLENGTH = min(prec.bitlength for prec in SUPPORTED_PRECISIONS.values())

@dataclass
class Problem:
    '''
    Utility class to aggregate all problem inputs into one object
    '''

    model_file: Union[str, Path]
    model: Sequential = field(init=False, repr=False)
    max_bit_constraint: int
    layer_idx: int = -1
    solution_shape: Tuple[int, int] = field(init=False)
    test_set: Tuple[np.ndarray, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        self.model = load_model(self.model_file)
        self.solution_shape = get_kernel_shape(
            self.model.get_layer(index=self.layer_idx)
        )

        # check if the max bit constraint is feasible:
        w, h = self.solution_shape
        min_bit_possible = MIN_BITLENGTH * w * h
        if self.max_bit_constraint < min_bit_possible:
            raise ValueError(f"Infeasible max-bit contraint: "
                             f"{self.max_bit_constraint}. Minimum is "
                             f"{min_bit_possible}")

        # currently hardcoded the dataset as MNIST
        _, self.test_set = prepare_mnist()
