import csv

import numpy as np

from .mixed_precision import SUPPORTED_PRECISIONS, get_bitlength

'''
Mixed-precision configuration
'''


STR_TYPE = '<U5'


def create_random_mpmat(size, choices=None):
    rank = len(size)
    if rank != 2:
        raise ValueError(f"MPMatrix has 2 dims, receiving {rank}!")

    if choices and not all(prec in SUPPORTED_PRECISIONS for prec in choices):
        not_supported = list(filter(lambda p: p not in SUPPORTED_PRECISIONS,
                                    choices))
        raise ValueError(f"Invalid precision(s): {not_supported}")

    if not choices:
        choices = list(SUPPORTED_PRECISIONS)

    # TODO: control this randomness by seed
    rng = np.random.default_rng()
    return rng.choice(choices, size).astype(STR_TYPE)


def mpmat_from_csv(fname):
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        return np.array([row for row in reader], dtype=STR_TYPE)


def mpmat_to_csv(mp_mat, fname):
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(mp_mat)


def total_bits(mp_mat):
    return np.sum(np.vectorize(get_bitlength)(mp_mat))
