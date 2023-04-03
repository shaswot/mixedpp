import sys
sys.path.append('..')

import numpy as np

from mixed_precision import SUPPORTED_PRECISIONS, get_bitlength
from mp_mat import create_random_mpmat
from mp_mat import mpmat_from_csv
from mp_mat import mpmat_to_csv
from mp_mat import total_bits
from mp_mat import STR_TYPE


def test_mpmat_from_and_to_csv():
    fname = 'test.csv'
    c = mpmat_from_csv(fname)
    assert c.dtype == STR_TYPE

    newfile = 'new.csv'
    mpmat_to_csv(c, newfile)

    new_c = mpmat_from_csv(newfile)
    np.testing.assert_equal(c, new_c)


def test_random_generate_mpmat(size, choices=None):
    c = create_random_mpmat(size, choices)
    assert c.dtype == STR_TYPE
    print(c)


def test_calc_total_bits_of_mpmat(size):
    bit_lengths = list(get_bitlength(type_) for type_ in SUPPORTED_PRECISIONS)
    LARGEST_LEN = max(bit_lengths)
    SMALLEST_LEN = min(bit_lengths)

    c = create_random_mpmat(size)
    sum_bits = total_bits(c)
    assert (SMALLEST_LEN * size[0] * size[1]) <= sum_bits <= (LARGEST_LEN * size[0] * size[1])


def main():
    test_mpmat_from_and_to_csv()
    test_random_generate_mpmat(size=(2, 3))
    test_random_generate_mpmat(size=(2, 3), choices=['TF32', 'BF16'])

    try:
        test_random_generate_mpmat(size=(2, 3, 4))
    except ValueError as e:
        print("Catched: ", e)

    test_calc_total_bits_of_mpmat(size=(2, 3))


if __name__ == '__main__':
    main()
