import sys
import random
import logging
sys.path.append("..")

import numpy as np

from mixed_precision import SUPPORTED_PRECISIONS, get_bitlength
from mp_mat import create_random_mpmat, total_bits
from operators import uniform_crossover
from operators import elemflip_mutation
from operators import repair_invalid_gene


def test_uniform_crossover(size):
    gene1 = create_random_mpmat(size)
    gene2 = create_random_mpmat(size)
    new1, new2 = uniform_crossover(gene1, gene2)

    assert np.logical_or(np.isin(new1, gene1), np.isin(new1, gene2)).all()
    assert np.logical_or(np.isin(new2, gene1), np.isin(new2, gene2)).all()


def test_elemflip_mutation(size):
    gene = create_random_mpmat(size)
    new_gene = elemflip_mutation(gene, choices=list(SUPPORTED_PRECISIONS))
    assert np.sum((new_gene != gene)) == 1


def test_repairing_procedure(size):
    MIN_BITLENGHT = min(get_bitlength(t) for t in SUPPORTED_PRECISIONS)
    w, h = size
    gene = create_random_mpmat(size)
    nbits = total_bits(gene)
    maxbits = max(MIN_BITLENGHT * w * h, round(0.85 * nbits))
    print(f"{nbits=}, {maxbits=}")

    repaired = repair_invalid_gene(gene, nbits, maxbits)
    repaired_nbits = total_bits(repaired)

    assert repaired_nbits <= maxbits, f"{repaired_nbits=}; {maxbits=}"
    assert (maxbits - repaired_nbits) <= MIN_BITLENGHT


def main():
    for _ in range(5):
        a = random.randint(2, 10)
        b = random.randint(2, 10)
        test_uniform_crossover(size=(a, b))
        test_elemflip_mutation(size=(a, b))
        test_repairing_procedure(size=(a, b))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
