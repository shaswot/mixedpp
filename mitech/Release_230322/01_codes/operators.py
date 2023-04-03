import math
import random
import logging

import numpy as np


# TODO: control randomness by seed

def uniform_crossover(parent1, parent2):
    assert parent1.shape == parent2.shape

    prob = np.random.uniform(size=parent1.shape)
    offspring1 = np.where(prob > 0.5, parent1, parent2)
    offspring2 = np.where(prob > 0.5, parent2, parent1)
    return offspring1, offspring2


def elemflip_mutation(parent, choices):
    idx = random.choice(list(np.ndindex(parent.shape)))
    if parent[idx] in choices:
        choices.remove(parent[idx])
    offspring = parent.copy()
    offspring[idx] = random.choice(choices)
    return offspring


def repair_invalid_gene(gene, nbits, maxbits):
    assert nbits > maxbits

    REPAIR_TABLE = {
        # prec : ( options to change to,     nbits diff)
        "TF32" : (("BF16", "FP16", "INT16"), 19 - 16),
        "BF16" : (("INT8", ),                16 - 8),
        "FP16" : (("INT8", ),                16 - 8),
        "INT16": (("INT8", ),                16 - 8),
    }

    new_gene = gene.copy()
    for prec, (options, nbits_diff) in REPAIR_TABLE.items():
        n_changes_needed = math.ceil((nbits - maxbits) / nbits_diff)
        mask = np.where(new_gene == prec, 1, 0)
        n_changes_available = np.sum(mask)

        n_actual_changes = min(n_changes_needed, n_changes_available)
        if n_changes_available > n_changes_needed:
            # reduce number of changes in mask just enough
            # so nbits is smaller than maxbits
            n_changes_undo = n_changes_available - n_changes_needed
            changing_idx = list(i for i in np.ndindex(mask.shape) if mask[i])
            undo_idx = random.sample(
                population=changing_idx,
                k=n_changes_undo,
            )
            for i in undo_idx:
                mask[i] = 0

        # repair
        new_prec = random.choice(options)
        new_gene = np.where(mask, new_prec, new_gene)
        old_nbits = nbits
        nbits -= n_actual_changes * nbits_diff
        logging.debug(f"{prec} -> {new_prec}: {n_changes_needed=} {n_changes_available=}; {old_nbits} => {nbits}")

        if nbits <= maxbits:
            break

    return new_gene
