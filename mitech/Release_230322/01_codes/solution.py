import logging

from mixed_precision import SUPPORTED_PRECISIONS
from mp_mat import create_random_mpmat, total_bits
from problem import Problem
from objective import mp_accuracy
from operators import uniform_crossover, elemflip_mutation
from operators import repair_invalid_gene
from ga import MonoRepresentationSolution


class MPMatmulSolution(MonoRepresentationSolution):
    """
        Solution for maximizing accuracy of NN with mixed-precision matmul
    """

    problem: Problem

    def __new__(cls, *args, **kwargs):
        # the `problem` attribute must be set before doing anything
        if not cls.problem:
            raise RuntimeError("Problem not defined")
        return super().__new__(cls)

    @classmethod
    def calculate_fitness(cls, phenotype):
        return mp_accuracy(
            mp_mat=phenotype,
            model=cls.problem.model,
            layer_idx=cls.problem.layer_idx,
            dataset=cls.problem.test_set
        )

    @classmethod
    def from_genotype(cls, genotype):
        nbits = total_bits(genotype)
        maxbits = cls.problem.max_bit_constraint
        if nbits > maxbits:
            logging.debug("Repairing a gene")
            genotype = repair_invalid_gene(genotype, nbits, maxbits)

        return super().from_genotype(genotype)


    @classmethod
    def crossover(cls, parent1, parent2):
        new1, new2 = uniform_crossover(parent1.genotype, parent2.genotype)
        return cls.from_genotype(new1), cls.from_genotype(new2)

    @classmethod
    def mutate(cls, parent):
        new = elemflip_mutation(parent.genotype,
                                choices=list(SUPPORTED_PRECISIONS))
        return cls.from_genotype(new)

    @classmethod
    def random_init(cls):
        return cls.from_genotype(create_random_mpmat(cls.problem.solution_shape))
