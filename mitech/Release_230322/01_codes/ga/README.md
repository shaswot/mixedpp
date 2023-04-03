# General-purpose Genetic Algorithm implementation 

This module contain an implementation of GA that aims to be easy to extend for various optimization problems. It seperate the GA implementation details from the problem-specific details by a `GASolution` interface, thus minimize code duplication.

## GA solution
To solve an optimization problem by GA, we need to model that problem in the terminologies of genetic algorithm. This includes:
1. What is a solution to the problem?
2. What is the fitness/objective function?
3. How do we encode the solution with a chromosome?
4. How do chromosomes cross-over and mutate?

Once we figured these things out, we can start implementing a custom `GASolution`.

The `GASolution` class (in `solution.py`) is the main abstraction of this module. It is an interface specifying behaviors that `GeneticAlgorithm` expected. We will implement our custom solution type by subclassing this class. Then, the detail of how a solution should be (initialization, crossover, mutation, objective function,...) is up to the user to implements. 

### Essential elements
A `GASolution` consists of 3 essential elements:
1. `phenotype`: the original form of the solution
2. `genotype`: the encoded form (or chromosome) of the solution
3. `fitness`: the value of objective function

Phenotype determines fitness of a solution, i.e, the objective function takes phenotype as a parameter. Genotype is where crossover and mutation is applied to create new solutions. The mapping between genotype and phenotype should be a *function* from genotype-space to phenotype-space. Being a function ensures that there is no way a genotype can represent two different phenotypes, although two genotypes  mapped to one phenotype.

### Writing a custom `GASolution` class
To create a custom `GASolution` type, write a new class subclassing `GASolution` and override these method

1. `calculate_fitness(cls, phenotype)`
    This method is the objective function. It takes a phenotype and calculate how good/fit the solution is to the problem. 
    The GA implementation expects this method to return a floating point number.
    **Note**: `GeneticAlgorithm` assumes larger fitness value is better

2. `genotype_to_phenotype(cls, genotype)`
    This method implements the mapping from genotype to phenotype. It takes a gene/chromosome as an argument and return the phenotype.

3. `phenotype_to_genotype(cls, phenotype)` (optional)
    This method implements the reverse mapping (from phenotype to genotype). 
    Since nothing guaranteed that there is a reverse mapping, implementing this method is optional.

4. `crossover(cls, parent1, parent2)`
    This method implements the crossover operator.
    It takes 2 instances of `GASolution` as 2 parents and return 2 instances of `GASolution`.

5. `mutate(cls, parent)`
    This method implements the mutation operator.
    It takes an instance of `GASolution` and return the mutated instance.

6. `random_init(cls)`
    This method randomly initialize a `GASolution` instance.
    Typically, in this method, we generate a random genotype, and then use a helper method named `from_genotype(cls, genotype)` to create the new instance

### Helper methods
The `GASolution` class also specifies two helper methods:
1. `from_genotype(cls, genotype)`
    This method use the genotype-to-phenotype mapping to create a new instance, given a genotype.
    Typically, it is used in `random_init`

2. `from_phenotype(cls, phenotype)`
    This method requires the reverse mapping `phenotype_to_genotype` implemented, and use that to create a new instance

Also, if genotype and phenotype of the custom solution type is identical (i.e, the genotype-to-phenotype mapping is the identity function), then instead of subclassing `GASolution`, you should subclassing the alternative `MonoRepresentationSolution`.

### Use the custom solution type
After defining the custom solution type, use it with GA by passing it (the custom class) to the GA

```(python)
class CustomSolution(GASolution):
    ...


def main():
    ga = GeneticAlgorithm(
        solution=CustomSolution,
        pop_size=20,
        num_generations=100,
        crossover_rate=0.25,
        mutation_rate=0.01,
    )

    # run the algorithm
    best_solution = ga.search()
```

#### Notes
`GASolution` implements 3 attribute getters. By default, these getters expose the 3 private attributes (`_phenotype`, `_genotype`, `_fitness`). If these attributes is mutable, it might be accidentally changed during execution, potentially causing incorrectness. In order to make sure this cannot happen, you can override these 3 getter to return a deep copy of the attribute, instead of returning the private attribute directly.

## GA implementation
This section discuss the implementation details of `GeneticAlgorithm`

One notable point is `GeneticAlgorithm` use roulette wheels as selection operator, so it expects `fitness` of `GASolution` to be a `float`.

If you'd like a non-number fitness value, a workaround is to override `build_roulette_wheel` method of `GeneticAlgorithm` to construct the roulette wheel by another way.