# -*- coding: utf-8 -*-
import time
import random
import logging
from typing import Tuple, Type, List

import numpy as np

from .solution import GASolution

'''Genetic Algorithm detailed implementation'''

# return the best (GASolution) from List[GASolution]
def _get_best_solution(population: List[GASolution]) -> GASolution:
    ''' Get the best solution from a population
        Assumming larger fitness value is better
    '''
    return max(population, key=lambda sol: sol.fitness)


class GeneticAlgorithm:
    """ GA implementation """

    def __init__(self, 
                 solution_type: Type[GASolution],
                 pop_size: int,
                 num_generations: int,
                 crossover_rate: float,
                 mutation_rate: float):
        ''' Set up the algorithm 
            Params:
                solution_type:      custom solution class
                pop_size:           population size
                num_generations:    number of generations to run the algorithm
                crossover_rate:     probability of crossover
                mutation_rate:      probability of mutation
        '''
        self.solution       = solution_type
        self.pop_size       = pop_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate
    
    def initialize_population(self) -> List[GASolution]:
        '''Initialize a starting population'''
        return [self.solution.random_init() for _ in range(self.pop_size)]
    
    def do_crossover(self, current_pop) -> Tuple[GASolution, GASolution]:
        '''Randomly choose two solutions from a population to crossover'''
        assert current_pop, "Current population is empty!"

        n = len(current_pop)
        i, j = 0, 0
        while i == j:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
        
        parent1 = current_pop[i]
        parent2 = current_pop[j]
        return self.solution.crossover(parent1, parent2)

    def do_mutation(self, current_pop) -> GASolution:
        '''Randomly choose a solution from a population to mutate'''
        assert current_pop, "Current population is empty!"

        parent = random.choice(current_pop)
        return self.solution.mutate(parent)
    
    def reproduce(self, current_pop) -> List[GASolution]:
        '''Create a new population from current population by crossover and mutation'''
        offspring_pop: List[GASolution] = []

        for _ in range(self.pop_size):
            if random.uniform(0.0, 1.0) < self.crossover_rate:
                o1, o2 = self.do_crossover(current_pop)
                offspring_pop.append(o1)
                offspring_pop.append(o2)
            
            if random.uniform(0.0, 1.0) < self.mutation_rate:
                o = self.do_mutation(current_pop)
                offspring_pop.append(o)
        
        return offspring_pop

    def build_roulette_wheel(self, population) -> List[float]:
        '''Create a roulette wheel on the given population'''
        roulette_wheel = np.empty((len(population), ))
        sum_fitness = sum(solution.fitness for solution in population)

        prob = 0.0
        for i, sol in enumerate(population):
            prob += sol.fitness / sum_fitness
            roulette_wheel[i] = prob

        return roulette_wheel
    
    def roll_roulette_wheel(self, population, roulette_wheel) -> GASolution:
        '''Choose a solution from a population, given its corresponding roulette wheel'''
        assert len(roulette_wheel) == len(population)

        prob = random.uniform(0.0, 1.0)
        output = None

        # assume the roulette wheel is strictly increasing,
        # perform a binary search
        left, right = 0, int(len(roulette_wheel)) - 1
        while True:
            mid = int(left + (right - left)/2)

            if roulette_wheel[mid] < prob and prob <= roulette_wheel[mid+1]:
                # found!
                output = population[mid+1]
                break
            
            if left == 0 and right == 0:
                # special case
                output = population[0]
                break

            if prob <= roulette_wheel[mid]:
                right = mid
            elif prob > roulette_wheel[mid+1]:
                left = mid + 1
        
        assert output, "Binary search go wrong!"
        return output

    def population_selection(self, current_pop: List[GASolution]) -> List[GASolution]:
        '''Select solutions from current population to create a new population, using roulette wheel'''
        new_pop: List[GASolution] = []

        best_solution = _get_best_solution(current_pop)
        new_pop.append(best_solution)       # always preserve the best solution
        
        roulette_wheel = self.build_roulette_wheel(current_pop)
        while len(new_pop) < self.pop_size:
            chosen_solution = self.roll_roulette_wheel(current_pop, roulette_wheel)
            new_pop.append(chosen_solution)
        
        return new_pop
    
    def search(self) -> GASolution:
        '''Main body of the algorithm'''

        population = self.initialize_population()
        current_best = _get_best_solution(population)

        # number of generations in that the fitness is not improved, close 0 is better
        num_unimproved = 0

        self.history = []
        self.history.append(current_best.fitness)

        start = time.perf_counter()
        for generation in range(self.num_generations):
            offsprings = self.reproduce(population)

            population = self.population_selection(population + offsprings)
            new_best = _get_best_solution(population)

            if new_best.fitness <= current_best.fitness:
                num_unimproved += 1 # the fitness is not better
            else:
                num_unimproved = 0  # the fitness is better 
            
            logging.info(
                f'Generation {generation+1:4d}:' \
                f'\tBest: {new_best.fitness:.6f}' \
                f'\tNum_unimproved: {num_unimproved}'
            )

            current_best = new_best
            self.history.append(current_best.fitness)

            # Early stopping
            if num_unimproved > int(0.25 * self.num_generations):
                logging.info(f'Early stopping at generation {generation+1}')
                break
        
        end = time.perf_counter()
        logging.info(f'GA: {generation+1} generations took {end-start} seconds')

        return current_best
