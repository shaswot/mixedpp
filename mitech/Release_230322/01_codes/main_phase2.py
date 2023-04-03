import sys
import logging
from argparse import ArgumentParser

from ga import GeneticAlgorithm
from mp_mat import mpmat_to_csv
from problem import Problem
from solution import MPMatmulSolution


def experiment(problem, pop_size, num_generations, crossover_rate, mutation_rate):
    MPMatmulSolution.problem = problem

    genetic_alg = GeneticAlgorithm(
        solution_type=MPMatmulSolution,
        pop_size=pop_size,
        num_generations=num_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
    )

    best_sol = genetic_alg.search()
    history = genetic_alg.history
    return best_sol, history


def main(args):
    problem = Problem(
        model_file=args.model_file,
        layer_idx=args.layer_idx,
        max_bit_constraint=args.max_bit_constraint,
    )

    best_sol, history = experiment(
        problem=problem,
        pop_size=args.pop_size,
        num_generations=args.num_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
    )

    if args.output_file:
        logging.info(f"Writing best solution to {args.output_file}")
        mpmat_to_csv(best_sol.phenotype, args.output_file)

    logging.info(f"Best acc: {best_sol.fitness:.6f}")


if __name__ == '__main__':
    '''
    Args list:
        problem args: model_file, layer_idx, max_bit_constraint
        GA args: pop_size, num_generations, crossover_rate, mutation_rate
        experiment args: output_file, log_file, log_level
    '''

    parser = ArgumentParser()

    # Problem input args
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to pretrained NN')
    parser.add_argument('--layer_idx', type=int, default=-1,
                        help='Index of FC layer chosen to do mixed-precision matmul')
    parser.add_argument('--max_bit_constraint', type=int, default=2500,
                        help='Maximum bits allowed for mixed-precision matmul')

    # GA params
    parser.add_argument('--pop_size', type=int, default=50,
                        help='Population size')
    parser.add_argument('--num_generations', type=int, default=30,
                        help='Number of generations')
    parser.add_argument('--crossover_rate', type=float, default=0.25,
                        help='Crossover rate')
    parser.add_argument('--mutation_rate', type=float, default=0.01,
                        help='Mutation rate')

	# Experiment args
    parser.add_argument('--output_file', type=str, required=False,
                        help='Output file for best solution found')
    parser.add_argument('--log_file', type=str, required=False,
                        help='Path to log file')

    args = parser.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(
            filename=args.log_file,
            mode='w',
            encoding='utf-8',
        ))

    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=handlers,
    )

    main(args)
