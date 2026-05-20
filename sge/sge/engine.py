import sys
import sge.grammar as grammar
import sge.logger as logger
from sge.distribution import DiagonalGaussianDistribution, create_genotype_from_samples
from datetime import datetime
from tqdm import tqdm
import copy
import numpy as np
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate, mutate_level, mutation_prob_mutation, mutate_100
from sge.operators.selection import tournament
from sge.operators.update import update_distributions, grammar_mutation
from sge.parameters import (
    params,
    SearchStrategy,
    AlgorithmMethod,
    GenotypeDistribution,
    set_parameters,
    load_parameters
)

# Global distribution object for genotype sampling
_genotype_distribution = None

class CMA_ES_Distribution:
    """Legacy placeholder. Use DiagonalGaussianDistribution from distribution module instead."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.normal(self.mean, self.std)

    def update(self, new_mean, new_std):
        self.mean = new_mean
        self.std = new_std

def generate_random_individual(max_expansions):
    """
    Generate a random individual with genotypes sampled from learned distributions.
    
    Uses the global CMA-ES-inspired distribution to sample genotype codons.
    This replaces uniform random initialization with learned Gaussian sampling.
    Falls back to uniform sampling when distribution is not initialized.
    """
    ind = {'fitness': None, 'tree_depth': None}
    
    if params['GENOTYPE_INIT'] == 'fixed':
        if params['GENOTYPE_DISTRIBUTION'] == GenotypeDistribution.CMA_ES:
            # Sample from learned Gaussian distributions per non-terminal
            if _genotype_distribution is not None:
                samples = _genotype_distribution.sample()
                genotype = create_genotype_from_samples(samples, grammar.get_non_terminals())
            else:
                # Fallback to uniform if distribution not initialized
                genotype = [[[-1, np.random.uniform(0, 1), -1] for _ in range(max_expansions[nt])] for nt in grammar.get_non_terminals()]
        else:
            # Default: uniform sampling
            genotype = [[[-1, np.random.uniform(0, 1), -1] for _ in range(max_expansions[nt])] for nt in grammar.get_non_terminals()]
    else:
        # Dynamic genotype initialization
        genotype = [[] for _ in grammar.get_non_terminals()]
    
    ind['genotype'] = genotype
    if params['ADAPTIVE_MUTATION']:
        ind['mutation_probs'] = [params['PROB_MUTATION'] for _ in genotype]
    
    if params['ALGORITHM_METHOD'] == AlgorithmMethod.COPSGE or params['ALGORITHM_METHOD'] == AlgorithmMethod.PSGE_COPSGE:
        ind['pcfg'] = grammar.get_pcfg()

    return ind


def make_initial_population(pop_size):
    count = grammar.get_count_references_to_non_terminals()
    for _ in range(pop_size):
        yield generate_random_individual(count)


def evaluate(ind, eval_func):
    mapping_values = [0 for _ in ind['genotype']]
    if params['ALGORITHM_METHOD'] == AlgorithmMethod.COPSGE or params['ALGORITHM_METHOD'] == AlgorithmMethod.PSGE_COPSGE:
        probabilistic_distribution = ind['pcfg']
    else:
        probabilistic_distribution = grammar.get_pcfg()
    phen, tree_depth, gram_counter = grammar.mapping(probabilistic_distribution, ind['genotype'], mapping_values)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth
    ind['grammar_counter'] = gram_counter


def setup(parameters_file_path = None):
    global _genotype_distribution
    
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    params['EXPERIMENT_NAME'] += "/" + str(params['LEARNING_FACTOR'] * 100)

    logger.prepare_dumps()
    np.random.seed(int(params['SEED']))
    grammar.set_path(params['GRAMMAR'])
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])
    grammar.read_grammar(params['LEARNING_STRATEGY'])
    
    if params['GENOTYPE_DISTRIBUTION'] == GenotypeDistribution.CMA_ES:
        print("Initializing CMA-ES-inspired genotype distribution")
        # Initialize CMA-ES-inspired genotype distribution
        # This allows learned sampling instead of uniform random initialization
        max_expansions = grammar.get_count_references_to_non_terminals()
        _genotype_distribution = DiagonalGaussianDistribution(
            non_terminals=grammar.get_non_terminals(),
            max_expansions=max_expansions,
            init_std=0.2
        )



def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    setup(parameters_file_path=parameters_file)
    population = list(make_initial_population(params['POPSIZE']))
    flag = False    # alternate False - best overall
    best = None
    it = 0
    for i in tqdm(population):
        if i['fitness'] is None:
            evaluate(i, evaluation_function)
    previous_population = copy.deepcopy(population)
    while it <= params['GENERATIONS']:        

        population.sort(key=lambda x: x['fitness'])

        # best individual overall
        if not best:
            best = copy.deepcopy(population[0])
            best_gen = copy.deepcopy(best)
        elif population[0]['fitness'] <= best['fitness']:
            best = copy.deepcopy(population[0])

        if flag:
            update_distributions(params['LEARNING_STRATEGY'], [best_gen] + population, params['LEARNING_FACTOR'], params['N_BEST'])
            flag = not flag
        else:
            update_distributions(params['LEARNING_STRATEGY'], population, params['LEARNING_FACTOR'], params['N_BEST'])
            flag = not flag
     
        #     if params['ADAPTIVE_LF']:
        #         params['LEARNING_FACTOR'] += params['ADAPTIVE_INCREMENT']

     
        logger.evolution_progress(it, population, best, best_gen, grammar.get_pcfg(),previous_population)
        
        # Update genotype distributions based on elite individuals
        # This refines the Gaussian sampling for next generation
        if params['GENOTYPE_DISTRIBUTION'] == GenotypeDistribution.CMA_ES:
            _genotype_distribution.update(population, params['N_BEST_GENOTYPE'])

        if params['SEARCH_STRATEGY'] == SearchStrategy.EDA or (params['SEARCH_STRATEGY'] == SearchStrategy.HYBRID and it % 2 == 0):
            new_population = list(make_initial_population(params['POPSIZE'] - params['ELITISM']))

            for i in tqdm(new_population):
                evaluate(i, evaluation_function)
            new_population.sort(key=lambda x: x['fitness'])
            # best individual from the current generation
            best_gen = copy.deepcopy(new_population[0])

            if params['REMAP']:
                for i in tqdm(population[:params['ELITISM']]):
                    evaluate(i, evaluation_function)
            new_population += population[:params['ELITISM']]

        else:
            new_population = []
            while len(new_population) < params['POPSIZE'] - params['ELITISM']:
                if np.random.uniform() < params['PROB_CROSSOVER']:
                    p1 = tournament(population, params['TSIZE'])
                    p2 = tournament(population, params['TSIZE'])
                    ni = crossover(p1, p2)
                else:
                    ni = tournament(population, params['TSIZE'])
                
                if params['ALGORITHM_METHOD'] == AlgorithmMethod.COPSGE or params['ALGORITHM_METHOD'] == AlgorithmMethod.PSGE_COPSGE:
                    ni = grammar_mutation(ni, params['PROB_MUTATION_GRAMMAR'], params['NORMAL_DIST_SD'])
                
                if params['ADAPTIVE_MUTATION']:
                    # Adaptive Facilitated Mutation
                    ni = mutation_prob_mutation(ni)
                    ni = mutate_level(ni)
                else:
                    if params['ALGORITHM_METHOD'] == AlgorithmMethod.COPSGE or params['ALGORITHM_METHOD'] == AlgorithmMethod.PSGE_COPSGE:
                        ni = mutate(ni, params['PROB_MUTATION'], ni['pcfg'])
                    else:
                        ni = mutate(ni, params['PROB_MUTATION'], grammar.get_pcfg())

                new_population.append(ni)


            # new_population += population[:params['ELITISM']]
            for i in tqdm(new_population):
                evaluate(i, evaluation_function)
            new_population.sort(key=lambda x: x['fitness'])
            # best individual from the current generation
            best_gen = copy.deepcopy(new_population[0])

            if params['REMAP']:
                for i in tqdm(population[:params['ELITISM']]):
                    evaluate(i, evaluation_function)
            new_population += population[:params['ELITISM']]

        previous_population = copy.deepcopy(population)
        population = new_population
        it += 1

