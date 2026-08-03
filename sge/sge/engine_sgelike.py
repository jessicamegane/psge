import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
import copy
import numpy as np
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate, mutate_level, mutation_prob_mutation, mutate_100, mutate_sge_like, mutate_not_aware
from sge.operators.selection import tournament
from sge.operators.update import independent_update
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)

def generate_random_individual(max_expansions):
    genotype = [[] for _ in grammar.get_non_terminals()]
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : None}


def make_initial_population(pop_size):
    count = grammar.get_count_references_to_non_terminals()
    for i in range(pop_size):
        yield generate_random_individual(count)


def evaluate(ind, eval_func):
    mapping_values = [0 for _ in ind['genotype']]
    phen_tokens, tree_depth, gram_counter = grammar.mapping(grammar.get_pcfg(), ind['genotype'], mapping_values)
    phen = "".join(phen_tokens)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen_tokens
    ind['phenotype_string'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth
    ind['grammar_counter'] = gram_counter


def setup(parameters_file_path = None):
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
    grammar.read_grammar()


def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    setup(parameters_file_path=parameters_file)
    population = list(make_initial_population(params['POPSIZE']))
    flag = False    # alternate False - best overall
    best = None
    it = 0
    previous_population = population
    while it <= params['GENERATIONS']:        
        for i in tqdm(population):
            if i['fitness'] is None:
                evaluate(i, evaluation_function)

        population.sort(key=lambda x: x['fitness'])

        logger.evolution_progress(it, population, population[0], population[params['ELITISM']], grammar.get_pcfg(),previous_population)

      
        new_population = population[:params['ELITISM']]
        while len(new_population) < params['POPSIZE']:
            if np.random.uniform() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            
            ni = mutate_not_aware(ni, params['PROB_MUTATION'])    # mutate not aware of any depth - random codon
            # ni = mutate_100(ni, params['PROB_MUTATION'])    # mutate effective
            # ni = mutate_sge_like(ni, params['PROB_MUTATION'])    # mutate flawed like sge
            # ni = mutate_not_aware(ni, params['PROB_MUTATION'])    # mutate not aware
            new_population.append(ni)

        previous_population = copy.deepcopy(population)
        population = new_population
        it += 1
        # input()
