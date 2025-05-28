import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
import copy
import numpy as np
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate, mutate_level, mutation_prob_mutation
from sge.operators.selection import tournament
from sge.operators.update import independent_update
from sge.operators.autopsge import initialize_vae, update_probs
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)


def generate_random_individual(max_expansions):
    if params['PROBS_UPDATE'] == 'autoPSGE':
        # TODO: implement multiple values of list
        # mapping_values = [0 for _ in ind['genotype']]
        genotype = [[[-1, np.random.uniform(0, 1), -1] for _ in range(max_expansions[nt])] for nt in grammar.get_non_terminals()]
        # tree_depth = grammar.mapping_rules(genotype, mapping_values)
    else:
        genotype = [[] for _ in grammar.get_non_terminals()]
        tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    if params['ADAPTIVE_MUTATION']:
        return {'genotype': genotype, 'fitness': None, 'tree_depth' : None, 'mutation_probs': [params['PROB_MUTATION'] for _ in genotype] }
    else:
        return {'genotype': genotype, 'fitness': None, 'tree_depth' : None, 'probabilities': copy.deepcopy(grammar.get_pcfg())}


def make_initial_population():
    # max_expansions = grammar.get_max_expansions()
    count = grammar.get_count_references_to_non_terminals()
    # print(max_expansions)
    # print(count)
    for i in range(params['POPSIZE']):
        yield generate_random_individual(count)


def evaluate(ind, eval_func):
    mapping_values = [0 for _ in ind['genotype']]
    # print(ind['genotype'])
    # print(mapping_values)
    # input()
    phen, tree_depth = grammar.mapping(ind['probabilities'], ind['genotype'], mapping_values)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth


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
    population = list(make_initial_population())
    flag = False    # alternate False - best overall
    best = None
    it = 0
    for i in tqdm(population):
        if i['fitness'] is None:
            evaluate(i, evaluation_function)
    if params['PROBS_UPDATE'] == 'autoPSGE':
        vae = initialize_vae(population, grammar.count_number_of_options_in_production())
    while it <= params['GENERATIONS']:        

        population.sort(key=lambda x: x['fitness'])
        if not best:
            best = copy.deepcopy(population[0])
        elif population[0]['fitness'] <= best['fitness']:
            best = copy.deepcopy(population[0])
            
        if params['PROBS_UPDATE'] == 'standard':
            # best individual overall
            if not best:
                best = copy.deepcopy(population[0])
            elif population[0]['fitness'] <= best['fitness']:
                best = copy.deepcopy(population[0])
        
            if not flag:
                independent_update(best, params['LEARNING_FACTOR'])
            else:
                independent_update(best_gen, params['LEARNING_FACTOR'])
            flag = not flag

            if params['ADAPTIVE_LF']:
                params['LEARNING_FACTOR'] += params['ADAPTIVE_INCREMENT']
        elif params['PROBS_UPDATE'] == 'autoPSGE':
            # print("before")
            # print(grammar.get_pcfg())
            # print('before update')
            vae, probs = update_probs(it, vae, population, grammar.count_number_of_options_in_production())
            # print(population[0]['probabilities'])
            # print("after")
            # print(probs)
            grammar.set_pcfg(probs)
     
        logger.evolution_progress(it, population, best, grammar.get_pcfg())

        new_population = []
        while len(new_population) < params['POPSIZE'] - params['ELITISM']:
            if np.random.uniform() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            if params['ADAPTIVE_MUTATION']:
                # if we want to use Adaptive Facilitated Mutation
                ni = mutation_prob_mutation(ni)
                ni = mutate_level(ni)
            else:
                ni = mutate(ni, params['PROB_MUTATION'])                
            new_population.append(ni)

        for i in tqdm(new_population):
            evaluate(i, evaluation_function)
        new_population.sort(key=lambda x: x['fitness'])
        # best individual from the current generation
        best_gen = copy.deepcopy(new_population[0])

        if params['REMAP']:
            for i in tqdm(population[:params['ELITISM']]):
                evaluate(i, evaluation_function)
        new_population += population[:params['ELITISM']]

        population = new_population
        it += 1

