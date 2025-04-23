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
from sge.operators.update import independent_update, dependent_update, subtree_dependent_update, longest_common_subtree, subtree_independent_update, subtree_parent_update
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)


def generate_random_individual():
    genotype = [[] for _ in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    if params['ADAPTIVE_MUTATION']:
        return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth, 'mutation_probs': [params['PROB_MUTATION'] for _ in genotype] }
    else:
        return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth, 'tree': None}


def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()


def evaluate(ind, eval_func):
    mapping_values = [0 for _ in ind['genotype']]
    phen, tree_depth, subtree_counter, tree = grammar.mapping(ind['genotype'], mapping_values)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen
    ind['fitness'] = quality
    ind['tree'] = tree
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth
    ind['subtree_counter'] = subtree_counter


def setup(parameters_file_path = None):
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    params['EXPERIMENT_NAME'] += "/" + str(params['LEARNING_FACTOR'] * 100)
    logger.prepare_dumps()
    np.random.seed(int(params['SEED']))
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar(params['PROBS_UPDATE'], params['LEVELS_UP'], params['LEVELS_DOWN'])


def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    setup(parameters_file_path=parameters_file)
    population = list(make_initial_population())
    flag = False    # alternate False - best overall
    best = None
    it = 0
    for i in tqdm(population):
        if i['fitness'] is None:
            evaluate(i, evaluation_function)
    while it <= params['GENERATIONS']:        

        population.sort(key=lambda x: x['fitness'])
        # best individual overall
        if not best:
            best = copy.deepcopy(population[0])
        elif population[0]['fitness'] <= best['fitness']:
            best = copy.deepcopy(population[0])
     
        if params['PROBS_UPDATE'] == 'standard':
            # independent_update_nbest(population, params['N_BEST'], params['LEARNING_FACTOR'])
            if not flag:
                independent_update(best, params['LEARNING_FACTOR'])
            else:
                independent_update(best_gen, params['LEARNING_FACTOR'])
            flag = not flag

            if params['ADAPTIVE_LF']:
                params['LEARNING_FACTOR'] += params['ADAPTIVE_INCREMENT']
        elif params['PROBS_UPDATE'] == 'dependent':
            dependent_update(population, params['LEARNING_FACTOR'], params['N_BEST'])
        elif params['PROBS_UPDATE'] == 'subtree_dependent':
            subtree_dependent_update(population, params['LEARNING_FACTOR'], params['N_BEST'])
        elif params['PROBS_UPDATE'] == 'common_subtree':
            # only previous rule
            # if not flag:
            #     size, subtree = longest_common_subtree(best['tree'], best['tree'])
            #     subtree_parent_update(subtree, params['LEARNING_FACTOR'],worst=True)
            # else:
            #     size, subtree = longest_common_subtree(best_gen['tree'], best_gen['tree'])
            #     subtree_parent_update(subtree, params['LEARNING_FACTOR'],worst=True)
            # flag = not flag

            b1 = best['tree']
            # print("best 1")
            # print(best['phenotype'])

            for i in range(1,10):
                if best['phenotype'] != population[i]['phenotype']:
                    b2 = population[i]['tree']
                    break

            # print("best 2")
            # print(population[i]['phenotype'])
            # print("common subtree")
            size, subtree = longest_common_subtree(b1, b2)
            # print(subtree)
            # print(size)
            # print(subtree)
            # input()

            # subtree_independent_update(subtree, params['LEARNING_FACTOR'])
            subtree_parent_update(subtree, params['LEARNING_FACTOR'])

            # get worst:
            worst = population[-1]['tree']

            for i in range(2,11):
                if population[-1]['phenotype'] != population[-i]['phenotype']:
                    b2 = population[-i]['tree']
                    break

            size, subtree = longest_common_subtree(worst, b2)
            
            subtree_parent_update(subtree, params['LEARNING_FACTOR'],worst=True)



     
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
        hash_counter = grammar.get_hash_counter()
        # print(hash_counter)
        # print(len(hash_counter))
        # input()
    # print(grammar.get_pcfg())
    # print(hash_counter)
    # print(len(hash_counter))
    # input()

