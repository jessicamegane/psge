import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
import copy
import numpy as np
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate, mutate_level
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)


def generate_random_individual():
    genotype = [[] for _ in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth, 'mutation_prob': grammar.get_mutation_prob()}


def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()


def evaluate(ind, eval_func):
    mapping_values = [0 for _ in ind['genotype']]
    phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
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
        
    params['EXPERIMENT_NAME'] += "/" + "prob_mut_probs_" + str(params['PROB_MUTATION_PROBS']) +"/gauss_" + str(params['GAUSS_SD']) + "/delay_" + str(params['DELAY'])+ "/rempa_" + str(params['REMAP']) 
    logger.prepare_dumps()
    np.random.seed(int(params['SEED']))
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])

def get_grammar_counter(genotype):
    """
    Function that counts how many times each production rule was expanded by the best individual
    """
    gram_counter = []
    for nt in grammar.get_dict().keys():
        expansion_list = genotype[grammar.get_non_terminals().index(nt)]
        counter = [0] * len(grammar.get_dict()[nt])
        for prod, _ in expansion_list:
            counter[prod] += 1
        gram_counter.append(counter)

    return gram_counter

def update_probs(best, lf):
    gram_counter = get_grammar_counter(best['genotype'])
    gram = grammar.get_pcfg()
    rows, columns = gram.shape
    mask = copy.deepcopy(grammar.get_mask())
    for i in range(rows):
        if np.count_nonzero(mask[i,:]) <= 1:
            continue
        total = sum(gram_counter[i])

        for j in range(columns):
            if not mask[i,j]:
                continue
            counter = gram_counter[i][j]
            old_prob = gram[i][j]

            if counter > 0:
                gram[i][j] = min(old_prob + lf * counter / total, 1.0)
            elif counter == 0:
                gram[i][j] = max(old_prob - lf * old_prob, 0.0)

        gram[i,:] = np.clip(gram[i,:], 0, np.infty) / np.sum(np.clip(gram[i,:], 0, np.infty))
    # update non_recursive options
    # grammar.compute_non_recursive_options()



def mutation_prob_mutation(ind):
    gram = ind['mutation_prob']
    new_p = []
    for p in gram:
        if np.random.uniform() < params['PROB_MUTATION_PROBS']:
            gauss = np.random.normal(0.0,params['GAUSS_SD'])
            # TODO: no futuro criar bounds
            p = max(p+gauss,0)
            p = min(p,1)
        new_p.append(p)
    ind['mutation_prob'] = new_p
    return ind

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
     
        if not params['DELAY']:
            if not flag:
                update_probs(best, params['LEARNING_FACTOR'])
            else:
                update_probs(best_gen, params['LEARNING_FACTOR'])
            flag = not flag

        logger.evolution_progress(it, population, best, grammar.get_pcfg())

        new_population = []
        while len(new_population) < params['POPSIZE'] - params['ELITISM']:
            if np.random.uniform() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            if params['MUTATE_GRAMMAR']:
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

        if params['DELAY']:
            if not flag:
                update_probs(best, params['LEARNING_FACTOR'])
            else:
                update_probs(best_gen, params['LEARNING_FACTOR'])
            flag = not flag

        population = new_population
        it += 1

