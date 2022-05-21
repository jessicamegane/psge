import random
import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
import copy
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate
from sge.utilities import ordered_set
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)


def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth, 'grammar': grammar.get_dict()}


def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()


def evaluate(ind, eval_func):
    mapping_values = [0 for i in ind['genotype']]
    if params['METHOD'] == 2:
        gram = ind['grammar']
    else:
        gram = grammar.get_dict()

    phen, tree_depth = grammar.mapping(ind['genotype'], gram, mapping_values)
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
    if params['METHOD'] == 1:
        params['EXPERIMENT_NAME'] += "/" + str(params['LEARNING_FACTOR'] * 100)
    elif params['METHOD'] == 2:
        params['EXPERIMENT_NAME'] += "/" + str(params['PROB_MUTATION_GRAMMAR'] * 100) + "/" + str(params['NORMAL_DIST_SD'])
    if params['DELAY'] > 0:
        params['EXPERIMENT_NAME'] += "/" + str(params['DELAY'])

    logger.prepare_dumps()
    random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])

def get_grammar_counter(genotype):
    gram_counter = {}
    for nt in grammar.get_dict().keys():
        expansion_list = genotype[grammar.get_non_terminals().index(nt)]
        counter = [0] * len(grammar.get_dict()[nt])
        for prod, _ in expansion_list:
            counter[prod] += 1
        gram_counter[nt] = counter

    return gram_counter

def update_probs(best, lf):
    gram_counter = get_grammar_counter(best['genotype'])
    for key, val in gram_counter.items():
        total = sum(val)
        if total > 0:
            l = [0] * len(val)
            for pos in range(len(val)):
                counter = val[pos]
                old_prob = grammar.get_dict()[key][pos][1]
                prob_updated = old_prob
                if counter > 0:
                    prob_updated = round(min(old_prob + lf * counter / total, 1.0),14)
                elif counter == 0:
                    prob_updated = round(max(old_prob - lf * prob_updated, 0.0),14)
                else:
                    print("ERROR - COUNTER < 0")
                    input()
                l[pos] = prob_updated

            # probabilities adjustment
            while(round(sum(l),3) != 1.0):
                if sum(l) > 1.0:
                    res = sum(l) - 1.0
                    diff = res / len(l)
                    for i in range(len(l)):
                        new = round(l[i] - diff,14)
                        l[i] = max(new,0.0)
                elif sum(l) < 1.0 and sum(l) > 0.0:
                    res = 1.0 - sum(l)
                    diff = res / len(l)
                    for i in range(len(l)):
                        new = round(l[i] + diff,14)
                        l[i] = min(diff,1.0)
                elif sum(l) < 0.0:
                    print("SHIT")
                    print(l)
                    input()

            for i in range(len(l)):
                grammar.get_dict()[key][i][1] = l[i]
                # gram.rulesNTerminal(key)[i][1] = l[i]
            # print(gram.rulesNTerminal(key))
    # print("Updated rules")
    # print(grammar.get_dict)

def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    setup(parameters_file_path=parameters_file)
    population = list(make_initial_population())
    flag = False    # alternar False - best overall
    best = None
    it = 0
    lf = params['LEARNING_FACTOR']
    flag_lf = True
    for i in tqdm(population):
        if i['fitness'] is None:
            evaluate(i, evaluation_function)
    while it <= params['GENERATIONS']:        
        population.sort(key=lambda x: x['fitness'])
        if params['METHOD'] == 1:
            if not best:
                best = copy.deepcopy(population[0])
            elif population[0]['fitness'] <= best['fitness']:
                # best overall
                best = copy.deepcopy(population[0])
        else:
            best = copy.deepcopy(population[0])

        if flag_lf:
            if not flag:
                update_probs(best, 1)
            else:
                update_probs(best_gen, 1)
            flag_lf = False
        else:
            if not flag:
                update_probs(best, params['LEARNING_FACTOR'])
            else:
                update_probs(best_gen, params['LEARNING_FACTOR'])
        if params['ADAPTIVE']:
            params['LEARNING_FACTOR'] += params['ADAPTIVE_INCREMENT']
        flag = not flag

     
        logger.evolution_progress(it, population, best, grammar.get_dict())

        new_population = []
        while len(new_population) < params['POPSIZE'] - params['ELITISM']:
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            ni = mutate(ni, params['PROB_MUTATION'])
            new_population.append(ni)

        for i in tqdm(new_population):
            evaluate(i, evaluation_function)
        new_population.sort(key=lambda x: x['fitness'])
        best_gen = copy.deepcopy(new_population[0])


        for i in tqdm(population[:params['ELITISM']]):
            evaluate(i, evaluation_function)
        new_population += population[:params['ELITISM']]


        population = new_population
        it += 1

