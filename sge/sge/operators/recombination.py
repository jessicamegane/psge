import numpy
import sge.grammar as grammar
from sge.parameters import params

def crossover(p1, p2):
    xover_p_value = 0.5
    gen_size = len(p1['genotype'])
    mask = [numpy.random.uniform() for i in range(gen_size)]
    genotype = []
    mutation_probs = []
    for index, prob in enumerate(mask):
        if prob < xover_p_value:
            genotype.append(p1['genotype'][index][:])
            if params['ADAPTIVE_MUTATION']:
                mutation_probs.append(p1['mutation_probs'][index])
        else:
            genotype.append(p2['genotype'][index][:])
            if params['ADAPTIVE_MUTATION']:
                mutation_probs.append(p2['mutation_probs'][index])
    mapping_values = [0] * gen_size
import random
import sge.grammar as grammar


def crossover(p1, p2):
    xover_p_value = 0.5
    gen_size = len(p1['genotype'])
    mask = [random.random() for i in range(gen_size)]
    genotype = []
    for index, prob in enumerate(mask):
        if prob < xover_p_value:
            genotype.append(p1['genotype'][index][:])
        else:
            genotype.append(p2['genotype'][index][:])
    mapping_values = [0] * gen_size
    # compute nem individual
    # TODO: choose probabilities from the highest fitness parent
    if p1['fitness'] < p2['fitness']:
        probs = p1['probabilities']
    else:
        probs = p2['probabilities']
        
    _, tree_depth = grammar.mapping(probs, genotype, mapping_values)
    return {'genotype': genotype, 'probabilities': probs, 'fitness': None, 'mapping_values': mapping_values, 'tree_depth': tree_depth}
