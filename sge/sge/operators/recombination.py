import numpy
import sge.grammar as grammar
from sge.parameters import params

def crossover(p1, p2):
    xover_p_value = 0.5
    gen_size = len(p1['genotype'])
    mask = [numpy.random.uniform() for i in range(gen_size)]
    genotype = []
    mutation_prob = []
    print(p1['mutation_probs'])
    for index, prob in enumerate(mask):
        if prob < xover_p_value:
            genotype.append(p1['genotype'][index][:])
            mutation_prob.append(p1['mutation_probs'][index])
        else:
            genotype.append(p2['genotype'][index][:])
            mutation_prob.append(p2['mutation_probs'][index])
    mapping_values = [0] * gen_size

    # compute nem individual
    _, tree_depth = grammar.mapping(genotype, mapping_values)
    return {'genotype': genotype, 'fitness': None, 'mapping_values': mapping_values, 'tree_depth': tree_depth, 'mutation_probs': mutation_prob}