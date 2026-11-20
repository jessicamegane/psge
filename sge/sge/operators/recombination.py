import copy
import numpy
import sge.grammar as grammar
from sge.parameters import params, AlgorithmMethod

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
    
    # compute nem individual
    if params['ALGORITHM_METHOD'] == AlgorithmMethod.COPSGE or params['ALGORITHM_METHOD'] == AlgorithmMethod.PSGE_COPSGE:
        gram = copy.deepcopy(p1['pcfg'] if p1['fitness'] < p2['fitness'] else p2['pcfg'])
        _, tree_depth, gram_counter = grammar.mapping(gram, genotype, mapping_values)
        offspring = {'genotype': genotype, 'fitness': None, 'mapping_values': mapping_values, 'tree_depth': tree_depth, 'grammar_counter': gram_counter, 'pcfg': gram}
    else:
        _, tree_depth, gram_counter = grammar.mapping(grammar.get_pcfg(), genotype, mapping_values)
        offspring = {'genotype': genotype, 'fitness': None, 'mapping_values': mapping_values, 'tree_depth': tree_depth, 'grammar_counter': gram_counter}

    if params['ADAPTIVE_MUTATION']:
        offspring['mutation_probs'] = mutation_probs
  
    return offspring
