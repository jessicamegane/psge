import sge.grammar as grammar
import copy
import numpy as np


def get_grammar_counter(individual):
    """
    Function that counts how many times each production rule was expanded by the provided individual
    """
    # TODO: fix this function with a stop when there is no extra mapping value - pode haver posicoes no genotipo 
    # que ja nao saop usadas no mapeamento, e por isso nao deviam ser usadas para atyualizar as probabilidades
    gram_counter = []
    for nt in grammar.get_dict().keys():
        expansion_list = individual['genotype'][grammar.get_non_terminals().index(nt)]
        counter = [0] * len(grammar.get_dict()[nt])
        for prod, _, _ in expansion_list:
            counter[prod] += 1
        gram_counter.append(counter)

    return gram_counter

def independent_update(best, lf):
    """
    Update mechanism used in the PSGE paper.
    """
    gram_counter = get_grammar_counter(best)
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