import sge.grammar as grammar
import copy
import numpy as np


def get_grammar_counter(individuals):
    """
    Count how many times each production rule was expanded across individuals.
    """
    nts = grammar.get_non_terminals()
    rules = grammar.get_dict()

    # one counter list per NT
    gram_counter = [[0] * len(rules[nt]) for nt in nts]

    for individual in individuals:
        genotype = individual["genotype"]
        for i, nt in enumerate(nts):
            counter = gram_counter[i]
            for prod, _, _ in genotype[i]:
                counter[prod] += 1

    return gram_counter

def independent_update(individuals, lf, n_best):
    """
    Update mechanism used in the PSGE paper.
    """
    gram_counter = get_grammar_counter(individuals[:n_best])
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

        gram[i,:] = np.clip(gram[i,:], 0, np.inf) / np.sum(np.clip(gram[i,:], 0, np.inf))
    # update non_recursive options
    # grammar.compute_non_recursive_options()