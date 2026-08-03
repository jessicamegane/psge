import sge.grammar as grammar
import copy
import numpy as np
from sge.parameters import LearningStrategy

'''
    MAIN UPDATE FUNCTION
'''

def update_distributions(learning_strategy, population, lf, n_best):
    if learning_strategy == LearningStrategy.INDEPENDENT:
        print("PSGE UPDATE")
        independent_update(population, lf, n_best)
    elif learning_strategy == LearningStrategy.DEPTH_BASED:
        print("DEPTH_BASED UPDATE")
        depth_based_update(population, lf, n_best)
    
'''
    INDEPENDENT UPDATE
    1. Get the count of how many times each production rule was used in the best individuals.
    2. For each production rule, if it was used, increase or decreaseits probability 
'''
def get_grammar_counter(individuals):
    counters = [ind['grammar_counter'] for ind in individuals]
    return [list(map(int, np.sum([c[i] for c in counters], axis=0)))
            for i in range(len(grammar.get_non_terminals()))]

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

        masked_indices = mask[i,:]
        gram[i, masked_indices] = np.clip(gram[i, masked_indices], 0, np.inf)
        gram[i, masked_indices] /= np.sum(gram[i, masked_indices])

'''
    DEPTH-BASED UPDATE
    1. Get the count of how many times each production rule was used in the best individuals, at each depth level.
    2. For each production rule, if it was used, increase or decrease its probability, but only for the depth level where it was used.
'''

def get_counter_individual_expansions(individuals):
    """Aggregate depth-based production counts across several individuals.

    An individual only records depths at which it expanded a given
    non-terminal.  Consequently, different individuals need not have the same
    depth keys.  Missing entries represent zero uses, rather than an error.
    """
    counters = [ind['grammar_counter'] for ind in individuals]
    aggregated = []

    for nt_i in range(len(grammar.get_non_terminals())):
        depths = set()
        for counter in counters:
            depths.update(counter[nt_i].keys())

        depth_counts = {}
        for depth in depths:
            observed = [counter[nt_i][depth]
                        for counter in counters
                        if depth in counter[nt_i]]
            depth_counts[depth] = list(map(int, np.sum(observed, axis=0)))
        aggregated.append(depth_counts)

    return aggregated



def depth_based_update(population, lf, n_best):
    gram = grammar.get_pcfg()
    nt_rows, depth_columns = gram.shape
    counter = get_counter_individual_expansions(population[:n_best])
    # print("counter\n",counter)
    # counter_bad = get_counter_individual_expansions(population[-n_best:])
    # print("counter bad \n", counter_bad)
    # input()
    # print(gram)
    # print(counter)
    p_mutation = 0.01
    amplitude_mutation = 0.05
    for nt_i in range(nt_rows):
        for depth_i in range(depth_columns):
            flag = False
            if len(gram[nt_i][depth_i]) <= 1:
                continue
            if counter[nt_i] != 0:
                if depth_i in counter[nt_i]:
                    if len(counter[nt_i][depth_i]) != len(gram[nt_i][depth_i]):
                        print("something bad happened!")
                        input()
                    for prod_i in range(len(counter[nt_i][depth_i])):
                        if counter[nt_i][depth_i][prod_i] > 0:
                            flag=True
                            # primeira versao
                            gram[nt_i][depth_i][prod_i] = gram[nt_i][depth_i][prod_i] + counter[nt_i][depth_i][prod_i] * lf / sum(counter[nt_i][depth_i])
                            # segunda versao e terceira
                            # gram[nt_i][depth_i][prod_i] = gram[nt_i][depth_i][prod_i] * (1+lf)**counter[nt_i][depth_i][prod_i]
            # if counter_bad[nt_i] != 0:
            #     if depth_i in counter_bad[nt_i]:
            #         if len(counter_bad[nt_i][depth_i]) != len(gram[nt_i][depth_i]):
            #             print("something bad happened!")
            #             input()
            #         for prod_i in range(len(counter_bad[nt_i][depth_i])):
            #             if counter_bad[nt_i][depth_i][prod_i] > 0:
            #                 flag= True
            #                 # terceira versao
            #                 gram[nt_i][depth_i][prod_i] = gram[nt_i][depth_i][prod_i] / (1+lf)**counter_bad[nt_i][depth_i][prod_i]
                        # else:
                        # # segunda versao
                        #     gram[nt_i][depth_i][prod_i] = gram[nt_i][depth_i][prod_i] - gram[nt_i][depth_i][prod_i] * lf
                # mutation on the value "gauss_p_mutation"
            for i_prod in range(len(gram[nt_i][depth_i])):
                if np.random.uniform() < p_mutation:
                    # segunda nova versao 2nd version caderno escrito
                    # gram[nt_i][depth_i][i_prod] = np.random.normal(gram[nt_i][depth_i][i_prod],amplitude_mutation)
                    # mutation "mut_0.001_"
                    # terceira nova versao 3rd version caderno escrito
                    if np.random.uniform() < 0.50:
                        gram[nt_i][depth_i][i_prod] = gram[nt_i][depth_i][i_prod] / (1+lf)
                    else:
                        gram[nt_i][depth_i][i_prod] = gram[nt_i][depth_i][i_prod] * (1+lf)
            gram[nt_i][depth_i] = np.clip(gram[nt_i][depth_i], 0, np.inf) / np.sum(np.clip(gram[nt_i][depth_i], 0, np.inf))
            if round(np.sum(gram[nt_i][depth_i]),3) > 1:
                print(gram[nt_i][depth_i])
                print("error in clip")
                input()


'''
    CONTEXT_AWARE UPDATE
    1. Get the count of how many times each production rule was used in the best individuals, at each depth level and for each parent non-terminal.
    2. For each production rule, if it was used, increase or decrease its probability, but only for the depth level and parent non-terminal where it was used.
'''


'''
    PROBABILITIES MUTATION
    Approach based on the Co-PSGE algorithm
    1. For each production rule, with a certain probability, mutate its probability using a normal distribution
    2. Normalize the probabilities of the production rules for each non-terminal 
'''

# def grammar_mutation(ind, prob_mutation, gauss_std):
#     print("COPSGE UPDATE")
#     ind['fitness'] = None
#     for idx_nt, nt_values in enumerate(ind['pcfg']):
#         if len(nt_values)  <= 1:
#             continue
#         for prob_idx in range(len(nt_values)):
#             if np.random.uniform() < prob_mutation:
#                 gauss = np.random.normal(0.0, gauss_std)
#                 diff = (gauss / (len(nt_values) - 1))
#                 nt_values[prob_idx] += (gauss + diff)
#                 nt_values -= diff
#                 nt_values = np.clip(nt_values, 0, np.inf)
#                 nt_values /= np.sum(np.clip(nt_values, 0, np.inf))
#                 ind['pcfg'][idx_nt] = nt_values
#                 break
#     return ind

def grammar_mutation(ind, prob_mutation, gauss_std):
    ind['fitness'] = None
    gram = ind['pcfg']
    rows, columns = gram.shape
    mask = copy.deepcopy(grammar.get_mask())
    for i in range(rows):
        if np.count_nonzero(mask[i,:]) <= 1:
            continue

        for j in range(columns):
            if not mask[i,j]:
                continue
            if np.random.uniform() < prob_mutation:
                gauss = np.random.normal(0.0, gauss_std)
                diff = (gauss / (columns - 1))
                gram[i][j] += (gauss + diff)
                gram[i] -= diff
                gram[i] = np.clip(gram[i], 0, np.inf)
                gram[i] /= np.sum(np.clip(gram[i], 0, np.inf))
                break

        masked_indices = mask[i,:]
        gram[i, masked_indices] = np.clip(gram[i, masked_indices], 0, np.inf)
        gram[i, masked_indices] /= np.sum(gram[i, masked_indices])
    return ind
