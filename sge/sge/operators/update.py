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

def get_individual_number_expansions(pop, counter):
    """
    Generates list , each position corresponds to one non-terminal.
    Each position has a dictionary which key is the depth, and value is a list in which the 
    index corresponds to the rule index and value at that position is the number of times it was expanded 
    """
    # print(counter)
    # input()
    for ind in pop:
        # TODO: instead of adding always, add only once for each individual in each rule etc
        genotype = ind['genotype']
        for nt_i, nt in enumerate(genotype):
            aux_dic = {}
            for index_rule, _, depth in nt:
                if counter[nt_i] == 0:
                    counter[nt_i] = {}
                if depth not in counter[nt_i]:
                    counter[nt_i][depth] = [0] * len(grammar.get_pcfg()[nt_i][0])
                if depth not in aux_dic:
                    aux_dic[depth] = []

                if index_rule not in aux_dic[depth]:
                    counter[nt_i][depth][index_rule] += 1
                    aux_dic[depth].append(index_rule)
            # counter[nt_i] = counter[nt_i]
    return counter

def dependent_update(population, lf, n_best=10):
    gram = grammar.get_pcfg()
    rows, columns = gram.shape
    counter = get_individual_number_expansions(population[:n_best], [0] * len(grammar.get_non_terminals()))
    # print("counter\n",counter)
    counter_bad = get_individual_number_expansions(population[-n_best:], [0] * len(grammar.get_non_terminals()))
    # print("counter bad \n", counter_bad)
    # input()
    p_mutation = 0.001
    amplitude_mutation = 0.05
    for nt_i in range(rows):
        for depth_i in range(columns):
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
            # for i_prod in range(len(gram[nt_i][depth_i])):
            #     if np.random.uniform() < p_mutation:
            #         gram[nt_i][depth_i][i_prod] = np.random.normal(gram[nt_i][depth_i][i_prod],amplitude_mutation)
                    # mutation "mut_0.001_"
                    # if np.random.uniform() < 0.50:
                    #     gram[nt_i][depth_i][i_prod] = gram[nt_i][depth_i][i_prod] / (1+lf)
                    # else:
                    #     gram[nt_i][depth_i][i_prod] = gram[nt_i][depth_i][i_prod] * (1+lf)
            gram[nt_i][depth_i] = np.clip(gram[nt_i][depth_i], 0, np.infty) / np.sum(np.clip(gram[nt_i][depth_i], 0, np.infty))
            if round(np.sum(gram[nt_i][depth_i]),3) > 1:
                print(gram[nt_i][depth_i])
                print("error in clip")
                input()
