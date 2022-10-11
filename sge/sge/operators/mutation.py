import copy
import numpy as np
import sge.grammar as grammar

def mutate(p, pmutation):
    p = copy.deepcopy(p)
    p['fitness'] = None
    size_of_genes = grammar.count_number_of_options_in_production()
    mutable_genes = [index for index, nt in enumerate(grammar.get_non_terminals()) if size_of_genes[nt] != 1 and len(p['genotype'][index]) > 0]
    for at_gene in mutable_genes:
        nt = list(grammar.get_non_terminals())[at_gene]
        temp = p['mapping_values']
        mapped = temp[at_gene]
        for position_to_mutate in range(0, mapped):
            if np.random.uniform() < pmutation:
                current_value = p['genotype'][at_gene][position_to_mutate]
                # gaussian mutation
                codon = np.random.normal(current_value[1], 0.5)
                codon = min(codon,1.0)
                codon = max(codon,0.0)
                if p['tree_depth'] >= grammar.get_max_depth():
                    non_recursive_prods, prob_non_recursive = grammar.get_non_recursive_options(nt)    
                    prob_aux = 0.0
                    for index, option in non_recursive_prods:
                        if prob_non_recursive == 0.0:
                            new_prob = 1.0 / len(non_recursive_prods)
                        else:
                            new_prob = (grammar.get_pcfg()[grammar.get_index_of_non_terminal()[nt],index] * 1.0) / prob_non_recursive
                        prob_aux += new_prob

                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index, option in enumerate(grammar.get_dict()[nt]):
                        prob_aux += grammar.get_pcfg()[grammar.get_index_of_non_terminal()[nt],index]
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                  
                p['genotype'][at_gene][position_to_mutate] = [expansion_possibility, codon]
    return p



def mutate_level(p):
    p = copy.deepcopy(p)
    p['fitness'] = None
    pmutation = p['mutation_prob']
    size_of_genes = grammar.count_number_of_options_in_production()
    mutable_genes = [index for index, nt in enumerate(grammar.get_non_terminals()) if size_of_genes[nt] != 1 and len(p['genotype'][index]) > 0]
    for at_gene in mutable_genes:
        nt = list(grammar.get_non_terminals())[at_gene]
        temp = p['mapping_values']
        mapped = temp[at_gene]
        for position_to_mutate in range(0, mapped):
            if np.random.uniform() < pmutation[at_gene]:
                current_value = p['genotype'][at_gene][position_to_mutate]
                # codon = random.random()
                # gaussian mutation
                codon = np.random.normal(current_value[1], 0.5)
                codon = min(codon,1.0)
                codon = max(codon,0.0)
                expansion_possibility = 0
                if p['tree_depth'] >= grammar.get_max_depth():
                    non_recursive_prods, prob_non_recursive = grammar.get_non_recursive_options(nt)    
                    prob_aux = 0.0
                    for index, option in non_recursive_prods:
                        if prob_non_recursive == 0.0:
                            new_prob = 1.0 / len(non_recursive_prods)
                        else:
                            new_prob = (grammar.get_pcfg()[grammar.get_index_of_non_terminal()[nt],index] * 1.0) / prob_non_recursive
                        prob_aux += new_prob

                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index, option in enumerate(grammar.get_dict()[nt]):
                        prob_aux += grammar.get_pcfg()[grammar.get_index_of_non_terminal()[nt],index]
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                  
                p['genotype'][at_gene][position_to_mutate] = [expansion_possibility, codon]
    return p