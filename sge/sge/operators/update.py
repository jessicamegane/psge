import sge.grammar as grammar
import copy
import numpy as np

def update_based_on_counter(gram_counter, lf):
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
    update_based_on_counter(gram_counter,lf)
    

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

def dependent_update(population, lf, n_best):
    gram = grammar.get_pcfg()
    rows, columns = gram.shape
    counter = get_individual_number_expansions(population[:n_best], [0] * len(grammar.get_non_terminals()))
    # print("counter\n",counter)
    counter_bad = get_individual_number_expansions(population[-n_best:], [0] * len(grammar.get_non_terminals()))
    # print("counter bad \n", counter_bad)
    # input()
    p_mutation = 0.01
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

def subtree_dependent_update(population, lf, n_best):
    gram = grammar.get_pcfg()
    counter = {}
    for ind in population[:n_best]:
        d = ind['subtree_counter']
        for key, value in d.items():
            counter[key] = counter.get(key, 0) + value

    for key, value in counter.items():
        if value <= 1:
            continue
        (hsh, symbol, expansion) = key
        if hsh not in gram[symbol]:
            # TODO: mudar expressao
            number_prods = len(gram[symbol][None])
            prob = 1.0 / number_prods
            gram[symbol][hsh] = np.full(number_prods, prob)

        gram[symbol][hsh][expansion] += value * lf
        # softmax
        gram[symbol][hsh] = np.clip(gram[symbol][hsh], 0, np.inf) / np.sum(np.clip(gram[symbol][hsh], 0, np.inf))


def subtree_grammar_counter_parent(root, grammar_counter):
    if root.symbol[1] == 'T':
        return

    index = grammar.ordered_non_terminals.index(root.symbol[0])
    if root.children:
        if root.parent == None or root.parent.parent == None:
            grammar_counter[index][None][root.index_children] += 1
        else:
            hsh = ''
            parent_siblings = root.parent.parent.children
            for child in parent_siblings:
                hsh += child.symbol[0]
            if hsh not in grammar_counter[index]:
                grammar_counter[index][hsh] = [0] * len(grammar.get_dict()[root.symbol[0]])

                grammar_counter[index][hsh][root.index_children]+= 1

        
        for child in root.children:
            subtree_grammar_counter_parent(child, grammar_counter)


def subtree_parent_update(subtree, lf, worst=False):
    gram = grammar.get_pcfg()
    # print(subtree)
    # print(gram)
    grammar_counter = []
    for nt in grammar.get_dict().keys():
        dic = {None: [0] * len(grammar.get_dict()[nt])}
        grammar_counter.append(dic)


    subtree_grammar_counter_parent(subtree, grammar_counter)
    # print(grammar_counter)

    i = 0
    for nt in gram:
        if len(nt[None]) <= 1:
            i += 1
            continue
        
        # print(nt)
        # input()

        for hsh, count in grammar_counter[i].items():
            lff = np.full(len(count), lf)
            if hsh in gram:
                if worst:
                    nt[hsh] -= count * lff
                else:
                    nt[hsh] += count * lff
                # normalize
                # nt[hsh] = np.clip(nt[hsh], 0, np.inf) / np.sum(np.clip(nt[hsh], 0, np.inf))
                # softmax
                e_x = np.exp(nt[hsh] - np.max(nt[hsh]))
                nt[hsh] = e_x / e_x.sum()
            else:
                # print("--")
                # print(hsh)
                # print(count)
                # print(nt[None])
                nr_prods = len(nt[None])
                prob = 1.0 / nr_prods
                # print(prob)
                nt[hsh] = np.full(nr_prods, prob)
                # print(nt[hsh])

                if worst:
                    nt[hsh] -= count * lff
                else:
                    nt[hsh] += count * lff
                # normalize
                # nt[hsh] = np.clip(nt[hsh], 0, np.inf) / np.sum(np.clip(nt[hsh], 0, np.inf))
                # softmax
                e_x = np.exp(nt[hsh] - np.max(nt[hsh]))
                nt[hsh] = e_x / e_x.sum()
                # print(nt)
                # print("update")
    
        i += 1
    # print(nt)
    # input()
    # print("updated grammar")
    print(gram)



def longest_common_subtree_in_progress(node1, node2, n1_level=0,n2_level=0):
    # If one node is None, no match can be made
    if not node1 or not node2:
        return 0, None

    print(f"Node1:{node1.symbol} Lvl={n1_level} \t Node2: {node2.symbol} Lvl={n2_level}")
    # If values match, continue with standard comparison
    if node1.symbol == node2.symbol:
        matches = []
        #for i in range(max(len(node1.children), len(node2.children))):
        for i in range(len(node1.children)):
            local_matches = []
            max_match = 0
            max_subtree = None

            for j in range(len(node2.children)):

                child1 = node1.children[i] #if i < len(node1.children) else None
                child2 = node2.children[j] #if i < len(node2.children) else None
                
                match, subtree = longest_common_subtree(child1, child2, n1_level+1, n2_level+1)
                
                if child1.symbol == child2.symbol:
                    
                    if match:
                        matches.append((match, subtree))
        
        if not matches:  # no children match
            return 1, grammar.Node(node1.symbol, node1.index_children)
        
        # Here, we ensure all matching subtrees are included, not just one
        current_subtree = grammar.Node(node1.symbol, node1.index_children)
        current_subtree.children = [subtree for _, subtree in matches if subtree]
        return sum(match for match, _ in matches) + 1, current_subtree
    
    # If values don't match, check if node1 matches any child of node2 or vice versa
    max_match = 0
    max_subtree = None
    
    for child2 in node2.children:
        match, subtree = longest_common_subtree(node1, child2, n1_level, n2_level+1)
        if match > max_match:
            max_match, max_subtree = match, subtree
    
    for child1 in node1.children:
        match, subtree = longest_common_subtree(child1, node2, n1_level+1, n2_level)
        if match > max_match:
            max_match, max_subtree = match, subtree
    
    return max_match, max_subtree



def longest_common_subtree(root1, root2):
    subtrees = {}
    max_size = [0]  # Using list to allow modification in nested functions
    result = [None]
    
    def serialize_and_count(node):
        if not node:
            return "#"
        
        # Serialize current node and all children
        serialized_children = [serialize_and_count(child) for child in node.children]
        serialized_children.sort()  # Order doesn't matter for equality
        
        # Create serialization string
        serialization = f"({node.symbol},[{','.join(serialized_children)}])"
        
        # Count nodes in current subtree
        size = 1 + sum(count_nodes(child) for child in node.children)
        
        # Update subtree dictionary
        if serialization not in subtrees:
            subtrees[serialization] = {"count": 0, "size": size, "node": node}
        subtrees[serialization]["count"] += 1
        
        # Update largest common subtree if this one appears in both trees
        if subtrees[serialization]["count"] == 2 and size > max_size[0]:
            max_size[0] = size
            result[0] = node
            
        return serialization
    
    def count_nodes(node):
        if not node:
            return 0
        return 1 + sum(count_nodes(child) for child in node.children)
    
    # Process both trees
    serialize_and_count(root1)
    serialize_and_count(root2)
    # print(result[0])
    return 0,result[0]

def clone_tree(node):
    if not node:
        return None
    new_node = grammar.Node(node.symbol)
    new_node.children = [clone_tree(child) for child in node.children]
    # print(new_node)
    return new_node
'''

def longest_common_subtree(node1, node2,  n1_level=0,n2_level=0):
      # If one node is None, no match can be made
        if not node1 or not node2:
            return 0, None
        print(f"Node1:{node1.symbol} Lvl={n1_level} \t Node2: {node2.symbol} Lvl={n2_level}")
        # If values match, continue with standard comparison
        if node1.symbol == node2.symbol:
            matches = []
            for i in range(max(len(node1.children), len(node2.children))):
                child1 = node1.children[i] if i < len(node1.children) else None
                child2 = node2.children[i] if i < len(node2.children) else None
                match, subtree = longest_common_subtree(child1, child2,n1_level+1, n2_level+1)
                if match:
                    matches.append((match, subtree))
            
            if not matches:  # no children match
                return 1, grammar.Node(node1.symbol)
            
            # Here, we ensure all matching subtrees are included, not just one
            current_subtree = grammar.Node(node1.symbol)
            current_subtree.children = [subtree for _, subtree in matches if subtree]
            return sum(match for match, _ in matches) + 1, current_subtree
        
        # If values don't match, check if node1 matches any child of node2 or vice versa
        max_match = 0
        max_subtree = None
        
        for child2 in node2.children:
            match, subtree = longest_common_subtree(node1, child2, n1_level, n2_level+1)
            if match > max_match:
                max_match, max_subtree = match, subtree
        
        for child1 in node1.children:
            match, subtree = longest_common_subtree(child1, node2, n1_level+1, n2_level)
            if match > max_match:
                max_match, max_subtree = match, subtree
        
        return max_match, max_subtree
'''

def subtree_grammar_counter(root, grammar_counter):
    print(root.symbol)
    if root.children:
        index = grammar.ordered_non_terminals.index(root.symbol[0])
        print(index)
        grammar_counter[index][root.index_children] += 1
        for child in root.children:
            subtree_grammar_counter(child, grammar_counter)

def subtree_independent_update(subtree, lf):

    gram = grammar.get_pcfg()
    # print("------")
    # print(gram)
    # print(subtree)
    grammar_counter = []
    for nt in grammar.get_dict().keys():
        counter = [0] * len(grammar.get_dict()[nt])
        grammar_counter.append(counter)
    # print(grammar_counter)
    subtree_grammar_counter(subtree, grammar_counter)
    # print("counter")
    # print(grammar_counter)
    # input()
    update_based_on_counter(grammar_counter, lf)
    


def conditional_update(population, lf, n_best):
    gram = grammar.get_pcfg()
    p_mutation = 0.0005
    for nt_i, nt_table in enumerate(gram):

        counter_nt_symb = population[0]['counter'][nt_i]
        for i in range(1, n_best):
            counter_nt_symb += population[i]['counter'][nt_i]
        # print("nt table")
        # print(nt_table)
    
        for previous_sym_i, previous_prod_table in enumerate(nt_table):
            if len(previous_prod_table) <= 2:
                continue
            # print("previous prod table")
            # print(previous_prod_table)
            for symb_i, prob in enumerate(previous_prod_table):
                old_prob = prob
                # print(counter_nt_symb)
                # print(symb_i)
                # print(previous_sym_i)
                counter = counter_nt_symb[previous_sym_i][symb_i]
                # counter_no_previous = counter_nt_symb[symb_i][-1]

                total = np.sum(counter_nt_symb[previous_sym_i])
                # print("beggining")
                # print(counter)
                # print(total)
                if counter > 0:
                    # print("update good")
                    # print(counter)
                    # print(old_prob)
                    gram[nt_i][previous_sym_i][symb_i] = min(old_prob + lf * counter / total, 1.0)
                    # print(gram[nt_i][previous_sym_i][symb_i])
                else:
                    # print("update bad")
                    # print(counter)
                    # print(old_prob)
                    gram[nt_i][previous_sym_i][symb_i] = max(old_prob - lf * old_prob, 0.0)
                    # print(gram[nt_i][previous_sym_i][symb_i])
                # FIXME: maybe here have a smaller lf
                # if np.random.uniform() < p_mutation:
                   
                #     if np.random.uniform() < 0.50:
                #         gram[nt_i][previous_sym_i][symb_i] = gram[nt_i][previous_sym_i][symb_i] / (1+lf)
                #     else:
                #         gram[nt_i][previous_sym_i][symb_i] = gram[nt_i][previous_sym_i][symb_i] * (1+lf)
            # print("gram updated q step")
            # print(gram[nt_i])
            # input()
            gram[nt_i][previous_sym_i] = np.clip(gram[nt_i][previous_sym_i], 0, np.infty) / np.sum(np.clip(gram[nt_i][previous_sym_i], 0, np.infty))

        # print("individual")
        # print(population[i]['phenotype'])
        # print(best)
        # print(gram)
        # input()