import re
import copy
from sge.utilities import ordered_set
import json
import numpy as np
from sge.parameters import LearningStrategy, AlgorithmMethod

class Grammar:
    """Class that represents a grammar. It works with the prefix notation."""
    NT = "NT"
    T = "T"
    NT_PATTERN = "(<.+?>)"
    RULE_SEPARATOR = "::="
    PRODUCTION_SEPARATOR = "|"
    ROOT_CONTEXT = "__ROOT__"
    PREVIOUS_START_CONTEXT = "__START__"
    CONTEXT_STRATEGIES = {
        LearningStrategy.CONTEXT_AWARE,
        LearningStrategy.CONTEXT_AWARE_DEPTH,
        LearningStrategy.CONTEXT_AWARE_PREVIOUS,
    }

    def __init__(self):
        self.grammar_file = None
        self.grammar = {}
        self.productions_labels = {}
        self.non_terminals, self.terminals = set(), set()
        self.ordered_non_terminals = ordered_set.OrderedSet()
        self.non_recursive_options = {}
        self.number_of_options_by_non_terminal = None
        self.start_rule = None
        self.max_depth = None
        self.max_init_depth = None
        self.max_number_prod_rules = 0
        self.pcfg = None
        self.pcfg_mask = None
        self.pcfg_path = None
        self.index_of_non_terminal = {}
        self.shortest_path = {}

    def set_path(self, grammar_path):
        self.grammar_file = grammar_path

    def set_pcfg_path(self, pcfg_path):
        self.pcfg_path = pcfg_path

    def set_min_init_tree_depth(self, min_tree_depth):
        self.max_init_depth = min_tree_depth

    def set_max_tree_depth(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def get_max_depth(self):
        return self.max_depth
    
    def get_max_init_depth(self):
        return self.max_init_depth

    def read_grammar(self, learning_strategy=None, algorithm_method=None):
        """
        Reads a Grammar in the BNF format and converts it to a python dictionary
        This method was adapted from PonyGE version 0.1.3 by Erik Hemberg and James McDermott
        """
        if self.grammar_file is None:
            raise Exception("You need to specify the path of the grammar file")


        with open(self.grammar_file, "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip() != "":
                    if line.find(self.PRODUCTION_SEPARATOR):
                        left_side, productions = line.split(self.RULE_SEPARATOR)
                        left_side = left_side.strip()
                        if not re.search(self.NT_PATTERN, left_side):
                            raise ValueError("Left side not a non-terminal!")
                        self.non_terminals.add(left_side)
                        self.ordered_non_terminals.add(left_side)
                        # assumes that the first rule in the file is the axiom
                        if self.start_rule is None:
                            self.start_rule = (left_side, self.NT)
                        temp_productions = []
                        for production in [production.strip() for production in productions.split(self.PRODUCTION_SEPARATOR)]:
                            temp_production = []
                            if not re.search(self.NT_PATTERN, production):
                                if production == "None":
                                    production = ""
                                self.terminals.add(production)
                                temp_production.append((production, self.T))
                            else:
                                for value in re.findall("<.+?>|[^<>]*", production):
                                    if value != "":
                                        if re.search(self.NT_PATTERN, value) is None:
                                            sym = (value, self.T)
                                            self.terminals.add(value)
                                        else:
                                            sym = (value, self.NT)
                                        temp_production.append(sym)
                            temp_productions.append(temp_production)                          
                        self.max_number_prod_rules = max(self.max_number_prod_rules, len(temp_productions))
                        if left_side not in self.grammar:
                            self.grammar[left_side] = temp_productions
        
        self.learning_strategy = LearningStrategy.from_string(learning_strategy) if learning_strategy is not None else None
        self.algorithm_method = AlgorithmMethod.from_string(algorithm_method) if algorithm_method is not None else None
        if (self.learning_strategy in self.CONTEXT_STRATEGIES and
                self.algorithm_method in {AlgorithmMethod.COPSGE,
                                          AlgorithmMethod.PSGE_COPSGE}):
            raise ValueError(
                "%s is only supported with shared-grammar PSGE/SGEF methods"
                % self.learning_strategy.value
            )
        self.generate_uniform_pcfg()
        if self.pcfg_path is not None:
            with open(self.pcfg_path) as f:
                loaded_pcfg = json.load(f)
            self.set_pcfg(loaded_pcfg)
        # self.compute_non_recursive_options()
        self.find_shortest_path()
        self.number_of_references_by_non_terminal = self.calculate_max_expansions_recursive({}, self.start_rule, self.max_depth)


    def find_shortest_path(self):
        open_symbols = []
        for nt in self.grammar.keys():
            self.minimum_path_calc((nt, 'NT'), open_symbols)
            
    def minimum_path_calc(self, current_symbol, open_symbols):
        if current_symbol[1] == self.T:
            return 0
        else:
            open_symbols.append(current_symbol)
            for derivation_option in self.grammar[current_symbol[0]]:
                max_depth = 0
                if current_symbol not in self.shortest_path:
                    self.shortest_path[current_symbol] = [999999]
                if bool(sum([i in open_symbols for i in derivation_option])):
                    continue
                if current_symbol not in derivation_option:
                    for symbol in derivation_option:
                        depth = self.minimum_path_calc(symbol, open_symbols)
                        depth += 1
                        if depth > max_depth:
                            max_depth = depth

                    if max_depth < self.shortest_path[current_symbol][0]:
                        self.shortest_path[current_symbol] = [max_depth]
                        if derivation_option not in self.shortest_path[current_symbol]:
                            self.shortest_path[current_symbol].append(derivation_option)
                    if max_depth == self.shortest_path[current_symbol][0]:
                        if derivation_option not in self.shortest_path[current_symbol]:
                            self.shortest_path[current_symbol].append(derivation_option)
            open_symbols.remove(current_symbol)
            return self.shortest_path[current_symbol][0]
                    
            

    def create_counter(self):
        self.counter = dict.fromkeys(self.grammar.keys(),[])
        for k in self.counter.keys():
            self.counter[k] = [0] * len(self.grammar[k])

    def generate_uniform_pcfg(self):
        """
        assigns uniform probabilities to grammar
        """
        if self.learning_strategy in self.CONTEXT_STRATEGIES:
            self.pcfg = [{} for _ in self.grammar]
            for i, nt in enumerate(self.grammar):
                self.index_of_non_terminal.setdefault(nt, i)
            self.pcfg_mask = None
        elif self.learning_strategy == LearningStrategy.DEPTH_BASED:
            array = np.empty(shape=(len(self.grammar.keys()),(self.max_depth + 1)),dtype=object)

            for i, nt in enumerate(self.grammar):
                number_prods = len(self.grammar[nt])
                prob = 1.0 / number_prods
                for j in range(self.max_depth+1):
                    array[i, j] = np.full(number_prods, prob)
                if nt not in self.index_of_non_terminal:
                    self.index_of_non_terminal[nt] = i
            self.pcfg = array
        else:
            array = np.zeros(shape=(len(self.grammar.keys()),self.max_number_prod_rules))
            for i, nt in enumerate(self.grammar):
                number_probs = len(self.grammar[nt])
                prob = 1.0 / number_probs
                array[i,:number_probs] = prob
                if nt not in self.index_of_non_terminal:
                    self.index_of_non_terminal[nt] = i
            self.pcfg = array
            self.pcfg_mask = self.pcfg != 0

    def generate_random_pcfg(self):
        """
        assigns random probabilities to grammar and a softmax so they are uniform
        """
        pcfg = []
        for i, nt in enumerate(self.grammar):
            number_probs = len(self.grammar[nt])
            array = np.random.rand(number_probs)
            array = array / np.sum(array)
            if nt not in self.index_of_non_terminal:
                self.index_of_non_terminal[nt] = i
            pcfg.append(array)
        self.pcfg = pcfg

    def get_mask(self):
        return self.pcfg_mask

    def get_index_of_non_terminal(self):
        return self.index_of_non_terminal

    def get_non_terminals(self):
        return self.ordered_non_terminals

    def count_number_of_options_in_production(self):
        if self.number_of_options_by_non_terminal is None:
            self.number_of_options_by_non_terminal = []
            for nt in self.ordered_non_terminals:
                self.number_of_options_by_non_terminal.append(len(self.grammar[nt]))
        return self.number_of_options_by_non_terminal

    def list_non_recursive_productions(self, nt):
        non_recursive_elements = []
        for options in self.grammar[nt]:
            for option in options:
                if option[1] == self.NT and option[0] == nt:
                    break
            else:
                non_recursive_elements += [options]
        return non_recursive_elements
    
    def _uniform_probabilities(self, nt_index):
        nt = list(self.get_non_terminals())[nt_index]
        count = len(self.grammar[nt])
        return np.full(count, 1.0 / count)

    def _default_context(self, current_depth=None):
        if self.learning_strategy == LearningStrategy.CONTEXT_AWARE:
            return self.ROOT_CONTEXT
        if self.learning_strategy == LearningStrategy.CONTEXT_AWARE_DEPTH:
            return (self.ROOT_CONTEXT, 0 if current_depth is None else current_depth)
        if self.learning_strategy == LearningStrategy.CONTEXT_AWARE_PREVIOUS:
            return self.PREVIOUS_START_CONTEXT
        return None

    def get_context_probabilities(self, grammar, nt_index, context=None,
                                  create=True):
        """Get a sparse context distribution, initializing it uniformly."""
        if grammar is None:
            grammar = self.pcfg
        if context is None:
            context = self._default_context()
        table = grammar[nt_index]
        if self.learning_strategy == LearningStrategy.CONTEXT_AWARE_DEPTH:
            parent, depth = context
            parent_table = table.setdefault(parent, {}) if create else table.get(parent)
            if parent_table is None:
                return self._uniform_probabilities(nt_index)
            if depth not in parent_table:
                if not create:
                    return self._uniform_probabilities(nt_index)
                parent_table[depth] = self._uniform_probabilities(nt_index)
            return parent_table[depth]
        if context not in table:
            if not create:
                return self._uniform_probabilities(nt_index)
            table[context] = self._uniform_probabilities(nt_index)
        return table[context]

    def get_probability(self, grammar, nt_index, index, current_depth=None,
                        context=None):
        if self.learning_strategy in self.CONTEXT_STRATEGIES:
            if context is None:
                context = self._default_context(current_depth)
            return self.get_context_probabilities(grammar, nt_index, context)[index]
        if grammar is None:
            return self.pcfg[nt_index,index]
        if self.algorithm_method == AlgorithmMethod.PSGE_COPSGE:
            if self.learning_strategy == LearningStrategy.DEPTH_BASED:
                mean = np.mean(np.array([grammar[nt_index], self.pcfg[nt_index][current_depth]]), axis=0)
                return mean[current_depth][index]
            # TODO: arranjar maneira de receber tbm a gramatica geral
            # 1st approach: average of both probabilities
            else:

                mean = np.mean(np.array([grammar[nt_index], self.pcfg[nt_index]]), axis=0)
                return mean[index]
            # 2nd approach: hadamart product: mul matrizes + norm
            # comb = np.multiply(grammar, self.pcfg)
            # comb /= np.sum(comb)
            # return comb[index]
            # 3rd approach: LSE Log-Sum-Exponent
            # alpha = 1.0
            # beta = 1.0
            # S = alpha * np.log(grammar[nt_index]) + beta * np.log(self.pcfg[nt_index])
            # comb = np.exp(S)
            # comb /= np.sum(comb)
            # return comb[index]
        else:
            if self.learning_strategy == LearningStrategy.DEPTH_BASED:
                return grammar[nt_index][current_depth][index]
            else:
                return grammar[nt_index,index]
        
    def get_probabilities_non_terminal(self, grammar, nt_index,
                                       current_depth=None, context=None):
        if self.learning_strategy in self.CONTEXT_STRATEGIES:
            if context is None:
                context = self._default_context(current_depth)
            return self.get_context_probabilities(grammar, nt_index, context)
        if grammar is None:
            return self.pcfg[nt_index]
        if self.algorithm_method == AlgorithmMethod.PSGE_COPSGE:
            if self.learning_strategy == LearningStrategy.DEPTH_BASED:
                mean = np.mean(np.array([grammar[nt_index][current_depth], self.pcfg[nt_index][current_depth]]), axis=0)
                return mean[current_depth]
            else:
                mean = np.mean(np.array([grammar[nt_index], self.pcfg[nt_index]]), axis=0)
                return mean
        else:
            if self.learning_strategy == LearningStrategy.DEPTH_BASED:
                return grammar[nt_index][current_depth]
            else:
                return grammar[nt_index]
    
    def generate_empty_grammar_counter(self):
        if (self.learning_strategy == LearningStrategy.DEPTH_BASED or
                self.learning_strategy in self.CONTEXT_STRATEGIES):
            gram_counter = [{} for nt in self.get_non_terminals()]
        else:
            gram_counter = [[0] * len(self.grammar[nt]) for nt in self.get_non_terminals()]
        return gram_counter


    def update_grammar_counter(self, grammar_counter, symbol,
                               expansion_possibility, depth, context=None):
        if self.learning_strategy in self.CONTEXT_STRATEGIES:
            nt = list(self.get_non_terminals())[symbol]
            number_productions = len(self.grammar[nt])
            if self.learning_strategy == LearningStrategy.CONTEXT_AWARE_DEPTH:
                parent, context_depth = context
                parent_table = grammar_counter[symbol].setdefault(parent, {})
                counts = parent_table.setdefault(
                    context_depth, [0] * number_productions
                )
            else:
                counts = grammar_counter[symbol].setdefault(
                    context, [0] * number_productions
                )
            counts[expansion_possibility] += 1
            return grammar_counter
        if self.learning_strategy == LearningStrategy.DEPTH_BASED:
            if depth not in grammar_counter[symbol]:
                nt = list(self.get_non_terminals())[symbol]
                grammar_counter[symbol][depth] = [0] * len(self.grammar[nt])
            grammar_counter[symbol][depth][expansion_possibility] += 1
        else:
            grammar_counter[symbol][expansion_possibility] += 1
        return grammar_counter


    def recursive_individual_creation(self, genome, symbol, current_depth, probs):
        # TODO: adapt according to genotype distribution
        codon = np.random.uniform()
        nt_index = self.index_of_non_terminal[symbol]
        if current_depth > self.max_init_depth:
            shortest_path = self.shortest_path[(symbol,'NT')]
            prob_non_recursive = 0.0
            for rule in shortest_path[1:]:
                index = self.grammar[symbol].index(rule)
                prob_non_recursive += self.get_probability(probs, nt_index, index)
            prob_aux = 0.0
            for rule in shortest_path[1:]:
                index = self.grammar[symbol].index(rule)
                new_prob = self.get_probability(probs, nt_index, index) / prob_non_recursive
                prob_aux += new_prob
                if codon <= round(prob_aux,3):
                    expansion_possibility = index
                    break
        else:
            prob_aux = 0.0
            for index in range(len(self.grammar[symbol])):
                prob_aux += self.get_probability(probs, nt_index, index)
                if codon <= round(prob_aux,3):
                    expansion_possibility = index
                    break

        genome[self.get_non_terminals().index(symbol)].append([expansion_possibility,codon,current_depth])
        expansion_symbols = self.grammar[symbol][expansion_possibility]
        depths = [current_depth]
        for sym in expansion_symbols:
            if sym[1] != self.T:
                depths.append(self.recursive_individual_creation(genome, sym[0], current_depth + 1, probs))
        return max(depths)


    def recursive_individual_creation_sge_like(self, genome, symbol, current_depth, probs):
        codon = np.random.uniform()
        nt_index = self.index_of_non_terminal[symbol]
        if current_depth > self.max_init_depth:
            shortest_path = self.shortest_path[(symbol,'NT')]
            prob = 0.0
            choices = shortest_path[1:]
            rule = choices[np.random.randint(0, len(choices))]
            index = self.grammar[symbol].index(rule)

            if self.get_probability(probs, nt_index, index) == 0.0:
                choices.remove(rule)
                rule = choices[np.random.randint(0, len(choices))]
                index = self.grammar[symbol].index(rule)
            k = 0
            for i in self.get_probabilities_non_terminal(probs, nt_index):
                if k == index:
                    break
                prob += i
                k += 1
            codon = np.random.uniform(prob, prob + self.get_probability(probs, nt_index, index))
            expansion_possibility = index
        else:
            prob_aux = 0.0
            for index in range(len(self.grammar[symbol])):
                prob_aux += self.get_probability(probs, nt_index, index)
                if codon <= round(prob_aux,3):
                    expansion_possibility = index
                    break

        genome[self.get_non_terminals().index(symbol)].append([expansion_possibility,codon,current_depth])
        expansion_symbols = self.grammar[symbol][expansion_possibility]
        depths = [current_depth]
        for sym in expansion_symbols:
            if sym[1] != self.T:
                depths.append(self.recursive_individual_creation_sge_like(genome, sym[0], current_depth + 1, probs))
        return max(depths)
    
    def mapping(self, probs, mapping_rules, positions_to_map=None, needs_python_filter=False):
        if positions_to_map is None:
            positions_to_map = [0] * len(self.ordered_non_terminals)
        # gram_counter = [[0] * len(self.grammar[nt]) for nt in self.get_non_terminals()]
        gram_counter = self.generate_empty_grammar_counter()
        previous_expansions = [self.PREVIOUS_START_CONTEXT
                               for _ in self.ordered_non_terminals]
        output = []
        max_depth = self._recursive_mapping(
            probs, mapping_rules, positions_to_map, gram_counter,
            self.start_rule, 0, output, None, previous_expansions
        )
        if self.grammar_file.endswith("pybnf"):
            if needs_python_filter:
                # Indentation filtering operates on the complete program.
                output = [self.python_filter("".join(output), True)]
            else:
                # Keep grammar terminals tokenized while applying substitutions.
                output = [self.python_filter(token, False) for token in output]
        return output, max_depth, gram_counter

    def _recursive_mapping(self, probs, mapping_rules, positions_to_map,
                           gram_counter, current_sym, current_depth, output,
                           parent_symbol=None, previous_expansions=None):
        depths = [current_depth]
        if current_sym[1] == self.T:
            output.append(current_sym[0])
        else:
            current_sym_pos = self.ordered_non_terminals.index(current_sym[0])
            choices_expand = self.grammar[current_sym[0]]
            shortest_path = self.shortest_path[current_sym]
            nt_index = self.index_of_non_terminal[current_sym[0]]
            if self.learning_strategy == LearningStrategy.CONTEXT_AWARE:
                context = parent_symbol or self.ROOT_CONTEXT
            elif self.learning_strategy == LearningStrategy.CONTEXT_AWARE_DEPTH:
                context = (parent_symbol or self.ROOT_CONTEXT, current_depth)
            elif self.learning_strategy == LearningStrategy.CONTEXT_AWARE_PREVIOUS:
                context = previous_expansions[nt_index]
            else:
                context = None

            if positions_to_map[current_sym_pos] >= len(mapping_rules[current_sym_pos]):
                # TODO: nota: cma es nao entra aqui, mas colocar na mesma aqui um if a definir o tipo de distribuição
                codon = np.random.uniform()
                if current_depth >= (self.max_depth - shortest_path[0]):
                    prob_non_recursive = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        prob_non_recursive += self.get_probability(probs, nt_index, index, current_depth, context)
                    prob_aux = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        if prob_non_recursive == 0.0:
                            new_prob = 1.0 / len(shortest_path[1:])
                        else:
                            new_prob = self.get_probability(probs, nt_index, index, current_depth, context) / prob_non_recursive
                        # new_prob = probs[nt_index][index] / prob_non_recursive
                        prob_aux += new_prob
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index in range(len(self.grammar[current_sym[0]])):
                        prob_aux += self.get_probability(
                            probs, nt_index, index, current_depth, context
                        )
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                mapping_rules[current_sym_pos].append([expansion_possibility,codon,current_depth])
            else:
                # re-mapping with new probabilities
                # IF I START GENOTYPE WITH -1, i generate a new codon
                # if mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]][0] == -1:
                #     codon = np.random.uniform()
                # else:
                codon = mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]][1]
                if current_depth >= (self.max_depth - shortest_path[0]):
                    prob_non_recursive = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        prob_non_recursive += self.get_probability(probs, nt_index, index, current_depth, context)
                    prob_aux = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        if prob_non_recursive == 0.0:
                            new_prob = 1.0 / len(shortest_path[1:])
                        else:
                            new_prob = self.get_probability(probs, nt_index, index, current_depth, context) / prob_non_recursive
                        # new_prob = probs[nt_index][index] / prob_non_recursive
                        prob_aux += new_prob
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index in range(len(self.grammar[current_sym[0]])):
                        prob_aux += self.get_probability(
                            probs, nt_index, index, current_depth, context
                        )
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                # update mapping rules com a updated expansion possibility
                # print(mapping_rules[current_sym_pos])
                mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]] = [expansion_possibility, codon, current_depth]
            gram_counter = self.update_grammar_counter(
                gram_counter, current_sym_pos, expansion_possibility,
                current_depth, context
            )

            if self.learning_strategy == LearningStrategy.CONTEXT_AWARE_PREVIOUS:
                previous_expansions[nt_index] = expansion_possibility

            current_production = expansion_possibility
            positions_to_map[current_sym_pos] += 1
            next_to_expand = choices_expand[current_production]
            for next_sym in next_to_expand:
                depths.append(
                    self._recursive_mapping(
                        probs, mapping_rules, positions_to_map, gram_counter,
                        next_sym, current_depth + 1, output,
                        current_sym[0], previous_expansions
                    ))
        return max(depths)

    def _recursive_mapping_hybrid_not_aware(self, probs, mapping_rules, positions_to_map, gram_counter, current_sym, current_depth, output):
        # codigo para a mutacao not aware, voltar a usar so float, em vez de converter de int p float
        depths = [current_depth]
        if current_sym[1] == self.T:
            output.append(current_sym[0])
        else:
            current_sym_pos = self.ordered_non_terminals.index(current_sym[0])
            choices_expand = self.grammar[current_sym[0]]
            shortest_path = self.shortest_path[current_sym]
            nt_index = self.index_of_non_terminal[current_sym[0]]

            if positions_to_map[current_sym_pos] >= len(mapping_rules[current_sym_pos]):
                codon = np.random.uniform()
                if current_depth >= (self.max_depth - shortest_path[0]):
                    prob_non_recursive = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        prob_non_recursive += self.get_probability(probs, nt_index, index, current_depth)
                    prob_aux = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        if prob_non_recursive == 0.0:
                            new_prob = 1.0 / len(shortest_path[1:])
                        else:
                            new_prob = self.get_probability(probs, nt_index, index, current_depth) / prob_non_recursive
                        # new_prob = probs[nt_index][index] / prob_non_recursive
                        prob_aux += new_prob
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index in range(len(self.grammar[current_sym[0]])):
                        prob_aux += self.get_probability(probs, nt_index, index, current_depth)
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                mapping_rules[current_sym_pos].append([expansion_possibility,codon,current_depth])
                # gram_counter[current_sym_pos][expansion_possibility] += 1 
                gram_counter = self.update_grammar_counter(gram_counter, current_sym_pos, expansion_possibility, current_depth)
            
            # update mapping rules com a updated expansion possibility
            # mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]] = [expansion_possibility,codon,current_depth]
            # current_production = expansion_possibility
            current_production = mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]][0]
            positions_to_map[current_sym_pos] += 1
            next_to_expand = choices_expand[current_production]
            for next_sym in next_to_expand:
                depths.append(
                    self._recursive_mapping_hybrid_not_aware(probs, mapping_rules, positions_to_map, gram_counter, next_sym, current_depth + 1, output))
        return max(depths)
    


    def _recursive_mapping_sge_like(self, probs, mapping_rules, positions_to_map, gram_counter, current_sym, current_depth, output):
        depths = [current_depth]
        if current_sym[1] == self.T:
            output.append(current_sym[0])
        else:
            current_sym_pos = self.ordered_non_terminals.index(current_sym[0])
            choices_expand = self.grammar[current_sym[0]]
            shortest_path = self.shortest_path[current_sym]
            nt_index = self.index_of_non_terminal[current_sym[0]]

            if positions_to_map[current_sym_pos] >= len(mapping_rules[current_sym_pos]):
                # codon = np.random.uniform()
                if current_depth >= (self.max_depth - shortest_path[0]):
                    prob = 0.0
                    choices = shortest_path[1:]
                    rule = choices[np.random.randint(0, len(choices))]
                    index = self.grammar[current_sym[0]].index(rule)

                    k = 0
                    for i in self.get_probabilities_non_terminal(probs, nt_index):
                        if k == index:
                            break
                        prob += i
                        k += 1
                    # TODO: HERE HERE HERE
                    codon = (prob + prob + self.get_probability(probs, nt_index, index)) / 2
                    # codon = np.random.uniform(prob, prob + self.get_probability(probs, nt_index, index))
                    expansion_possibility = index
                else:
                    # TODO: HERE HERE HERE
                    # print(choices_expand)
                    if len(choices_expand) == 1:
                        expansion_possibility = 0
                        codon = 0.01
                    else:
                        index = np.random.randint(0, len(choices_expand))

                        prob = 0.0
                        # if self.get_probability(probs, nt_index, index) == 0.0:
                        #     continue
                        k = 0
                        for i in self.get_probabilities_non_terminal(probs, nt_index):
                            if k == index:
                                break
                            prob += i
                            k += 1

                        codon = (prob + prob + self.get_probability(probs, nt_index, index)) / 2
                        expansion_possibility = index

                    # UNTIL HERE
                mapping_rules[current_sym_pos].append([expansion_possibility,codon,current_depth])
         
            # update mapping rules com a updated expansion possibility
            # mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]] = [expansion_possibility,codon,current_depth]
                gram_counter[current_sym_pos][expansion_possibility] += 1   
            # current_production = expansion_possibility
            current_production = mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]][0]
            positions_to_map[current_sym_pos] += 1
            next_to_expand = choices_expand[current_production]
            for next_sym in next_to_expand:
                depths.append(
                    self._recursive_mapping_sge_like(probs, mapping_rules, positions_to_map, gram_counter, next_sym, current_depth + 1, output))
        return max(depths)
    

    def compute_non_recursive_options(self):
        for key in self.grammar.keys():
            prob_non_recursive = 0.0
            non_recursive_prods = []
            for index, option in enumerate(self.grammar[key]):
                for s in option:
                    if s[0] == key:
                        break
                else:
                    prob_non_recursive += self.pcfg[self.index_of_non_terminal[key],index]
                    non_recursive_prods.append([index, option])
            self.non_recursive_options[key] = [non_recursive_prods, prob_non_recursive]

    def get_non_recursive_options(self, symbol):
        return self.non_recursive_options[symbol]


    def get_dict(self):
        return self.grammar

    def get_pcfg(self):
        return self.pcfg

    def set_pcfg(self, pcfg):
        if self.learning_strategy in self.CONTEXT_STRATEGIES:
            if not isinstance(pcfg, list) or len(pcfg) != len(self.grammar):
                raise ValueError("Context grammar probability table has invalid size")
            restored = copy.deepcopy(pcfg)

            def convert_vectors(value, expected_size):
                if isinstance(value, dict):
                    return {key: convert_vectors(item, expected_size)
                            for key, item in value.items()}
                array = np.asarray(value, dtype=float)
                if (array.ndim != 1 or len(array) != expected_size or
                        not np.all(np.isfinite(array))):
                    raise ValueError("Invalid context probability vector")
                total = np.sum(array)
                if np.any(array < 0) or total <= 0:
                    raise ValueError("Invalid context probability vector")
                return array / total

            converted = [
                convert_vectors(
                    table,
                    len(self.grammar[list(self.get_non_terminals())[nt_index]]),
                )
                for nt_index, table in enumerate(restored)
            ]
            if self.learning_strategy == LearningStrategy.CONTEXT_AWARE_DEPTH:
                converted = [
                    {
                        parent: {
                            int(depth): probabilities
                            for depth, probabilities in depth_table.items()
                        }
                        for parent, depth_table in table.items()
                    }
                    for table in converted
                ]
            elif self.learning_strategy == LearningStrategy.CONTEXT_AWARE_PREVIOUS:
                converted = [
                    {
                        (key if key == self.PREVIOUS_START_CONTEXT else int(key)):
                        probabilities
                        for key, probabilities in table.items()
                    }
                    for table in converted
                ]
            self.pcfg = converted
            self.pcfg_mask = None
            return
        restored = np.array(pcfg, copy=True)
        if self.pcfg is not None and restored.shape != np.asarray(self.pcfg).shape:
            raise ValueError(
                "Checkpoint grammar probability shape %s does not match grammar shape %s"
                % (restored.shape, np.asarray(self.pcfg).shape)
            )
        self.pcfg = restored
        self.pcfg_mask = self.pcfg != 0

    def get_shortest_path(self):
        return self.shortest_path
    

    def calculate_max_expansions_recursive(self, DP, curr_symbol, curr_depth):
        #print("Calculating max expansions for", curr_symbol, "at depth", curr_depth)
        curr_sym,curr_nt = curr_symbol
        if curr_nt == self.T:
            return {}
        if curr_depth == 0:
            return {}
        final_res = {}
        if curr_sym in DP:
            if curr_depth in DP[curr_sym]:
                return DP[curr_sym][curr_depth]
        for rule in self.grammar[curr_sym]:
            rule_res = {}
            # For each rule, we calculate the maximum expansions recursively
            for symbol in rule:
                sym,nt = symbol
                if nt == self.T:
                    continue
                aux_res = self.calculate_max_expansions_recursive(DP, symbol, curr_depth - 1)
                for k,v in aux_res.items():
                    rule_res[k] = rule_res.get(k, 0) + v
            for k,v in rule_res.items():
                final_res[k] = max(final_res.get(k, 0), v)
        final_res[curr_sym] = final_res.get(curr_sym, 0) + 1
        DP[curr_sym] = DP.get(curr_sym, {})
        DP[curr_sym][curr_depth] = final_res
        return final_res

    def calculate_max_expansions_iterative(self, grammar, max_depth):
        """
        Calculate an approximation of the maximum number of expansions in a tree with depth max_depth
        using iterative dynamic programming.
        """
        non_terminals = list(grammar.keys())
        
        # dp[depth][non_terminal] = max expansions at that depth
        dp = [{nt: 0 for nt in non_terminals} for _ in range(max_depth + 1)]
        
        # Base case: depth 0
        for nt in non_terminals:
            dp[0][nt] = 1
        
        # Fill the DP table iteratively
        for depth in range(1, max_depth + 1):
            for nt in non_terminals:
                max_expansions = 0
                
                for production in grammar[nt]:  # Each production is a list of tuples
                    current_expansions = 1
                    
                    for symbol in production:
                        if symbol[1] == self.NT:  # Check if the symbol is a non-terminal
                            current_expansions *= dp[depth - 1][symbol[0]]
                        # Terminals (Grammar.T) contribute a factor of 1
                    
                    max_expansions = max(max_expansions, current_expansions)
                
                dp[depth][nt] = max_expansions
        
        return dp[max_depth]

    
    def count_references_to_non_terminals(self):
        return self.number_of_references_by_non_terminal


    @staticmethod
    def python_filter(txt, needs_python_filter):
        """ Create correct python syntax.
        We use {: and :} as special open and close brackets, because
        it's not possible to specify indentation correctly in a BNF
        grammar without this type of scheme."""
        txt = txt.replace(r"\le", "<=")
        txt = txt.replace(r"\ge", ">=")
        txt = txt.replace(r"\l", "<")
        txt = txt.replace(r"\g", ">")
        txt = txt.replace(r"\eb", "|")
        if needs_python_filter:
            indent_level = 0
            tmp = txt[:]
            i = 0
            while i < len(tmp):
                tok = tmp[i:i+2]
                if tok == "{:":
                    indent_level += 1
                elif tok == ":}":
                    indent_level -= 1
                tabstr = "\n" + "  " * indent_level
                if tok == "{:" or tok == ":}" or tok == "\\n":
                    tmp = tmp.replace(tok, tabstr, 1)
                i += 1
                # Strip superfluous blank lines.
                txt = "\n".join([line for line in tmp.split("\n") if line.strip() != ""])
        return txt

    def get_start_rule(self):
        return self.start_rule

    def __str__(self):
        grammar = self.grammar
        text = ""
        for key in self.ordered_non_terminals:
            text += key + " ::= "
            for options in grammar[key]:
                for option in options:
                    text += option[0]
                if options != grammar[key][-1]:
                    text += " | "
            text += "\n"
        return text

# Create one instance and export its methods as module-level functions.
# The functions share state across all uses
# (both in the user's code and in the Python libraries), but that's fine
# for most programs and is easier for the casual user


_inst = Grammar()
set_path = _inst.set_path
set_pcfg_path = _inst.set_pcfg_path
read_grammar = _inst.read_grammar
get_non_terminals = _inst.get_non_terminals
count_number_of_options_in_production = _inst.count_number_of_options_in_production
list_non_recursive_productions = _inst.list_non_recursive_productions
recursive_individual_creation = _inst.recursive_individual_creation
mapping = _inst.mapping
start_rule = _inst.get_start_rule
set_max_tree_depth = _inst.set_max_tree_depth
set_min_init_tree_depth = _inst.set_min_init_tree_depth
get_max_depth = _inst.get_max_depth
get_non_recursive_options = _inst.get_non_recursive_options
# compute_non_recursive_options = _inst.compute_non_recursive_options
get_count_references_to_non_terminals = _inst.count_references_to_non_terminals
get_dict = _inst.get_dict
get_pcfg = _inst.get_pcfg
set_pcfg = _inst.set_pcfg
get_probability = _inst.get_probability
get_probabilities_non_terminal = _inst.get_probabilities_non_terminal
get_context_probabilities = _inst.get_context_probabilities
get_mask = _inst.get_mask
get_shortest_path = _inst.get_shortest_path
get_index_of_non_terminal = _inst.get_index_of_non_terminal
ordered_non_terminals = _inst.ordered_non_terminals
max_init_depth = _inst.get_max_init_depth
python_filter = _inst.python_filter
if __name__ == "__main__":
    np.random.seed(42)
    g = Grammar("grammars/regression.txt", 9)
    genome = [[0], [0, 3, 3], [0], [], [1, 1]]
    mapping_numbers = [0] * len(genome)
    print(g.mapping(genome, mapping_numbers, needs_python_filter=True))
