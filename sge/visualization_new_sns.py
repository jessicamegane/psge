import re
import os
import json
from plotnine import *
# from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial

plt.rcParams["figure.figsize"] = (20,15)
plt.rcParams['axes.grid'] = True
# plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'FFFFFF'
plt.rcParams['axes.facecolor'] = 'FFFFFF'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams.update({'font.size': 26})

NT = "NT"
T = "T"
NT_PATTERN = "(<.+?>)"
RULE_SEPARATOR = "::="
PRODUCTION_SEPARATOR = "|"


"""
    Probabilities analysis, made both for standard probabilities array and for depth dependency.
"""

def load_dataframe(path, run_file, generation):
    run = re.search(r'\d+', run_file).group()
    df = pd.DataFrame()    
    iteration_paths = os.listdir(os.path.join(path, run_file))
    # print(iteration_paths)
    for iteration in iteration_paths:
        if "iteration" not in iteration:
            # exclude parameters and progress report
            continue
        iteration_number = re.search(r'\d+', iteration).group()
        if (generation > 0 and generation != int(iteration_number)):
            continue
        iteration_file = os.path.join(path, run_file, iteration)
        if not os.path.isfile(iteration_file):
            continue
        f = open(iteration_file)
        data = json.load(f)

        grammar = data[0]['grammar']
        for nt_index, nt in enumerate(grammar):
            # if nt_index == 0:
            #     continue
            if type(nt[0]) == list:
                if df.empty:
                    df = pd.DataFrame(columns=['run','generation','nt', 'symb', 'depth', 'prob'])
                for depth_index, depth in enumerate(nt):
                    for rule_index, rule in enumerate(depth):
                        df2 = [int(run), iteration_number, nt_index,  rule_index,  depth_index,  rule]
                        df.loc[len(df)] = df2
                        # rule contains the probability
            else:
                if df.empty:
                    df = pd.DataFrame(columns=['run','generation','nt', 'symb', 'prob'])
                # in case is a number, it means that it contains the probability
                for rule_index, rule in enumerate(nt):
                        df2 = [int(run), iteration_number, nt_index,  rule_index,  rule]
                        df.loc[len(df)] = df2
                        # rule contains the probability

    return df

def read_probabilities(path, generation=-1):
    # processar valores
    folders = os.listdir(path)
    if "aggregated_probabilities.csv" in folders:
        if generation > 0:
            data = pd.read_csv(os.path.join(path,"aggregated_probabilities.csv"), sep="\t", header=0)
            return data[data['generation'] == generation]
        return pd.read_csv(os.path.join(path,"aggregated_probabilities.csv"), delim_whitespace=True, header=0)
    # df = pd.DataFrame(columns=['run','generation','nt', 'symb', 'depth', 'prob'])
    df = pd.DataFrame()
    for folder in folders:
        if "fold" in folder:
            fold = re.search(r'fold\d+', folder).group()
            fold = re.search(r'\d+', fold).group()
            run_folders = os.listdir(os.path.join(path, folder))
            for f in run_folders:
                if "run" not in f:
                    continue
                new_df = load_dataframe(os.path.join(path, folder), f, generation)
                new_df['fold'] = [fold] * len(new_df)
                if df.empty:
                    df = new_df
                else:
                    df = pd.concat([df, new_df])
        else:
            f = folder
            if "run" not in f:
                continue
            new_df = load_dataframe(path, f, generation)
            if df.empty:
                df = new_df
            else:
                df = pd.concat([df, new_df])

    df.to_csv(path + 'aggregated_probabilities.csv', index=False, sep='\t', header=True)
    return df

def read_grammar_as_labels(problem):
    grammars = {"quartic": "grammars/quad_regression.pybnf", "pagie": "grammars/regression.pybnf", "bostonhousing": "grammars/bostonhousing.bnf", "bioavailability": "grammars/feat_engineering_bioav_torch.pybnf"}
    non_terminals = []
    grammar = {}
    path = grammars[problem]
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("#") and line.strip() != "":
                if line.find(PRODUCTION_SEPARATOR):
                    left_side, productions = line.split(RULE_SEPARATOR)
                    left_side = left_side.strip()
                    if not re.search(NT_PATTERN, left_side):
                        raise ValueError("Left side not a non-terminal!")

                    temp_productions = []
                    for production in [production.strip() for production in productions.split(PRODUCTION_SEPARATOR)]:
                        temp_production = []
                        if not re.search(NT_PATTERN, production):
                            if production == "None":
                                production = ""
                            temp_production.append((production, T))
                        else:
                            for value in re.findall("<.+?>|[^<>]*", production):
                                if value != "":
                                    if re.search(NT_PATTERN, value) is None:
                                        sym = (value, T)
                                    else:
                                        sym = (value, NT)
                                    temp_production.append(sym)
                        temp_productions.append(temp_production)                          
                    if left_side not in grammar:
                        grammar[left_side] = temp_productions
                        non_terminals.append(left_side)
    return non_terminals, grammar

def plot_probabilities_depth(path, df, non_terminals, grammar, display=True):
    mean_df = df.groupby(['generation','nt','symb','depth']).mean().reset_index()
    
    number_nt = mean_df['nt'].max()
    for i in range(1,int(number_nt) + 1):
        non_terminal_label = non_terminals[i]
        df = mean_df[mean_df['nt'] == float(i)]
        p = (ggplot(df, aes(x='generation', y='prob', color='factor(symb)')) 
        + labs(x='Generations', y='Probability')
        + scale_color_discrete(labels = grammar[non_terminal_label], name='Production Rules')
        + geom_line()
        + facet_wrap('depth')
        + ggtitle(f'Probabilities evolution for non-terminal {non_terminal_label}')
        )

        if display:
            print(p)
        p.save(filename = f'{path}/prob_{non_terminal_label}.png')

def plot_probabilities(path, df, non_terminals, grammar, display=True):
    mean_df = df.groupby(['generation','nt','symb']).mean().reset_index()
    
    print(mean_df)
    number_nt = mean_df['nt'].max()
    for i in range(1,int(number_nt) + 1):
        non_terminal_label = non_terminals[i]
        df = mean_df[mean_df['nt'] == float(i)]
        df = df[df['symb'] < len(grammar[non_terminal_label])]

        p = (ggplot(df, aes(x='generation', y='prob', color='factor(symb)')) 
        + labs(x='Generations', y='Probability')
        + scale_color_discrete(labels = grammar[non_terminal_label], name='Production Rules')
        + geom_line()
        + ggtitle(f'Probabilities evolution for non-terminal {non_terminal_label}')
        )

        if display:
            print(p)
        p.save(filename = f'{path}/prob_{non_terminal_label}.png')

def process_probabilities(problem, path):
    data = read_probabilities(path)
    non_terminals, grammar = read_grammar_as_labels(problem)
    if 'depth' in data.columns:
        plot_probabilities_depth(path, data, non_terminals, grammar, display=False)
    else:
        plot_probabilities(path, data, non_terminals, grammar, display=False)

def probabilities(problem, paths, multiprocess=False):
    if multiprocess:
        func = partial(process_probabilities,problem=problem)
        n = multiprocessing.Process(target=func, args=(paths))
        n.start()
        n.join()
    else:
        for exp_name, path in paths:
            print(path)
            data = read_probabilities(path)
            non_terminals, grammar = read_grammar_as_labels(problem)
            if 'depth' in data.columns:
                plot_probabilities_depth(path, data, non_terminals, grammar, display=False)
            else:
                plot_probabilities(path, data, non_terminals, grammar, display=False)




def probabilities_visualizer_path(path, problem):
    # barplot of probabilities of given grammar (or path)
    # visualizer for one json grammar
    non_terminals, grammar = read_grammar_as_labels(problem)
    # f = open(path)
    # data = json.load(f)
    # print(data)
    # print(grammar)    
    save_path = path + '/barplot_probabilities/' + 'best_train_psge'
    data = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4512355717653513, 0.0727775848697871, 0.47598684336486174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7376251540113984, 0.0008219061008910205, 0.22861312496439265, 0.03293981492331794, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.06452996227733603, 9.466611593890934e-06, 9.466611593890934e-06, 0.00014311283785715001, 0.7988970905562057, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 1.4185702949999466e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 0.0001163867562328725, 1.1650481258325223e-05, 0.022295812957152408, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 0.0033207219755559234, 9.466611593890934e-06, 7.489416573723386e-05, 3.0316827968078145e-05, 9.466611593890934e-06, 9.466611593890934e-06, 5.15014169593947e-05, 9.466611593890934e-06, 0.00011281680538307593, 1.3878900829885018e-05, 0.000649735653251617, 9.466611593890934e-06, 9.466611593890934e-06, 0.00018402319098186193, 9.466611593890934e-06, 1.50438872694007e-05, 9.466611593890934e-06, 9.466611593890934e-06, 0.000951916829741659, 9.466611593890934e-06, 1.1670448560685559e-05, 9.466611593890934e-06, 2.1323781671087706e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 4.0287868240851986e-05, 1.722467994731812e-05, 9.466611593890934e-06, 9.466611593890934e-06, 1.4522553492077167e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 5.6453831954330625e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 6.0609912046142455e-05, 9.466611593890934e-06, 9.714844307631549e-05, 7.024754592617154e-05, 9.466611593890934e-06, 3.0316827968078145e-05, 1.1670448560685559e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 1.50438872694007e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 1.644406671892217e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 1.50438872694007e-05, 9.466611593890934e-06, 0.00033086866056721, 0.00040962202927639685, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 0.0003157934384535554, 0.0036770865663245463, 9.466611593890934e-06, 9.466611593890934e-06, 1.1672971875910995e-05, 9.466611593890934e-06, 0.0004614211800711497, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 0.00012378010495153565, 0.00027203246516126407, 1.5752995695431617e-05, 9.466611593890934e-06, 0.01239611859370464, 1.644406671892217e-05, 1.1650481258325223e-05, 9.466611593890934e-06, 9.466611593890934e-06, 0.0004999351124816369, 9.466611593890934e-06, 0.00021939615196112281, 9.466611593890934e-06, 9.466611593890934e-06, 3.635255109054833e-05, 9.466611593890934e-06, 1.644406671892217e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 1.1672971875910995e-05, 0.001791947110336834, 9.466611593890934e-06, 0.00032702784357366456, 2.57873725322432e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 0.0005102892444874989, 1.3876377514659608e-05, 1.802556583724322e-05, 9.144156584883015e-05, 9.466611593890934e-06, 0.0005401856868524163, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 3.0568172340453475e-05, 9.466611593890934e-06, 0.00016094254893944164, 2.422228329771761e-05, 9.466611593890934e-06, 1.802556583724322e-05, 9.466611593890934e-06, 9.466611593890934e-06, 2.1316781916857807e-05, 9.466611593890934e-06, 2.329348208069315e-05, 0.00015749319571866963, 9.466611593890934e-06, 9.063116923356902e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 2.6443282382839657e-05, 1.820709977188229e-05, 1.178174396419028e-05, 9.466611593890934e-06, 9.466611593890934e-06, 2.3974238760535063e-05, 1.1672971875910995e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 2.3716581630166215e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 0.0002999459104999715, 2.5692587962289476e-05, 9.466611593890934e-06, 1.8665415873719446e-05, 0.00036093539373755666, 0.022342299169649388, 6.409096906791455e-05, 0.0007184724480999801, 9.466611593890934e-06, 8.958560212422181e-05, 9.466611593890934e-06, 9.466611593890934e-06, 2.2684356050983132e-05, 9.466611593890934e-06, 9.466611593890934e-06, 1.4965300666657151e-05, 0.0003315130150749512, 9.466611593890934e-06, 9.466611593890934e-06, 1.338630860529376e-05, 3.635255109054833e-05, 2.377662295285367e-05, 1.644406671892217e-05, 9.466611593890934e-06, 9.466611593890934e-06, 4.823524964357582e-05, 1.154245345532046e-05, 1.588618214064089e-05, 9.466611593890934e-06, 1.4737692883310802e-05, 2.6911720131200623e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 3.174753867358543e-05, 0.00018600862905059583, 9.466611593890934e-06, 1.1650481258325223e-05, 9.466611593890934e-06, 0.0004711062781287337, 9.466611593890934e-06, 9.466611593890934e-06, 0.0005616770633799331, 9.466611593890934e-06, 1.644406671892217e-05, 2.1316781916857807e-05, 9.466611593890934e-06, 9.466611593890934e-06, 9.466611593890934e-06, 1.802556583724322e-05, 1.1672971875910995e-05, 0.0004296893748722241, 1.1672971875910995e-05, 9.466611593890934e-06, 9.466611593890934e-06, 0.017649745187803257, 4.1379866867092226e-05, 2.9036558527797167e-05, 2.9335880042030486e-05, 9.466611593890934e-06, 1.1650481258325223e-05, 1.1820634287516792e-05, 0.0003431217997781682, 9.466611593890934e-06, 9.466611593890934e-06, 1.1650481258325223e-05, 0.03963969149509837]]
    for nt_index, probabilities in enumerate(data):
        if nt_index != 0:
            f1 = plt.figure()
            print("non_terminal:", nt_index)
            rules = len(grammar[non_terminals[nt_index]])
            print("number rules:", rules)
            print(grammar[non_terminals[nt_index]])
            probabilities = probabilities[:rules]
            labels = [''.join(tup[0] for tup in sublist) for sublist in grammar[non_terminals[nt_index]]]
            print(labels)
            df = pd.DataFrame({'probs': probabilities, 'label': labels})
            df = df[df['probs'] > 0.01] 
            ax = sns.barplot(data=df, x='label', y='probs')
            ax.set(xlabel='Rules', ylabel='Probabilities', title=f'Grammar ind best fitness train, nt {non_terminals[nt_index]}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f'{save_path}/barplot_nt_{non_terminals[nt_index]}.png')


def probabilities_visualizer_mean_folders(problem, path, generation):
    # plots probabilities at generation X, mean of folders
    non_terminals, grammar = read_grammar_as_labels(problem)
    save_path = path + '/barplot_probabilities/' + 'generation_' + str(generation)
    df = read_probabilities(path, generation)

    if 'depth' in df.columns:
        df = df.groupby(['nt','symb','depth']).mean().reset_index()
        df = df[df['prob'] > 0.04] 
        number_nt = df['nt'].max()
        for nt_index in range(0, int(number_nt) + 1):
            f1 = plt.figure()
            labels = [''.join(tup[0] for tup in sublist) for sublist in grammar[non_terminals[nt_index]]]
            data = df[df['nt'] == nt_index]
            # data = data[data['prob'] > 0.02] 
            # data = data[data['symb'] < len(grammar[non_terminals[nt_index]])]
            # ax = sns.barplot(data=data, x='symb', y='prob', col="depth")
            ax = sns.catplot(data = data, x='symb', y='prob', kind="bar", col="depth", col_wrap=2, height=15, aspect=3.3)
            # ax.bar_label(ax.containers[0], fontsize=20);
            # ax.set(xlabel='Rules', ylabel='Probabilities', title=f'Probabilities nt {non_terminals[nt_index]}')
            # ax.set_xticklabels(labels)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f'{save_path}/barplot_nt_{non_terminals[nt_index]}.png')

    else:
        df = df.groupby(['nt','symb']).mean().reset_index()
        df = df[df['prob'] > 0.015] 
        number_nt = df['nt'].max()
        for nt_index in range(0, int(number_nt) + 1):
            f1 = plt.figure()
            labels = [''.join(tup[0] for tup in sublist) for sublist in grammar[non_terminals[nt_index]]]
            data = df[df['nt'] == nt_index]
            data = data[data['symb'] < len(grammar[non_terminals[nt_index]])]

            ax = sns.barplot(data=data, x='symb', y='prob')
            ax.bar_label(ax.containers[0], fontsize=20);
            ax.set(xlabel='Rules', ylabel='Probabilities', title=f'Probabilities nt {non_terminals[nt_index]}')
            # ax.set_xticklabels(labels)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f'{save_path}/barplot_nt_{non_terminals[nt_index]}.png')




"""
    Performance analysis
"""

# FUNçÂO QUE LEIA OS PROGRESS REPORT DE TODAS AS RUNS

def read_data(path, exp_name):
    print(path)
    print(exp_name)
    folders = os.listdir(path)
    # if "aggregated_data.csv" in folders:
    #     return pd.read_csv(os.path.join(path,"aggregated_data.csv"), delim_whitespace=True, header=0, na_values="nan")
    # else:
    data = []
    for folder in folders:
        if "run" not in folder:
            continue
        file_path = os.path.join(path, folder, "progress_report.csv")
        column_names = ['generation','best_fit','mean_fit','std_fit','best_test','tree_depth','mean_depth','median_depth','best_length','mean_length','median_length']
        df = pd.read_csv(file_path, delim_whitespace=False, sep = ';', names=column_names, na_values="nan")
        run = re.search(r'\d+', folder).group()
        df['run'] = [run] * (len(df.index))
        df['algorithm'] = [exp_name] * (len(df.index))

        data.append(df)
    
    new_table = pd.concat(data, axis=0)
    # new_table = new_table.drop('mean_fit', axis=1)
    # new_table = new_table.drop('std_fit', axis=1)
    print(new_table.size)
    new_table.to_csv(path + 'aggregated_data.csv', index=False, sep='\t', header=True, na_rep="nan")
    return new_table


def performance_plot_runs(problem, df, path, display=True):
    ###### Performance - average with different runs
    print(path)
    df = df.drop('mean_fit', axis=1)
    df = df.drop('std_fit', axis=1)
    print(df)
    mean_df = df.groupby(['generation','algorithm']).mean().reset_index()
    
    p = (ggplot(df, aes(x='generation', y='best_fit', color='factor(run)')) 
    + labs(x='Generations', y='Error')
    + scale_color_discrete(name = 'Run')
    + geom_line(linetype='dotted')
    + geom_line(mean_df, aes(y='best_fit'))
    + ggtitle(f'Performance for {problem} problem over runs - {path}')
    )

    if display:
        print(p)

    p.save(filename = f'{path}/{problem}_performance_plot_runs.png')

def boxplot(problem, df, generation, display=False):
    f1 = plt.figure()
    # remove outliers : fliersize=0
    bp = df[df['generation'] == generation]

    ax = sns.boxplot(data=bp, x='algorithm', y='best_fit')
    ax.set(xlabel='Algorithm', ylabel='Error', title=f'Boxplot for {problem} problem - Generation {generation}')

    plt.savefig(f'{problem}_boxplot_{generation}.png')

    if display:
        plt.show()

    if df['best_test'].sum() > 0:
        f2 = plt.figure()
        ax = sns.boxplot(data=bp, x='algorithm', y='best_test')
        ax.set(xlabel='Algorithm', ylabel='Error', title=f'Boxplot for {problem} problem - Generation {generation} - Test')

        plt.savefig(f'{problem}_boxplot_test_{generation}.png')
        if display:
            plt.show()

def boxplot_depth(problem, df, generation, display=False):
    # remove outliers : fliersize=0
    f1 = plt.figure()
    bp = df[df['generation'] == generation]

    ax = sns.boxplot(data=bp, x='algorithm', y='tree_depth', fliersize=0)
    ax.set(xlabel='Algorithm', ylabel='Depth', title=f'Boxplot of depth for {problem} problem - Generation {generation}')

    plt.savefig(f'{problem}_boxplot_depth_{generation}.png')

    if display:
        plt.show()

def boxplot_length(problem, df, generation, display=False):
    # remove outliers : fliersize=0
    f1 = plt.figure()
    bp = df[df['generation'] == generation]

    ax = sns.boxplot(data=bp, x='algorithm', y='best_length', fliersize=0)
    ax.set(xlabel='Algorithm', ylabel='Genotype size', title=f'Boxplot of length for {problem} problem - Generation {generation}')

    plt.savefig(f'{problem}_boxplot_length_{generation}.png')

    if display:
        plt.show()


def performance_plot(problem, df, display=False):
    f1 = plt.figure()
    ax = sns.lineplot(data=df, x='generation', y='best_fit', hue='algorithm', estimator='median', errorbar='sd')
    ax.set(xlabel='Generations', ylabel='Error', title=f'Performance for {problem} problem')
    ax.legend(title='Algorithms')
    # algorithms=['Standard', 'Dependent']
    # for i in range(len(algorithms)):
    #     ax.legend_.texts[i].set_text(algorithms[i])
    plt.savefig(f'{problem}_performance_plot.png')

    if display:
        plt.show()
   
    if df['best_test'].sum() > 0:
        f2 = plt.figure()

        ax = sns.lineplot(data=df, x='generation', y='best_test', hue='algorithm', estimator='median', errorbar=None)
        ax.set(xlabel='Generations', ylabel='Error', title=f'Performance for {problem} problem - Test')
        ax.legend(title='Algorithms')
        # algorithms=['Standard', 'Dependent']
        # for i in range(len(algorithms)):
        #     ax.legend_.texts[i].set_text(algorithms[i])
        plt.savefig(f'{problem}_performance_plot_test.png')

        if display:
            plt.show()

def performance(problem, paths):
    df = pd.DataFrame()

    for exp_name, path in paths:
        data = read_data(path, exp_name)
        # performance_plot_runs(problem, data, path, display=False)
        if df.empty:
            df = data
        else:
            df = pd.concat([df, data], axis=0)
    
    performance_plot(problem, df, display=False)
    boxplot(problem, df, 1, display=False)
    boxplot(problem, df, 200, display=False)
    boxplot(problem, df, 500, display=False)
    boxplot_depth(problem, df, 1, display=False)
    boxplot_depth(problem, df, 200, display=False)
    boxplot_depth(problem, df, 500, display=False)
    boxplot_length(problem, df, 1, display=False)
    boxplot_length(problem, df, 200, display=False)
    boxplot_length(problem, df, 500, display=False)
    # boxplot_depth(problem, df, 10, display=False)
    # boxplot_depth(problem, df, 50, display=False)
    # boxplot_depth(problem, df, 100, display=False)
    # boxplot_depth(problem, df, 200, display=False)
    # boxplot_depth(problem, df, 300, display=False)
    



def analysis(paths):
    for exp_name, path in paths:
        print("-----------")
        print(path)
        # best individual
        data = read_data(path, exp_name)
        data = data.reset_index()
        best_idx = data['best_fit'].idxmin()
        print(data.loc[[best_idx]])
        best_idx = data['best_test'].idxmin()
        print(data.loc[[best_idx]])

def analysis_probabilities(paths, generation):
    # gives probabilities at generation gen in the form of an array
    for _, path in paths:
        df = read_probabilities(path)

        df = df[df['generation'] == generation]
        df = df.groupby(['generation','nt','symb']).mean().reset_index()
        # print(mean_df)
        probs = []
        for i in range(10):
            p = []
            bp = df[df['nt'] == i]
            for ind in bp.index:
                p.append(bp['prob'][ind])

            # print(bp)
            # print(p)
            probs.append(p)
        
        print("PROBS GEN ", generation)
        print(probs)
        # mean_df = df.groupby(['generation','algorithm']).median().reset_index()
        # std_df = df.groupby(['generation','algorithm']).std().reset_index()


def bio_analysis():
    fi = 'resources/bio.csv'
    df = pd.read_csv(fi, sep=';', encoding = 'utf-8', skiprows=1, header=None).astype(np.float64)
    print(df)
    # # Assuming df is your DataFrame
    # # Store the column names before deletion
    # original_columns = df.columns.tolist()

    # # Delete columns where all values are 0
    # df = df.loc[:, (df != 0).any(axis=0)]

    # # Store the column names after deletion
    # remaining_columns = df.columns.tolist()

    # # Determine which columns were deleted
    # deleted_columns = list(set(original_columns) - set(remaining_columns))

    # print("Deleted columns:", deleted_columns)
    # # print(df)

    original_columns = df.columns.tolist()

    # Delete columns where all values are 0 except for one
    df = df.loc[:, (df != 0).sum() > 1]

    # Store the column names after deletion
    remaining_columns = df.columns.tolist()

    # Determine which columns were deleted
    deleted_columns = list(set(original_columns) - set(remaining_columns))

    print("Deleted columns:", deleted_columns)
    print(len(deleted_columns))

def bioav_feature_comparison():
    empt_columns = [128, 130, 131, 132, 133, 134, 135, 137, 138, 139, 12, 14, 15, 143, 144, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 38, 167, 168, 169, 173, 51, 52, 53, 54, 183, 185, 59, 187, 61, 192, 66, 67, 68, 69, 194, 71, 196, 199, 202, 75, 205, 206, 79, 83, 94, 109, 111, 112, 115, 119, 120, 121, 123, 125, 127]
    empty_columns = [i+1 for i in empt_columns]
    empty_columns.sort()
    print(empty_columns)
    archetti = [5, 9, 26, 34, 69, 84, 110, 126, 150, 188, 195, 206, 214, 218, 224, 234, 240]
    gp = [213, 29, 225, 143, 215, 219, 221, 27, 63, 85]
    info_gain = [219, 226, 213, 63, 225, 220, 221, 85]
    random_forest = [219, 226, 224, 223, 91, 241, 221, 19, 236, 22]

    psg_average = [0, 2, 28, 37, 90, 179, 229, 230, 240]
    psge_average = [i+1 for i in psg_average]

    # p > 0.015
    # dpsg_average = [6, 7, 8, 9, 12, 13, 16, 20, 22, 31, 37, 50, 52, 54, 59, 63, 64, 73, 77, 83, 87, 88, 95, 97, 98, 99, 100, 110, 111, 112, 114, 115, 120, 164, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 188, 189, 193, 198, 204, 209, 210, 211, 215, 217, 218, 225, 226, 230, 236, 237, 239, 242, 244, 245]
    # p > 0.02
    # dpsg_average = [6, 7, 8, 12, 13, 16, 22, 31, 37, 52, 59, 64, 73, 77, 83, 87, 97, 98, 99, 100, 110, 111, 114, 115, 120, 164, 177, 178, 179, 180, 181, 182, 183, 185, 188, 193, 198, 204, 209, 211, 215, 225, 237, 245]
    # dpsge_average = [i-4 for i in dpsg_average]

    # p > 0.04
    dpsg_04 = [7, 59, 111, 178, 179, 181, 182, 218, 237, 245]
    dpsge_04_average = [i-4 for i in dpsg_04]

    # p > 0.04
    dpsg_10_10_04 = [4, 8, 9, 33, 42, 50, 89, 234, 235, 245]
    dpsge_10_10_04 = [i-4 for i in dpsg_10_10_04]

    dpsg_17_04 = [4, 8, 9, 33, 42, 50, 95, 223, 234, 235, 237, 244, 245]
    dpsge_17_04_average = [i-4 for i in dpsg_17_04]


    psg_best_train = [11, 104, 179, 229, 240]
    psge_best_train = [i-4 for i in psg_best_train]


    # Convert lists to sets for easier comparison
    empty_columns_set = set(empty_columns)
    archetti_set = set(archetti)
    gp_set = set(gp)
    info_gain_set = set(info_gain)
    random_forest_set = set(random_forest)
    psge_average_set = set(psge_average)
    psge_best_train_set = set(psge_best_train)
    dpsge_10_10_04_set = set(dpsge_10_10_04)
    dpsge_04_set = set(dpsge_04_average)
    dpsge_17_04_set = set(dpsge_17_04_average)

    # Find elements in common between empty columns and each list
    archetti_empty_common = len(archetti_set.intersection(empty_columns_set))
    gp_empty_common = len(gp_set.intersection(empty_columns_set))
    info_gain_empty_common = len(info_gain_set.intersection(empty_columns_set))
    random_forest_empty_common = len(random_forest_set.intersection(empty_columns_set))
    psge_average_empty_common = len(psge_average_set.intersection(empty_columns_set))
    psge_best_train_empty_common = len(psge_best_train_set.intersection(empty_columns_set))
    dpsge_10_10_04_empty_common = len(dpsge_10_10_04_set.intersection(empty_columns_set))
    dpsge_04_average_empty_common = len(dpsge_04_set.intersection(empty_columns_set))
    dpsge_17_04_average_empty_common = len(dpsge_17_04_set.intersection(empty_columns_set))


    # Find elements in common between each pair of lists
    archetti_gp_common = len(archetti_set.intersection(gp_set))
    archetti_info_gain_common = len(archetti_set.intersection(info_gain_set))
    archetti_random_forest_common = len(archetti_set.intersection(random_forest_set))
    archetti_psge_average_common = len(archetti_set.intersection(psge_average_set))
    archetti_psge_best_train_common = len(archetti_set.intersection(psge_best_train_set))
    archetti_dpsge_10_10_04_common = len(archetti_set.intersection(dpsge_10_10_04_set))
    archetti_dpsge_04_common = len(archetti_set.intersection(dpsge_04_set))
    archetti_dpsge_17_04_common = len(archetti_set.intersection(dpsge_17_04_set))

    gp_info_gain_common = len(gp_set.intersection(info_gain_set))
    gp_random_forest_common = len(gp_set.intersection(random_forest_set))
    gp_psge_average_common = len(gp_set.intersection(psge_average_set))
    gp_psge_best_train_common = len(gp_set.intersection(psge_best_train_set))
    gp_dpsge_10_10_04_common = len(gp_set.intersection(dpsge_10_10_04_set))
    gp_dpsge_04_common = len(gp_set.intersection(dpsge_04_set))
    gp_dpsge_17_04_common = len(gp_set.intersection(dpsge_17_04_set))


    info_gain_random_forest_common = len(info_gain_set.intersection(random_forest_set))
    info_gain_psge_average_common = len(info_gain_set.intersection(psge_average_set))
    info_gain_psge_best_train_common = len(info_gain_set.intersection(psge_best_train_set))
    info_gain_dpsge_10_10_04_common = len(info_gain_set.intersection(dpsge_10_10_04_set))
    info_gain_dpsge_04_common = len(info_gain_set.intersection(dpsge_04_set))
    info_gain_dpsge_17_04_common = len(info_gain_set.intersection(dpsge_17_04_set))

    random_forest_psge_average_common = len(random_forest_set.intersection(psge_average_set))
    random_forest_psge_best_train_common = len(random_forest_set.intersection(psge_best_train_set))
    random_forest_dpsge_10_10_04_common = len(random_forest_set.intersection(dpsge_10_10_04_set))
    random_forest_dpsge_04_common = len(random_forest_set.intersection(dpsge_04_set))
    random_forest_dpsge_17_04_common = len(random_forest_set.intersection(dpsge_17_04_set))

    psge_best_train_psge_average_common = len(psge_best_train_set.intersection(psge_average_set))
    psge_best_train_dpsge_10_10_04_common = len(psge_best_train_set.intersection(dpsge_10_10_04_set))
    psge_best_train_dpsge_04_common = len(psge_best_train_set.intersection(dpsge_04_set))
    psge_best_train_dpsge_17_04_common = len(psge_best_train_set.intersection(dpsge_17_04_set))

    psge_average_common_dpsge_10_10_04_common = len(psge_average_set.intersection(dpsge_10_10_04_set))
    psge_average_common_dpsge_04_common = len(psge_average_set.intersection(dpsge_04_set))
    psge_average_common_dpsge_17_04_common = len(psge_average_set.intersection(dpsge_17_04_set))

    dpsge_04_common_dpsge_17_04_common = len(dpsge_04_set.intersection(dpsge_17_04_set))
    dpsge_04_common_dpsge_10_10_common = len(dpsge_04_set.intersection(dpsge_10_10_04_set))

    dpsge_17_04_common_dpsge_10_10 = len(dpsge_17_04_set.intersection(dpsge_10_10_04_set))
    print("Number of features:")
    print("Archetti:", len(archetti_set))
    print("GP:", len(gp_set))
    print("Info Gain:", len(info_gain_set))
    print("Random Forest:", len(random_forest_set))
    print("PSGE Average:", len(psge_average_set))
    print("PSGE Best Train:", len(psge_best_train_set))
    print("DPSGE 10 10 04:", len(dpsge_10_10_04_set))
    print("DPSGE 10 10 bug ? 04:", len(dpsge_04_set))
    print("DPSGE 17 04:", len(dpsge_17_04_set))

    print()

    # Print results
    print("Elements in common with empty columns:")
    print("Archetti:", archetti_empty_common)
    print("GP:", gp_empty_common)
    print("Info Gain:", info_gain_empty_common)
    print("Random Forest:", random_forest_empty_common)
    print("PSGE Average:", psge_average_empty_common)
    print("PSGE Best Train:", psge_best_train_empty_common)
    print("DPSGE 10 10 04:", dpsge_10_10_04_empty_common)
    print("DPSGE 10 10 bug ? 04:", dpsge_04_average_empty_common)
    print("DPSGE 17 04:", dpsge_17_04_average_empty_common)

    print()


    print("Elements in common between each pair of lists:")
    print("Archetti - GP:", archetti_gp_common)
    print("Archetti - Info Gain:", archetti_info_gain_common)
    print("Archetti - Random Forest:", archetti_random_forest_common)
    print("Archetti - PSGE Average:", archetti_psge_average_common)
    print("Archetti - PSGE Best Train:", archetti_psge_best_train_common)
    print("Archetti - DPSGE 10 10 04:", archetti_dpsge_10_10_04_common)
    print("Archetti - DPSGE 10 10 bug ? 04:", archetti_dpsge_04_common)
    print("Archetti - DPSGE 17 04:", archetti_dpsge_17_04_common)
    print("GP - Info Gain:", gp_info_gain_common)
    print("GP - Random Forest:", gp_random_forest_common)
    print("GP - PSGE Average:", gp_psge_average_common)
    print("GP - PSGE Best Train:", gp_psge_best_train_common)
    print("GP - DPSGE 10 10 04:", gp_dpsge_10_10_04_common)
    print("GP - DPSGE 10 10 bug ? 04:", gp_dpsge_04_common)
    print("GP - DPSGE 17 04:", gp_dpsge_17_04_common)
    print("Info Gain - Random Forest:", info_gain_random_forest_common)
    print("Info Gain - PSGE Average:", info_gain_psge_average_common)
    print("Info Gain - PSGE Best Train:", info_gain_psge_best_train_common)
    print("Info Gain - DPSGE 10 10 04:", info_gain_dpsge_10_10_04_common)
    print("Info Gain - DPSGE 10 10 bug ? 04:", info_gain_dpsge_04_common)
    print("Info Gain - DPSGE 17 04:", info_gain_dpsge_17_04_common)

    print("Random Forest - PSGE Average:", random_forest_psge_average_common)
    print("Random Forest - PSGE Best Train:", random_forest_psge_best_train_common)
    print("Random Forest - DPSGE 10 10 04:", random_forest_dpsge_10_10_04_common)
    print("Random Forest - DPSGE 10 10 bug ? 04:", random_forest_dpsge_04_common) 
    print("Random Forest - DPSGE 17 04:", random_forest_dpsge_17_04_common)

    print("PSGE Average - PSGE Best Train:", psge_best_train_psge_average_common)
    print("PSGE Average - DPSGE 10 10 04:", psge_average_common_dpsge_10_10_04_common)
    print("PSGE Average - DPSGE 10 10 bug ? 04:", psge_average_common_dpsge_04_common)
    print("PSGE Average - DPSGE 17 04:", psge_average_common_dpsge_17_04_common)

    print("PSGE Best Train - DPSGE 10 10 04:", psge_best_train_dpsge_10_10_04_common)
    print("PSGE Best Train - DPSGE 10 10 bug ? 04:", psge_best_train_dpsge_04_common)
    print("PSGE Best Train - DPSGE 17 04:", psge_best_train_dpsge_17_04_common)

    print("DPSGE 17 04 - DPSGE 10 10: ", dpsge_17_04_common_dpsge_10_10)
    print("DPSGE 17 04 - DPSGE 10 10 bug ? 04:", dpsge_04_common_dpsge_17_04_common)

    print("DPSGE 10 10 bug ? 04 - DPSGE 10 10 04:", dpsge_04_common_dpsge_10_10_common)

if __name__ == "__main__":
    # VARIABLES
    problem = "quartic"
    paths = [
             "standard/1.0/",
            #  "standard/0.5/",
            #  "dependent/gauss_0.001_n_best_4_n_worst_4/1.0/",
            #  "dependent/gauss_0.001_n_best_8_n_worst_8/1.0/",
            #  "dependent/gauss_0.05_n_best_4_n_worst_4/1.0/",
            #  "dependent/gauss_0.05_n_best_4_n_worst_4/0.5/",
            #  "dependent/gauss_0.05_n_best_1_n_worst_1/1.0/",
            # "dependent/gauss_0.05_n_best_1_n_worst_1/0.5/",
            # "dependent/1stversion_n_best_10/1.0/",
            # "dependent/1stversion_n_best_20/1.0/",     
            #  "dependent/mut_0.001_n_best_8_n_worst_8/1.0/",
            #  "dependent/n_best_8_n_worst_8/1.0/",
            #  "dependent/n_best_4_n_worst_4/1.0/",
            #  "dependent/n_best_4_2nd/1.0/",
            "standard_quad_fixed_depthmut/1.0/",
            "dependent_quad_fixed_depthmut/1.0/"

             ]
    
    problem='pagie'
    paths = [
        # "standard_pagie/1.0/",
        # "standard_pagie_fixed_depthmut/1.0/",
        "standard_pagie_fixed_depthmut_fixtorch/1.0/",
        "standard_pagie_fixed_depthmut_new_mut/1.0/",
        # "dependent_pagie_fixed_depthmut/1stversion_n_best_20/1.0/",
        # "dependent_pagie/1stversion_n_best_20/1.0/",
        "dependent_pagie_fixed_depthmut_fixtorch/1.0/",
        "dependent_pagie_fixed_depthmut_fixtorch_new_mut/1.0/",
    #     "standard_pagie/1.0/",
    #     "dependent_pagie/1stversion_n_best_20/0.5/",
    #     "dependent_pagie/1stversion_n_best_10/1.0/", 
    #     "dependent_pagie/1stversion_n_best_1/1.0/", 
    ]
    # comparing mutations
    paths = [
        "dependent_pagie_mut_depth/1stversion_n_best_20/1.0/",
        "dependent_pagie_old_mut_depth/1stversion_n_best_20/1.0/",
    ]
    # new parameters: (standard/dependent pagie = new mutation fix torch)
    paths = [
        ("PSGE", "standard_pagie/1.0/"),
        # "standard_pagie_nbest_20_pmut_01/1.0",
        ("PSGE nbest 20", "standard_pagie_nbest_20/1.0"),
        ("PSGE nbest 10", "standard_pagie_nbest_10/1.0"),
        ("PSGE nbest 5", "standard_pagie_nbest_5/1.0"),
        # "dependent_pagie/1stversion_n_best_20/1.0/",
        # "dependent_pagie/2ndversion_n_best_10_pmut_005/1.0/",
        # "dependent_pagie/2ndversion_n_best_10_pmut_01/1.0/",
        # "dependent_pagie/1stversion_n_best_20/1.0/",
        # "dependent_pagie/2ndversion_n_best_20_pmut_001/1.0/",
        # "dependent_pagie/2ndversion_n_best_20_pmut_005/1.0/",
        # "dependent_pagie/2ndversion_n_best_20_pmut_01/1.0/",
        # "dependent_pagie/2ndversion_n_best_20_pmut_015/1.0/",
        # "dependent_pagie/2ndversion_n_best_30_pmut_001/1.0/",
        # "dependent_pagie/2ndversion_n_best_30_pmut_005/1.0/",
        # "dependent_pagie/2ndversion_n_best_30_pmut_01/1.0/",
        # "dependent_pagie/3rdversion_n_best_20_pmut_01/1.0/",
        # "dependent_pagie/3rdversion_n_best_30_pmut_01/1.0/"
    ]

    # paths = [
    #     "standard_bh/1.0/",
    #     "dependent_bh/1stversion_n_best_20/1.0",
    #     "dependent_bh/2ndversion_n_best_20_pmut_01/1.0",
    #     "dependent_bh/3rdversion_n_best_20_pmut_01/1.0/",
    #     "dependent_bh/3rdversion_n_best_30_pmut_01/1.0/",
    # ]

    paths = [
        ("PSGE LF 1%", "dependent_bh_rrse/standard/1.0/"),
        # "dependent_bh_rrse/standard_plus_3rdversion_pmut_01/1.0/",
        # "dependent_bh_rrse/2ndversion_n_best_20_pmut_01/1.0/",
        # "dependent_bh_rrse/2ndversion_n_best_20_pmut_01/0.5/",
        # "dependent_bh_rrse/2ndversion_n_best_20_pmut_01/1.5/",
        # "dependent_bh_rrse/3rdversion_n_best_10_pmut_01/0.5/",
        ("DPSGE 3rd nbest 10 pmut 01 LF 1%", "dependent_bh_rrse/3rdversion_n_best_10_pmut_01/1.0/"),
        # "dependent_bh_rrse/3rdversion_n_best_20_pmut_01/1.0/",
        # "dependent_bh_rrse/3rdversion_n_best_50_pmut_01/1.0/",
    ]

    problem='bostonhousing'
    # paths = [
    #     "standard_bh_fixed_depthmut_fixtorch_new_mut/1.0/",
    #     # "dependent_bh_fixed_depthmut_fixtorch_new_mut/1stversion_n_best_20/1.0/",
    #     # "dependent_bh_fixed_depthmut_fixtorch/1stversion_n_best_20/1.0/",
    #     # "standard_bh_fixed_depthmut_fixtorch/1.0/"
    # #     "standard_bh/1.0/",
    # #     "dependent_bh/1stversion_n_best_20/1.0/",
    # ]

    problem="bioavailability"
    paths = [
        ("SGE", "sge_bioav/depth_17/"),
        ("PSGE", "standard_bioav/depth_17/1.0/"),
        ("DPSGE 3rd nbest 10 pmut 01", "dependent_bioav/depth_17/3rdversion_n_best_10_pmut01/1.0/")
    ]

    problem="bioavailability"
    paths = [
        # ("SGE 10 10", "sge_bioav/"),
        ("SGE 6 17", "sge_bioav/depth_17/"),
        # ("PSGE 10 10", "standard_bioav/1.0/"),
        ("PSGE 6 17", "standard_bioav/depth_17/1.0/"),
        # ("DPSGE 10 10 old", "dependent_bioav/3rdversion_n_best_10_pmut01/1.0/"),
        # ("DPSGE 10 10", "dependent_bioav/depth_10_10/3rdversion_n_best_10_pmut01/1.0/"),
        ("DPSGE 6 17", "dependent_bioav/depth_17/3rdversion_n_best_10_pmut01/1.0/"),
        # ("DPSGE PPB BIOAV", "dependent_bioav_how/depth_10_10/3rdversion_n_best_10_pmut01/1.0/")
    ]   

    # probabilities_visualizer_mean_folders(problem, "dependent_bioav/3rdversion_n_best_10_pmut01/1.0/", 499)
    # probabilities_visualizer_mean_folders(problem, "dependent_bioav/depth_10_10/3rdversion_n_best_10_pmut01/1.0/", 499)

    # probabilities_visualizer_mean_folders(problem, "standard_bioav/1.0/", 499)
    # probabilities_visualizer_mean_folders(problem, "standard_bioav/depth_17/1.0/", 499)
    # probabilities_visualizer_mean_folders(problem, "dependent_bioav/3rdversion_n_best_10_pmut01/1.0/", 499)
    # probabilities_visualizer_mean_folders(problem, "dependent_bioav/depth_17/3rdversion_n_best_10_pmut01/1.0/", 499)

    # bioav_feature_comparison()
    # probabilities(problem, paths)

    # problem="ld50"
    # paths = [
    #     ("SGE", "sge_ld50/"),
    #     ("PSGE", "standard_ld50/1.0/"),
    #     ("DPSGE 3rd nbest 10 pmut 01", "dependent_ld50/3rdversion_n_best_10_pmut01/1.0/")
    # ]

    # problem="ppb"
    # paths = [
    #     ("SGE", "sge_ppb/"),
    #     ("PSGE", "standard_ppb/1.0/"),
    #     ("DPSGE 3rd nbest 10 pmut 01", "dependent_ppb/3rdversion_n_best_10_pmut01/1.0/")
    # ]

    performance(problem, paths)

    # bioav_feature_comparison()
    # analysis( [("PSGE","standard_bioav/1.0/")])


    # probabilities(problem, paths)
    # performance(problem, paths)
# "dependent_bioav/3rdversion_n_best_10_pmut01/1.0/"
    # probabilities_visualizer_mean_folders(problem, "standard_bioav/1.0/", 499)
    # probabilities_visualizer_mean_folders(problem, "dependent_bioav/3rdversion_n_best_10_pmut01/1.0/", 499)


    # data = read_data("dependent_sggp_final/8_sgg_mutation_gauss/1.0/")
    # avg = get_mean_std(data)
    # plot_data(data, avg)
    # plot_performance("quad",
    #                  "standard/1.0/",
    #                 #  "dependent/1/1.0/",  # funcao que considera divisao pela soma das expansoes daquela depth
    #                 #  "dependent/10/1.0/",
    #                  "dependent/20/1.0/",
    #                 #  "dependent/100/1.0/", 
    #                  "dependent_new/10/1.0/", # funcao inspirada pelo sggp sem considerar os maus
    #                 #  "dependent_new/20/1.0/",
    #                 #  "dependent_sggp/10/1.0/", # inspirado pelo sggp, considera tambem os maus, mas usa o numero de vezes que a regra foi expandida em vez se foi ou nao expandida por aquele
    #                  "dependent_sggp/4/1.0/",
    #                  "dependent_sggp/4_sgg/1.0/", # sggp usa só numero de invidisuos que expandiram aquela regra
    #                  "dependent_sggp/10_sgg/1.0/",
    #                  "dependent_sggp/4_sgg_mutation/1.0/", # SGGP mutation, p_mut = 0.001, amplitude = 0.01
    #                  "dependent_sggp/4_sgg_mutation_gauss/1.0/", # SGGP mutation with gauss 0.05
    #                  )
