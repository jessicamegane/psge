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

def load_dataframe(path, run_file):
    run = re.search(r'\d+', run_file).group()
    df = pd.DataFrame()    
    iteration_paths = os.listdir(os.path.join(path, run_file))
    for iteration in iteration_paths:
        if "iteration" not in iteration:
            # exclude parameters and progress report
            continue
        iteration_number = re.search(r'\d+', iteration).group()
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

def read_probabilities(path):
    # processar valores
    folders = os.listdir(path)
    if "aggregated_probabilities.csv" in folders:
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
                new_df = load_dataframe(os.path.join(path, folder), f)
                new_df['fold'] = [fold] * len(new_df)
                if df.empty:
                    df = new_df
                else:
                    df = pd.concat([df, new_df])
        else:
            f = folder
            if "run" not in f:
                continue
            new_df = load_dataframe(path, f)
            if df.empty:
                df = new_df
            else:
                df = pd.concat([df, new_df])

    df.to_csv(path + 'aggregated_probabilities.csv', index=False, sep='\t', header=True)
    return df

def read_grammar_as_labels(problem):
    grammars = {"quartic": "grammars/quad_regression.pybnf", "pagie": "grammars/regression.pybnf", "bostonhousing": "grammars/bostonhousing.bnf"}
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
        for path in paths:
            print(path)
            data = read_probabilities(path)
            non_terminals, grammar = read_grammar_as_labels(problem)
            if 'depth' in data.columns:
                plot_probabilities_depth(path, data, non_terminals, grammar, display=False)
            else:
                plot_probabilities(path, data, non_terminals, grammar, display=False)


"""
    Performance analysis
"""

# FUNçÂO QUE LEIA OS PROGRESS REPORT DE TODAS AS RUNS

def read_data(path):
    
    folders = os.listdir(path)
    if "aggregated_data.csv" in folders:
        return pd.read_csv(os.path.join(path,"aggregated_data.csv"), delim_whitespace=True, header=0, na_values="nan")
    else:
        data = []
        for folder in folders:
            if "run" not in folder:
                continue
            file_path = os.path.join(path, folder, "progress_report.csv")
            column_names = ['generation','best_fit','mean_fit','std_fit','best_test','tree_depth','mean_depth','median_depth']
            df = pd.read_csv(file_path, delim_whitespace=False, sep = ';', names=column_names, na_values="nan")
            run = re.search(r'\d+', folder).group()
            df['run'] = [run] * (len(df.index))
            df['algorithm'] = [path] * (len(df.index))

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
    ax.set(xlabel='Algorithm', ylabel='Depth', title=f'Boxplot for {problem} problem - Generation {generation}')

    plt.savefig(f'{problem}_boxplot_depth_{generation}.png')

    if display:
        plt.show()


def performance_plot(problem, df, display=False):
    f1 = plt.figure()
    ax = sns.lineplot(data=df, x='generation', y='best_fit', hue='algorithm', estimator='mean', errorbar='sd')
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

        ax = sns.lineplot(data=df, x='generation', y='best_test', hue='algorithm', estimator='mean', errorbar='sd')
        ax.set(xlabel='Generations', ylabel='Error', title=f'Performance for {problem} problem - Test')
        ax.legend(title='Algorithms')
        algorithms=['Standard', 'Dependent']
        for i in range(len(algorithms)):
            ax.legend_.texts[i].set_text(algorithms[i])
        plt.savefig(f'{problem}_performance_plot_test.png')

        if display:
            plt.show()

def performance(problem, paths):
    df = pd.DataFrame()

    for path in paths:
        data = read_data(path)
        # performance_plot_runs(problem, data, path, display=False)
        if df.empty:
            df = data
        else:
            df = pd.concat([df, data], axis=0)
    
    performance_plot(problem, df, display=False)
    boxplot(problem, df, 200, display=False)
    boxplot(problem, df, 300, display=False)
    boxplot_depth(problem, df, 10, display=False)
    boxplot_depth(problem, df, 50, display=False)
    boxplot_depth(problem, df, 100, display=False)
    boxplot_depth(problem, df, 200, display=False)
    boxplot_depth(problem, df, 300, display=False)
    

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
    paths = [
        "dependent_pagie_mut_depth/1stversion_n_best_20/1.0/",
        "dependent_pagie_old_mut_depth/1stversion_n_best_20/1.0/",
    ]

    # problem='bostonhousing'
    # paths = [
    #     "standard_bh_fixed_depthmut_fixtorch_new_mut/1.0/",
    #     "dependent_bh_fixed_depthmut_fixtorch_new_mut/1stversion_n_best_20/1.0/",
    #     # "dependent_bh_fixed_depthmut_fixtorch/1stversion_n_best_20/1.0/",
    #     # "standard_bh_fixed_depthmut_fixtorch/1.0/"
    # #     "standard_bh/1.0/",
    # #     "dependent_bh/1stversion_n_best_20/1.0/",
    # ]

    
    performance(problem, paths)

    # probabilities(problem, paths)
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
