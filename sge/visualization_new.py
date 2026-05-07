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
    Performance analysis
"""

# FUNçÂO QUE LEIA OS PROGRESS REPORT DE TODAS AS RUNS

def read_data(path, exp_name):
    print(path)
    print(exp_name)
    folders = os.listdir(path)
    if "aggregated_data.csv" in folders:
        return pd.read_csv(os.path.join(path,"aggregated_data.csv"), delimiter='\t', header=0, na_values="nan")
    else:
        data = []
        for folder in folders:
            if "run" not in folder:
                continue
            file_path = os.path.join(path, folder, "progress_report.csv")
            column_names = ['generation','best_fit','mean_fit','std_fit','best_test','mean_test','std_test','best_tree_depth','mean_depth','median_depth','best_length','mean_length','median_length','unique_percentage']
            df = pd.read_csv(file_path, delim_whitespace=False, sep = '\t', names=column_names, na_values="nan")
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

        ax = sns.lineplot(data=df, x='generation', y='best_test', hue='algorithm', estimator='mean', errorbar=None)
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
    boxplot(problem, df, 150, display=False)
    boxplot(problem, df, 300, display=False)

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

if __name__ == "__main__":

    problem="pagie"
    paths=[
        # ("PSGE", "standard/pagie_torch/"),
        ("PSGE", "/media/cdv/nvme980pro/jessica/search_strategy/results/standard/pagie/1.0/"),
        ("EDA PSGE no elitism", "/media/cdv/nvme980pro/jessica/search_strategy/results/eda_no_elitism/pagie/1.0/"),
        ("EDA PSGE elitism remap", "/media/cdv/nvme980pro/jessica/search_strategy/results/eda_elitism_remap/pagie/1.0/"),
        ("EDA PSGE elitism remap 10 best", "/media/cdv/nvme980pro/jessica/search_strategy/results/eda_elitism_remap_update_10_best/pagie/1.0/"),


    ]
    # performance(problem, paths)

    problem="5parity"
    paths=[
        # ("PSGE", "standard/5parity_torch/"),
        ("PSGE 2", "/media/storage/jessica/search_strategy/results/standard/5parity/1.0/"),
        ("PSGE", "/media/storage/jessica/search_strategy/results/standard_/5parity/1.0/"),
        ("PSGE fix mapping", "/media/storage/jessica/search_strategy/results/standard_fixed_mapping/5parity/1.0/"),
        ("PSGE alternate", "/media/storage/jessica/search_strategy/results/standard_alternate_bests/5parity/1.0/"),
        ("EDA PSGE no elitism", "/media/storage/jessica/search_strategy/results/eda_no_elitism/5parity/1.0/"),
        ("EDA PSGE elitism remap", "/media/storage/jessica/search_strategy/results/eda_elitism_remap/5parity/1.0/"),
        ("EDA PSGE elitism remap 10 best", "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_10_best/5parity/1.0/"),
        ("EDA PSGE elitism remap 500 best 1.0", "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_500_best/5parity/1.0/"),
        ("EDA PSGE elitism remap 500 best 5.0", "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_500_best/5parity/5.0/"),

    ]

    paths=[
        ("PSGE", "/media/storage/jessica/search_strategy/results/standard_fixed_mapping/5parity/1.0/"),
        ("EDA PSGE 1.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/5parity/1.0/"),
        ("EDA PSGE 1.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/5parity/1.0/"),
        ("EDA PSGE 1.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/5parity/1.0/"),
        ("EDA PSGE 5.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/5parity/5.0/"),
        ("EDA PSGE 5.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/5parity/5.0/"),
        ("EDA PSGE 5.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/5parity/5.0/"),

    ]

    # performance(problem, paths)

    problem="quartic"
    paths=[
        # ("PSGE", "standard/5parity_torch/"),
        ("PSGE", "/media/storage/jessica/search_strategy/results/standard_/quartic/1.0/"),
        ("PSGE alternate", "/media/storage/jessica/search_strategy/results/standard_alternate_bests/quartic/1.0/"),
        ("EDA PSGE elitism remap 500 best 1.0", "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_500_best/quartic/1.0/"),
        ("EDA PSGE elitism remap 500 best 5.0", "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_500_best/quartic/5.0/"),
        ("EDA PSGE elitism remap 1 best 1.0", "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_1_best/quartic/1.0/"),

    ]


    paths=[
        # ("PSGE fix mapping", "/media/storage/jessica/search_strategy/results/standard_fixed_mapping/5parity/1.0/"),
        ("PSGE", "/media/storage/jessica/search_strategy/results_better_map/standard/quartic/1.0/"),
        ("EDA PSGE 1.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/quartic/1.0/"),
        ("EDA PSGE 1.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/quartic/1.0/"),
        ("EDA PSGE 1.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/quartic/1.0/"),
        ("EDA PSGE 5.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/quartic/5.0/"),
        ("EDA PSGE 5.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/quartic/5.0/"),
        ("EDA PSGE 5.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/quartic/5.0/"),

    ]
    
    # performance(problem, paths)


    problem="pagie"
    paths=[
        ("PSGE", "/media/storage/jessica/search_strategy/results_better_map/standard/pagie/1.0/"),
        ("EDA PSGE 1.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/pagie/1.0/"),
        ("EDA PSGE 1.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/pagie/1.0/"),
        ("EDA PSGE 1.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/pagie/1.0/"),
        ("EDA PSGE 5.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/pagie/5.0/"),
        ("EDA PSGE 5.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/pagie/5.0/"),
        ("EDA PSGE 5.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/pagie/5.0/"),

    ]



    problem="pagie"
    paths=[
        # ("PSGE", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/standard/pagie/1.0/"),
        ("PSGE 1.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest1_elitism/pagie/1.0/"),
        ("PSGE 1.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest100/pagie/1.0/"),
        ("PSGE 1.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest500/pagie/1.0/"),
        ("PSGE 5.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest1_elitism/pagie/5.0/"),
        ("PSGE 5.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest100/pagie/5.0/"),
        ("PSGE 5.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest500/pagie/5.0/"),

    ]


    problem="5parity"
    paths=[
        ("PSGE ?", "/media/storage/jessica/search_strategy/results/standard/5parity/1.0/"),
        ("PSGE online", "/media/storage/jessica/search_strategy/results/standard_alternate_bests/5parity/1.0/"),
        ("PSGE optimized", "/media/storage/jessica/search_strategy/results/standard_fixed_mapping/5parity/1.0/"),
        # ("PSGE 1.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest100/pagie/1.0/"),
        # ("PSGE 1.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest500/pagie/1.0/"),
        # ("PSGE 5.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest1_elitism/pagie/5.0/"),
        # ("PSGE 5.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest100/pagie/5.0/"),
        # ("PSGE 5.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/pagie no eda/remap_nbest500/pagie/5.0/"),

    ]



    problem="koza2"
    paths=[
        ("PSGE", "/media/storage/jessica/search_strategy/results_better_map/standard/koza2/1.0/"),
        ("EDA PSGE 1.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/koza2/1.0/"),
        ("EDA PSGE 1.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/koza2/1.0/"),
        ("EDA PSGE 1.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/koza2/1.0/"),
        ("EDA PSGE 5.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/koza2/5.0/"),
        ("EDA PSGE 5.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/koza2/5.0/"),
        ("EDA PSGE 5.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/koza2/5.0/"),

    ]

    problem="nguyen5"
    paths=[
        ("PSGE", "/media/storage/jessica/search_strategy/results_better_map/standard/nguyen5/1.0/"),
        ("EDA PSGE 1.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/nguyen5/1.0/"),
        ("EDA PSGE 1.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/nguyen5/1.0/"),
        ("EDA PSGE 1.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/nguyen5/1.0/"),
        ("EDA PSGE 5.0 1 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest1_elitism/nguyen5/5.0/"),
        ("EDA PSGE 5.0 100 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest100/nguyen5/5.0/"),
        ("EDA PSGE 5.0 500 best", "/media/storage/jessica/search_strategy/results_better_map/eda_remap_nbest500/nguyen5/5.0/"),

    ]

    performance(problem, paths)