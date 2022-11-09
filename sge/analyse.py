import json
from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors


plt.ioff()


def load_population(path):
    with open(path, 'r') as f:
        population = json.load(f)
    return population

params = [
	(0.75, 0.0025),
	(0.75, 0.005),
	(0.75, 0.01),
	(1.0, 0.0025),
	(1.0, 0.005),
	(1.0, 0.01),
]
tab_colors = list(mcolors.TABLEAU_COLORS)
alls = []

base = "psge"
def analyse(paths, bases, epochs, grammar_sizes, label_lists, label_indexes, setting, colors, fig_name='plot'):
    #For each experiment
    for path, base, grammar_size, labels, label_index, colors_r in zip(paths, bases, grammar_sizes, label_lists, label_indexes, colors):
        mut_prob = [] #Mutation probabilities per generation
        fit = []
        averages = []
        std = []
        max = []
        min = []
        print(path)
        #For each generation
        for it in range(epochs):
            #Create mutation probability lists for this generation
            mut_prob.append([])
            for rule in range(grammar_size):
                mut_prob[-1].append([])
            fit.append([])
            #For each run
            for run in range(1,31):
                #try:
                folder = os.path.join(path, 'run_' + str(run), 'iteration_' + str(it) + '.json')
                pop = load_population(folder)
                pop = pop[1:]
                #print(f"Number of rules:  {len(pop[0]['mutation_prob'])}/{len(label_index)}")
                #For each grammar non terminal selected for visualization
                for rule in label_index:
                    if setting == "best":
                            #print(pop[0]['mutation_prob'], rule)
                        rule_mutation_probability = pop[0]['mutation_prob'][rule]
                        mut_prob[it][rule].append(rule_mutation_probability)
                        #pop_fitness = pop[0]['fitness']
                        #fit[it].append(pop_fitness)
                    elif setting == "average":
                        #print(it, rule, mut_prob) 
                        foo = [x['mutation_prob'][rule] for x in pop]
                        mut_prob[it][rule].append(np.average(foo))
                        #pop_fitness = np.average([x['fitness'] for x in pop])
                        #fit[it].append(pop_fitness)
                    elif setting == "top10":
                        pop = pop[:10]
                        #print(it, rule, mut_prob) 
                        foo = [x['mutation_prob'][rule] for x in pop]
                        #print(foo)
                        mut_prob[it][rule].append(np.average(foo))
                        #pop_fitness = np.average([x['fitness'] for x in pop])
                        #fit[it].append(pop_fitness)

                    #except:
                    #    print(folder + " error")
                    #    pass

        print(len(mut_prob), len(mut_prob[0]), len(mut_prob[0][label_index[0]]))


        for x in range(grammar_size):
            averages.append([])
            std.append([])
            #max.append([])
            #min.append([])
        for i in label_index:
            for it in range(len(mut_prob)):
                #print(mutit, i)
                foo = np.average(mut_prob[it][i])
                #print(len(mut_prob[it][i]))
                averages[i].append(foo)
                #print(averages)
                foo = np.std(mut_prob[it][i])
                std[i].append(foo)
                #max[i].append(np.max(mut_prob[it][i]))
                #min[i].append(np.min(mut_prob[it][i]))

        fig = plt.figure()
        for rule, label, color in zip(label_index, labels, colors_r):
            print(f"Plotting {len([x for x in range(epochs)])}, {len(averages[rule])}")
            plt.plot([x for x in range(epochs)], averages[rule], label = label, color=tab_colors[rule])
            plt.ylim([0.085, 0.11])
            #plt.fill_between(
            #    [x for x in range(epochs)],
            #    max[rule],
            #    min[rule], alpha=0.2, color=tab_colors[color])

        plt.title(f"{fig_name}")
        #plt.legend(loc='center left', bbox_to_anchor=(1,0.5))	
        plt.legend()	
        print(f"Saving fig {path}/{fig_name}.png")
        plt.savefig(f'{path}/{fig_name}.png')
        print("Fig is closed")
        plt.close(fig)

        #print(f"mut_prob: {param[0]} delta_prob: {param[1]} fit: {np.average(fit)}\n{averages}\n{std}\n")
        #alls.append([path, np.average(fit), averages, std])
    #alls.sort(key= lambda x: x[1])
    #for config in alls:
    #	print(f"[{config[0]},{config[1]}] {config[2]} {config[3]} {config[4]}")	

labels_extended = [
    ("start", 0, 0), 
    ("expr_vs_var", 1, 1), 
    ("expr", 2, 2),
    ("expr_op", 3, 3),
    ("op", 4, 4),
    ("pre_op", 5, 5),
    ("trig_op", 6, 6),
    ("exp_log_op", 7, 7),
    ("var", 8, 8),
]
labels_standard = [
	("start", 0, 0), 
    ("expr", 1, 1),
    ("op", 2, 2),
    ("pre_op", 3, 3),
    ("var", 4, 4),
]
analyse(
    paths=[
    "/home/jessica/mut_level/psge/sge/mutation_level_bh/prob_mut_1.0_gauss_sd_0.001/1.0/",
    "/home/jessica/mut_level/psge/sge/mutation_level_bh_extended/prob_mut_1.0_gauss_sd_0.001/1.0/",
	],
    bases=[
        'psge', 
        'psge', 
        'psge', 
        'psge'
        ],
    epochs=201,
    grammar_sizes=[
        5,
        10,
    ],
    label_lists=[
        [x[0] for x in labels_standard],
        [x[0] for x in labels_extended]
    ],
    label_indexes=[
        [x[1] for x in labels_standard],
        [x[1] for x in labels_extended]       
    ],
    setting = "best",
    colors=[
        [x[2] for x in labels_standard],
        [x[2] for x in labels_extended]
    ],
    fig_name="all"
)
