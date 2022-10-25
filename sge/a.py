import json
from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import os

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

alls = []

base = "psge"
def analyse(paths, bases, epochs, grammar_sizes, label_lists, label_indexes):
    #For each experiment
    for path, base, grammar_size, labels, label_index in zip(paths, bases, grammar_sizes, label_lists, label_indexes):
        mut_prob = [] #Mutation probabilities per generation
        fit = []
        setting = "best"
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
                if setting == "best":
                    #For each grammar non terminal selected for visualization
                    for rule in label_index:
                        #print(pop[0]['mutation_prob'], rule)
                        rule_mutation_probability = pop[0]['mutation_prob'][rule]
                        mut_prob[it][rule].append(rule_mutation_probability)
                    pop_fitness = pop[0]['fitness']
                    fit[it].append(pop_fitness)
                elif setting == "average":
                    for rule in label_index:
                        #print(it, rule, mut_prob) 
                        foo = [x['mutation_prob'][rule] for x in pop]
                        #print(foo)
                        mut_prob[it][rule].append(np.average(foo))
                    pop_fitness = np.average([x['fitness'] for x in pop])
                    fit[it].append(pop_fitness)
                    #except:
                    #    print(folder + " error")
                    #    pass

        print(len(mut_prob), len(mut_prob[0]), len(mut_prob[0][label_index[0]]))


        for x in range(grammar_size):
            averages.append([])
            std.append([])
            max.append([])
            min.append([])
        for i in label_index:
            for it in range(len(mut_prob)):
                #print(mutit, i)
                foo = np.average(mut_prob[it][i])
                #print(len(mut_prob[it][i]))
                averages[i].append(foo)
                #print(averages)
                foo = np.std(mut_prob[it][i])
                std[i].append(foo)
                max[i].append(np.max(mut_prob[it][i]))
                min[i].append(np.min(mut_prob[it][i]))

        fig = plt.figure()
        for rule, label in zip(label_index, labels):
            print(f"Plotting {len([x for x in range(epochs)])}, {len(averages[rule])}")
            plt.plot([x for x in range(epochs)], averages[rule], label = label)
            plt.fill_between(
                [x for x in range(epochs)],
                max[rule],
                min[rule], alpha=0.2)

        plt.title(f"{path}")
        plt.legend()	
        print("Saving fig")
        plt.savefig(f'{path}.png')
        print("Fig is closed")
        plt.close(fig)

        #print(f"mut_prob: {param[0]} delta_prob: {param[1]} fit: {np.average(fit)}\n{averages}\n{std}\n")
        #alls.append([path, np.average(fit), averages, std])
    #alls.sort(key= lambda x: x[1])
    #for config in alls:
    #	print(f"[{config[0]},{config[1]}] {config[2]} {config[3]} {config[4]}")	

labels_extended = [
    #("start", 0), 
    #("expr_vs_var", 1), 
    #("expr", 2),
    ("expr_op", 3),
    #("op", 4),
    #("pre_op", 5),
    #("trig_op", 6),
    ("exp_log_op", 7),
    #("var", 8),
    #("var_x", 9),
]
analyse(
    paths=[
        '/home/jessica/psge/sge/mutation_level_extended/prob_mut_1.0_gauss_sd_0.0025/1.0/',
        #'/home/jessica/psge/sge/mutation_level_extended/standard/1.0/',
        #'/home/jessica/psge/sge/mutation_level_old_gram/prob_mut_1.0_gauss_sd_0.0025/1.0/',
        #'/home/jessica/psge/sge/mutation_level_old_gram/standard/1.0/',
],
    bases=['psge', 'psge', 'psge', 'psge'],
    epochs=101,
    grammar_sizes=[
        10,
        #10,
       # 5,
        #5
    ],
    label_lists=[
        [x[0] for x in labels_extended]
    ],
    label_indexes=[
        [x[1] for x in labels_extended]       
    ],
)

"""
       if base == "copsge":
            mut_prob = []
            fit = []

            print(path)
            for it in range(epochs):
                mut_prob.append([[], [], [], [], []])
                fit.append([])
                for run in range(1,31):
                    try:
                        folder = path + 'run_' + str(run) + '/iteration_' + str(it) + '.json'
                        pop = load_population(folder)

                        mut_prob[it][0].append(np.average([x['mutation_prob'][0] for x in pop]))
                        mut_prob[it][1].append(np.average([x['mutation_prob'][1] for x in pop]))
                        mut_prob[it][2].append(np.average([x['mutation_prob'][2] for x in pop]))
                        mut_prob[it][3].append(np.average([x['mutation_prob'][3] for x in pop]))
                        mut_prob[it][4].append(np.average([x['mutation_prob'][4] for x in pop]))
                        
                        pop_fitness = np.average([x['fitness'] for x in pop])
                        fit[it].append(pop_fitness)
                    except:
                        print('mutation_level/prob_mut_' + str(param[0]) + '_gauss_sd_' + str(param[1]) + '/1.0/run_' + str(run) + '/iteration_' + str(it) + '.json')
                        pass

            print(len(mut_prob), len(fit))

            averages = [[], [], [], [], []]
            std = [[], [], [], [], []]
            for i in range(5):
                for it in range(len(mut_prob)):
                    foo = np.average(mut_prob[it][i])
                    averages[i].append(foo)
                    #print(averages)
                    foo = np.std(mut_prob[it][i])
                    std[i].append(foo)
            fig = plt.figure()
            plt.plot([x for x in range(epochs)], averages[0], label = "start")
            plt.plot([x for x in range(epochs)], averages[1], label = "expr")
            plt.plot([x for x in range(epochs)], averages[2], label = "op")
            plt.plot([x for x in range(epochs)], averages[3], label = "preop")
            plt.plot([x for x in range(epochs)], averages[4], label = "var")
            plt.title(f"mut_prob {param[0]} mut gauss {param[1]}")
            plt.legend()	
            plt.savefig(f'{param}.png')
            plt.close(fig)
            #print(f"mut_prob: {param[0]} delta_prob: {param[1]} fit: {np.average(fit)}\n{averages}\n{std}\n")
            alls.append([param[0], param[1], np.average(fit), averages, std])
        else:
"""