import sge.grammar as grammar
import sge.grammar as grammar_dpsge
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
# from sklearn.manifold import TSNE
import pandas as pd
import re
import pickle

from openTSNE import TSNE

def get_token(sym):
    for nt_pos, key in enumerate(grammar.get_non_terminals()):
        options = grammar.get_dict()[key]
        for option_pos, option in enumerate(options):
            if option[0][0] == sym:
                return nt_pos * 100 + option_pos


def embeddings_total(grammar_func, path, embbedding, max_size, fitness_values, runs, iterations, algorithm, algo):
    folders = os.listdir(path)

    for run_folder in folders:
        
        if "run" not in run_folder:
            continue
        run = re.search(r'\d+', run_folder).group()
        iteration_paths = os.listdir(os.path.join(path,run_folder))

        for iteration_file_name in iteration_paths:
            if "iteration" not in iteration_file_name:
                continue
            iteration_number = re.search(r'\d+', iteration_file_name).group()
            iteration_file = os.path.join(path, run_folder, iteration_file_name)
            if not os.path.isfile(iteration_file):
                continue

            population = json.load(open(iteration_file))
            # best_ind = population[0]
            # mapping_values = [0 for _ in best_ind['genotype']]
            # fitness_values += [best_ind['fitness']]
            iterations += [iteration_number]
            runs += [run]
            algorithm += [algo]

            if algo == "SGE":
                best_ind = population[0]
                mapping_values = [0 for _ in best_ind['genotype']]
                fitness_values += [best_ind['fitness']]
                phen_array, tree_depth = grammar_sge.mapping_with_array(best_ind['genotype'], mapping_values)
            elif algo == "Co-PSGE":
                best_ind = population[0]
                mapping_values = [0 for _ in best_ind['genotype']]
                fitness_values += [best_ind['fitness']]
                phen_array, tree_depth = grammar.mapping_with_array(best_ind['genotype'], np.array(best_ind['pcfg']), mapping_values)
            elif algo == "PSGE":
                gram = population[0]['grammar']
                best_ind = population[1]
                mapping_values = [0 for _ in best_ind['genotype']]
                fitness_values += [best_ind['fitness']]
                g = []
                for nt_list in gram:
                    l = []
                    for depth in nt_list:
                        l.append(np.array(depth))
                    g.append(l)
                grammar_func.set_pcfg(g)
                phen_array, tree_depth = grammar_func.mapping_with_array(best_ind['genotype'], mapping_values)
            elif algo == "DPSGE":
                gram = population[0]['grammar']
                best_ind = population[1]
                mapping_values = [0 for _ in best_ind['genotype']]
                fitness_values += [best_ind['fitness']]
                g = []
                for nt_list in gram:
                    l = []
                    for depth in nt_list:
                        l.append(np.array(depth))
                    g.append(l)
                grammar_func.set_pcfg(g)
                # TODO: arranjar esta funcao a ver se funciona com a depth
                phen_array, tree_depth = grammar_func.mapping_with_array(best_ind['genotype'], mapping_values)
            
            emb = list(filter(lambda x: x is not None, 
                    map(get_token, phen_array)
                    )
                    )
            max_size = max(max_size, len(emb))

            embbedding.append(np.array(emb))

    # new_values = embbedding_train.transform(embbedding.reshape(1, -1))
    # algorithm += [algo] * len(fitness_values)
    return embbedding, fitness_values, iterations, runs, algorithm, max_size


def treino_tsne():
    path = "dependent_bh_rrse/Treino/1.0/run_1/"
    if os.path.isfile(path+"embbedding_train.sav"):
        embbedding_train = pickle.load(open(path + "embbedding_train.sav","rb"))
        max_size = 2194
        return embbedding_train, max_size

    if os.path.isfile(os.path.join(path,'train_data.json')):
        path_file = os.path.join(path,'train_data.json')
        with open(path_file) as f:
            data = json.load(f)
            max_size = data['max_size']
            embbedding = np.asarray(data['embbedding'])

    else:
        population = json.load(open(os.path.join(path,'iteration_0.json')))
        embbedding = []
        max_size = 0
        fitness_values = []
        for ind in population[1:]:
            mapping_values = [0 for _ in ind['genotype']]
            fitness_values +=  [ind['fitness']]
            phen_array, tree_depth = grammar.mapping_with_array(ind['genotype'], mapping_values)
            emb = list(filter(lambda x: x is not None, 
                    map(get_token, phen_array)
                    )
                    )
            max_size = max(max_size, len(emb))
            embbedding.append(np.array(emb))
        temp = np.zeros((len(embbedding), max_size))
        for i,e in enumerate(embbedding):
            temp[i, : e.shape[0]] = e

        embbedding = temp

        with open(path+'train_data.json','w') as f:
            data = {'embbedding': embbedding.tolist(), 'max_size': max_size}
            json.dump(data, f)

    # tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=23)
    # new_values = tsne_model.fit_transform(embbedding)
    tsne_model = TSNE(perplexity=50, n_components=2, initialization='pca', n_iter=5000, random_state=23)
    embbedding_train = tsne_model.fit(embbedding)

    pickle.dump(embbedding_train, open(path + "embbedding_train.sav","wb"))

    return embbedding_train, max_size


def visualize(new_values, fitness_values, iterations, runs, algorithm, csv_file):
    
    import plotly.express as px
    import pandas as pd

    sns.scatterplot(x = new_values[:, 0], y = new_values[:, 1], size=fitness_values)

    plt.show()
    
    fitness_values = list(map(lambda x: 100 if x > 100 else x, fitness_values))
    data = {'p1' : new_values[:, 0],
            'p2' : new_values[:, 1], 
            'iteration': iterations,
            'run': runs,
            'algorithm': algorithm,
            'error' : fitness_values}
    df = pd.DataFrame(data)

    df.to_csv(csv_file, index=False, sep=';', header=True)
    # fig = px.scatter(df, x="p1", y="p2", color="error",
    #                 hover_data=['error', 'iteration', 'fold', 'algorithm'])
    fig = px.scatter(df, x="p1", y="p2", color="error",
                    hover_data=['error', 'iteration', 'algorithm'], symbol=df['algorithm'])

    fig.show()

def visualize_pandas(df, problem):
    import plotly.express as px
    import pandas as pd

    sns.scatterplot(x = df['p1'], y = df['p2'], size=df['error'])
    plt.show()
    df = df[df['iteration'] == 400]
    fig = px.scatter(df, x="p1", y="p2", color="error",
                    hover_data=['error', 'iteration', 'algorithm'], symbol=df['algorithm'], title=problem)
    # fig = px.scatter(df_sge, x="p1", y="p2", color="error",
    #                 hover_data=['error', 'iteration', 'fold', 'algorithm'])
    # fig.update_traces(marker=dict(symbol="diamon-open"))
    # fig = px.scatter(df_psge, x="p1", y="p2", color="error",
    #                 hover_data=['error', 'iteration', 'fold', 'algorithm'])
    # fig.update_traces(marker=dict(symbol="circle-open"))

    fig.show()



if __name__ == '__main__':
    # folder_sge = "/media/cdv/Sistema Reservado/jessica/__GlucoseAbsys/sge_again/Horizon120"
    # folder_psge =  "/media/cdv/Sistema Reservado/jessica/dependency/2ndversion_n_best_3_pmut_01/1.0/Horizon120"
    # folder =  "dependent_bh_rrse/3rdversion_n_best_10_pmut_01/1.0/"
    # folder_standard = "dependent_bh_rrse/standard/1.0/"
    folder =  "dependent_pagie/3rdversion_n_best_20_pmut_01/1.0/"
    folder_standard = "standard_pagie/1.0/"
    problem = "Pagie"
    embbedding_train, max_size = treino_tsne()
    print("Modelo Treinado")
    
    # csv_file = "data_trainned_sge_psge.csv"
    csv_file = "data_trainned_psge_dependent_bh.csv"
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file, sep=";", header=0)
        visualize_pandas(df, problem)
    else:
   
        embbedding = []
        fitness_values = []
        runs = []
        iterations = []
        algorithm = []


        # grammar.set_path('grammars/bostonhousing_torch.pybnf')
        grammar.set_path('grammars/regression_torch.pybnf')
        grammar.read_grammar(probs_update="standard")
        grammar.set_max_tree_depth(10)
        grammar.set_min_init_tree_depth(10)



        embbedding, fitness_values, iterations, runs, algorithm, max_size = embeddings_total(grammar, folder_standard, embbedding, max_size, fitness_values, runs, iterations, algorithm, 'PSGE')
        print("PSGE treinado")

        # grammar_dpsge.set_path('grammars/bostonhousing_torch.pybnf')
        grammar_dpsge.set_path('grammars/regression_torch.pybnf')
        grammar_dpsge.read_grammar(probs_update="dependent")
        grammar_dpsge.set_max_tree_depth(10)
        grammar_dpsge.set_min_init_tree_depth(10)
        text = ''

        embbedding, fitness_values, iterations, runs, algorithm, max_size = embeddings_total(grammar_dpsge, folder, embbedding, max_size, fitness_values, runs, iterations, algorithm, 'DPSGE')
        print("DPSGE treinado")


        temp = np.zeros((len(embbedding), max_size))
        for i,e in enumerate(embbedding):
            temp[i, : e.shape[0]] = e
        embbedding = temp

        new_values = embbedding_train.transform(embbedding)
        # pickle.dump(new_values, open("embbedding_sge_psge.sav","wb"))

        visualize(new_values, fitness_values, iterations, runs, algorithm, csv_file)
        # load_sge(folder_sge, embbedding_train, max_size)


