from ast import Num
import numpy as np
from sge.parameters import params
import json
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def evolution_progress(generation, pop, best, gram):
    fitness_samples = []
    depth_samples = []
    genotype_samples = []
    for i in pop:
        fitness_samples.append(i['fitness'])
        depth_samples.append(i['tree_depth'])
        genotype_samples.append(sum(len(sublist) for sublist in i['genotype']))
    best_len = sum(len(sublist) for sublist in best['genotype'])
    # data = '%4d\t%.6e\t%.6e\t%.6e\t%.6e' % (generation, best['fitness'], np.mean(fitness_samples), np.std(fitness_samples), best['other_info']['test_error'])
    data = f"{generation};{best['fitness']};{np.nanmean(fitness_samples)};{np.nanstd(fitness_samples)};{best['other_info']['test_error']};{best['tree_depth']};{np.nanmean(depth_samples)};{np.nanmedian(depth_samples)};{best_len};{np.nanmean(genotype_samples)};{np.nanmedian(genotype_samples)}"

    if params['VERBOSE']:
        print(data)
    save_progress_to_file(data)
    # save probabilities
    to_save = []
    to_save.append({"grammar": gram})
    if generation % params['SAVE_STEP'] == 0:
        to_save = save_step(to_save, generation, pop)
    else:
        # save only best ind
        to_save.append({"genotype": best['genotype'],"fitness": best['fitness']})
    open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'w').write(json.dumps(to_save, cls=NumpyEncoder))


def save_progress_to_file(data):
    with open('%s/run_%d/progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')


def save_step(to_save, generation, population):
    for i in population:
        to_save.append({"genotype": i['genotype'],"fitness": i['fitness']})

    # open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(json.dumps(to_save))
    return to_save

def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower)
    open('%s/run_%d/parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()