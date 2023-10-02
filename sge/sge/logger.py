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
    fitness_samples = [i['fitness'] for i in pop]
    data = f"{generation};{best['fitness']};{best['genotype']};{best['phenotype']};{best['mutation_probs']};{gram};{np.mean(fitness_samples)};{np.std(fitness_samples)};{best['other_info']['test_error']}"
    if params['VERBOSE']:
        print(data)
    save_progress_to_file(data)
    if generation % params['SAVE_STEP'] == 0:
        save_step(generation, pop, gram)
    # save probabilities
    # to_save = []
    # to_save.append({"grammar": gram})
    # folder = params['EXPERIMENT_NAME'] + '/last_' + str(params['RUN'])
    # if not os.path.exists(folder):
    #     os.makedirs(folder,  exist_ok=True)
    # open('%s/generation_%d.json' % (folder,(generation)), 'w').write(json.dumps(to_save, cls=NumpyEncoder))


def save_progress_to_file(data):
    with open('%s/run_%d/progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')


# def save_step(generation, population):
#     c = json.dumps(population)
#     open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(c)

def save_step(generation, population, gram):
    to_save = []
    to_save.append({"grammar": gram.tolist()})
    for i in population[:4]:
        to_save.append({"fitness": i['fitness'], "mutation_probs": i["mutation_probs"]})

    open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(json.dumps(to_save))

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