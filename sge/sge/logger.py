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
    
def calculate_unique_percentage(population_phenotypes):
    unique_count = len(set(population_phenotypes))
    total_count = len(population_phenotypes)
    return (unique_count / total_count) * 100 if total_count > 0 else 0

def evolution_progress(generation, pop, best, best_gen, gram):
    fitness_samples = [i['fitness'] for i in pop]
    test_error_samples = [i['other_info']['test_error'] for i in pop]
    depth_samples = [i['tree_depth'] for i in pop]
    length_genotype_best = sum(len(i) for i in best['genotype'])
    phenotypes = [i['phenotype'] for i in pop]
    unique_percentage = calculate_unique_percentage(phenotypes)

    data = '%4d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.2f' % (
        generation,
        best['fitness'],
        best_gen['fitness'],
        np.nanmean(fitness_samples),
        np.nanstd(fitness_samples),
        best.get('other_info', {}).get('test_error', np.nan),  # safe access
        np.nanmean(test_error_samples),
        np.nanstd(test_error_samples),
        best['tree_depth'],
        np.nanmean(depth_samples),
        np.nanmedian(depth_samples),
        length_genotype_best,
        unique_percentage
    )

    if params['VERBOSE']:
        print(data)

    save_progress_to_file(data)

    if generation % params['SAVE_STEP'] == 0:
        save_step(generation, pop)

    grammar_data = {"generation": generation, "grammar": gram}

    with open('%s/run_%d/grammar_probabilities.json' % (params['EXPERIMENT_NAME'],params['RUN']), 'a') as f:
        json.dump(grammar_data, f, cls=NumpyEncoder)
        f.write(',\n')

def save_progress_to_file(data):
    with open('%s/run_%d/progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')


def save_step(generation, population):
    c = json.dumps(population)
    open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(c)


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