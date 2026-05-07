import numpy as np
from enum import Enum
from sge.parameters import params
import json
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Enum):
            return obj.value
        return json.JSONEncoder.default(self, obj)
    
def calculate_unique_percentage(population_phenotypes, previous_population=None):
    unique_phenotypes = set(population_phenotypes)
    unique_count = len(unique_phenotypes)
    total_count = len(population_phenotypes)
    unique_percentage = (unique_count / total_count) * 100 if total_count > 0 else 0

    if previous_population is not None:
        previous_phenotypes = set(ind['phenotype'] for ind in previous_population)
        new_count = sum(1 for p in population_phenotypes if p not in previous_phenotypes)
        percentage_new_individuals = (new_count / total_count) * 100 if total_count > 0 else 0
        return unique_percentage, percentage_new_individuals

    return unique_percentage, 0


def evolution_progress(generation, pop, best, best_gen, gram, previous_population=None):
    fitness_samples = [i['fitness'] for i in pop]
    test_error_samples = [i['other_info']['test_error'] for i in pop]
    depth_samples = [i['tree_depth'] for i in pop]
    length_used_genotype_best = sum(i for i in best['mapping_values'])
    phenotypes = [i['phenotype'] for i in pop]
    unique_percentage, percentage_new_individuals = calculate_unique_percentage(phenotypes, previous_population)

    data = '%4d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.2f\t%.2f' % (
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
        length_used_genotype_best,
        unique_percentage,
        percentage_new_individuals
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

    # to_save = []
    # to_save.append({"grammar": gram})
    # folder = params['EXPERIMENT_NAME'] + '/last_' + str(params['RUN'])
    # if not os.path.exists(folder):
    #     os.makedirs(folder,  exist_ok=True)
    # open('%s/generation_%d.json' % (folder,(generation)), 'w').write(json.dumps(to_save, cls=NumpyEncoder))


def save_progress_to_file(data):
    with open('%s/run_%d/progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')


def save_step(generation, population):
    c = json.dumps(population)
    open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(c)


def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower, cls=NumpyEncoder)
    open('%s/run_%d/parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()