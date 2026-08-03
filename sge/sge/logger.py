import time
import numpy as np
from enum import Enum
from sge.parameters import params
import json
import os
import tempfile
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Enum):
            return obj.value
        return json.JSONEncoder.default(self, obj)


def _log_folder():
    return params.get('LOG_FOLDER', params['RUN_FOLDER'])


def _phenotype_key(phenotype):
    return (phenotype,) if isinstance(phenotype, str) else tuple(phenotype)


def calculate_unique_percentage(population_phenotypes, previous_population=None):
    population_phenotypes = [_phenotype_key(phenotype)
                             for phenotype in population_phenotypes]
    unique_phenotypes = set(population_phenotypes)
    unique_count = len(unique_phenotypes)
    total_count = len(population_phenotypes)
    unique_percentage = (unique_count / total_count) * 100 if total_count > 0 else 0

    if previous_population is not None:
        previous_phenotypes = set(_phenotype_key(ind['phenotype'])
                                  for ind in previous_population)
        new_count = sum(1 for p in population_phenotypes if p not in previous_phenotypes)
        percentage_new_individuals = (new_count / total_count) * 100 if total_count > 0 else 0
        return unique_percentage, percentage_new_individuals

    return unique_percentage, 0


def levenshtein_distances(tokens, previous_tokens):
    """Return raw and max-length-normalized Levenshtein token distance."""
    tokens = list(_phenotype_key(tokens))
    previous_tokens = list(_phenotype_key(previous_tokens))
    previous_row = list(range(len(previous_tokens) + 1))
    for row_index, token in enumerate(tokens, start=1):
        current_row = [row_index]
        for column_index, previous_token in enumerate(previous_tokens, start=1):
            current_row.append(min(
                current_row[-1] + 1,
                previous_row[column_index] + 1,
                previous_row[column_index - 1] + (token != previous_token),
            ))
        previous_row = current_row

    raw_distance = previous_row[-1]
    maximum_length = max(len(tokens), len(previous_tokens))
    normalized_distance = raw_distance / maximum_length if maximum_length else 0.0
    return raw_distance, normalized_distance


def evolution_progress(generation, pop, best, best_gen, gram, previous_population=None):
    fitness_samples = [i['fitness'] for i in pop]
    test_error_samples = [i['other_info']['test_error'] for i in pop]
    depth_samples = [i['tree_depth'] for i in pop]
    length_used_genotype_best = sum(i for i in best['mapping_values'])
    phenotypes = [i['phenotype'] for i in pop]
    unique_percentage, percentage_new_individuals = calculate_unique_percentage(phenotypes, previous_population)
    if previous_population:
        previous_best = min(previous_population, key=lambda individual: individual['fitness'])
        raw_distance, normalized_distance = levenshtein_distances(
            best_gen['phenotype'], previous_best['phenotype']
        )
    else:
        raw_distance, normalized_distance = 0, 0.0

    data = '%4d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.2f\t%.2f\t%d\t%.6e' % (
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
        np.nanstd(depth_samples),
        length_used_genotype_best,
        unique_percentage,
        percentage_new_individuals,
        raw_distance,
        normalized_distance
    )

    if params['VERBOSE']:
        print(data)

    save_progress_to_file(data)

    if generation % params['SAVE_STEP'] == 0:
        save_step(generation, pop)

    grammar_data = {"generation": generation, "grammar": gram}

    with open('%s/grammar_probabilities.json' % (_log_folder()), 'a') as f:
        json.dump(grammar_data, f, cls=NumpyEncoder)
        f.write(',\n')

    # to_save = []
    # to_save.append({"grammar": gram})
    # folder = params['EXPERIMENT_NAME'] + '/last_' + str(params['RUN_FOLDER'])
    # if not os.path.exists(folder):
    #     os.makedirs(folder,  exist_ok=True)
    # open('%s/generation_%d.json' % (folder,(generation)), 'w').write(json.dumps(to_save, cls=NumpyEncoder))


def save_progress_to_file(data):
    with open('%s/progress_report.csv' % (_log_folder()), 'a') as f:
        f.write(data + '\n')


def save_step(generation, population):
    c = json.dumps(population, cls=NumpyEncoder)
    with open('%s/iteration_%d.json' % (_log_folder(), generation), 'a') as output:
        output.write(c)


def save_distribution(generation, distribution):
    if distribution is None:
        return
    dist_data = {
        'generation': generation,
        'distribution': distribution.get_state()
    }
    path = '%s/genotype_distribution.json' % (_log_folder())
    with open(path, 'a') as f:
        json.dump(dist_data, f, cls=NumpyEncoder)
        f.write('\n')


def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    params_lower['command'] = ' '.join(os.sys.argv)
    c = json.dumps(params_lower, cls=NumpyEncoder)
    with open('%s/parameters.json' % (_log_folder()), 'a') as output:
        output.write(c)

def prepare_dumps():
    try:
        params['RUN_FOLDER'] = str(params['EXPERIMENT_NAME']) + "/run_" + str(params['RUN']) + "_" + str(int(time.time() * 1000000))
        os.makedirs('%s' % (params['RUN_FOLDER']))
    except FileExistsError:
        pass
    params['LOG_FOLDER'] = params['RUN_FOLDER']
    save_parameters()


def _atomic_json_write(path, data):
    path = Path(path)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', prefix='.%s-' % path.name,
            suffix='.tmp', dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(data, temporary, cls=NumpyEncoder, indent=2)
            temporary.write('\n')
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def prepare_recovery_logs(run_folder, checkpoint_file, completed_generation,
                          next_generation):
    """Create a new log segment without modifying any pre-crash logs."""
    run_path = Path(run_folder).resolve()
    recovery_root = run_path / 'recovery_logs'
    recovery_root.mkdir(parents=True, exist_ok=True)
    segment_name = 'resume_%d_from_generation_%d' % (
        int(time.time() * 1000000), next_generation
    )
    segment = recovery_root / segment_name
    segment.mkdir()

    manifest_path = run_path / 'recovery_manifest.json'
    if manifest_path.exists():
        with manifest_path.open('r', encoding='utf-8') as source:
            manifest = json.load(source)
        if not isinstance(manifest, dict) or not isinstance(manifest.get('recoveries'), list):
            raise ValueError('Invalid recovery manifest: %s' % manifest_path)
    else:
        manifest = {'version': 1, 'run_folder': str(run_path), 'recoveries': []}

    manifest['recoveries'].append({
        'checkpoint_file': str(Path(checkpoint_file).resolve()),
        'completed_generation': int(completed_generation),
        'next_generation': int(next_generation),
        'created_at_unix_microseconds': int(time.time() * 1000000),
        'log_segment': str(segment.relative_to(run_path)),
    })
    _atomic_json_write(manifest_path, manifest)
    params['LOG_FOLDER'] = str(segment)
    save_parameters()
    return str(segment)


def get_canonical_log_segments(run_folder):
    """Return log folders and generation bounds for the recovered run history."""
    run_path = Path(run_folder).resolve()
    manifest_path = run_path / 'recovery_manifest.json'
    if not manifest_path.exists():
        return [{
            'path': str(run_path),
            'start_generation': 0,
            'end_generation': None,
        }]

    with manifest_path.open('r', encoding='utf-8') as source:
        manifest = json.load(source)
    recoveries = manifest.get('recoveries', [])
    if not isinstance(recoveries, list):
        raise ValueError('Invalid recovery manifest: %s' % manifest_path)

    segments = []
    current_path = run_path
    current_start = 0
    for recovery in recoveries:
        completed = int(recovery['completed_generation'])
        segments.append({
            'path': str(current_path),
            'start_generation': current_start,
            'end_generation': completed,
        })
        current_path = run_path / recovery['log_segment']
        current_start = int(recovery['next_generation'])
    segments.append({
        'path': str(current_path),
        'start_generation': current_start,
        'end_generation': None,
    })
    return segments


def read_canonical_progress(run_folder):
    """Read progress rows from canonical log segments without changing source logs."""
    rows = []
    for segment in get_canonical_log_segments(run_folder):
        progress_path = Path(segment['path']) / 'progress_report.csv'
        if not progress_path.exists():
            continue
        with progress_path.open('r', encoding='utf-8') as source:
            for line in source:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    generation = int(stripped.split('\t', 1)[0])
                except ValueError:
                    continue
                if generation < segment['start_generation']:
                    continue
                end_generation = segment['end_generation']
                if end_generation is not None and generation > end_generation:
                    continue
                rows.append(stripped)
    return rows
