import sys
import sge.grammar as grammar
import sge.logger as logger
import sge.checkpoint as checkpoint
from sge.distribution import DiagonalGaussianDistribution
from datetime import datetime
from tqdm import tqdm
import copy
import numpy as np
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate, mutate_level, mutation_prob_mutation, mutate_100
from sge.operators.selection import tournament
from sge.operators.update import update_distributions, grammar_mutation
from sge.parameters import (
    params,
    SearchStrategy,
    AlgorithmMethod,
    GenotypeDistribution,
    set_parameters,
    load_parameters
)

# Global distribution object for genotype sampling
_genotype_distribution = None

class CMA_ES_Distribution:
    """Legacy placeholder. Use DiagonalGaussianDistribution from distribution module instead."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.normal(self.mean, self.std)

    def update(self, new_mean, new_std):
        self.mean = new_mean
        self.std = new_std

def generate_random_individual(max_expansions):
    """
    Generate a random individual with genotypes sampled from learned distributions.
    
    Uses the global CMA-ES-inspired distribution to sample genotype codons.
    This replaces uniform random initialization with learned Gaussian sampling.
    Falls back to uniform sampling when distribution is not initialized.
    """
    ind = {'fitness': None, 'tree_depth': None}
    
    if params['GENOTYPE_INIT'] == 'fixed':
        if params['GENOTYPE_DISTRIBUTION'] == GenotypeDistribution.CMA_ES:
            # Sample from learned Gaussian distributions per non-terminal
            if _genotype_distribution is not None:
                # sample_genotype returns a PSGE-style genotype split per non-terminal
                genotype = _genotype_distribution.sample_genotype()
            else:
                # Fallback to uniform if distribution not initialized
                genotype = [[[-1, np.random.uniform(0, 1), -1] for _ in range(max_expansions[nt])] for nt in grammar.get_non_terminals()]
        else:
            # Default: uniform sampling
            genotype = [[[-1, np.random.uniform(0, 1), -1] for _ in range(max_expansions[nt])] for nt in grammar.get_non_terminals()]
    else:
        # Dynamic genotype initialization
        genotype = [[] for _ in grammar.get_non_terminals()]
    
    ind['genotype'] = genotype
    if params['ADAPTIVE_MUTATION']:
        ind['mutation_probs'] = [params['PROB_MUTATION'] for _ in genotype]
    
    if params['ALGORITHM_METHOD'] == AlgorithmMethod.COPSGE or params['ALGORITHM_METHOD'] == AlgorithmMethod.PSGE_COPSGE:
        ind['pcfg'] = grammar.get_pcfg()

    return ind


def make_initial_population(pop_size):
    count = grammar.get_count_references_to_non_terminals()
    for _ in range(pop_size):
        yield generate_random_individual(count)


def has_zero_fitness(population):
    """Return whether the evaluated population contains an exact solution."""
    return any(individual.get('fitness') == 0 for individual in population)


def evaluate(ind, eval_func):
    mapping_values = [0 for _ in ind['genotype']]
    if params['ALGORITHM_METHOD'] == AlgorithmMethod.COPSGE or params['ALGORITHM_METHOD'] == AlgorithmMethod.PSGE_COPSGE:
        probabilistic_distribution = ind['pcfg']
    else:
        probabilistic_distribution = grammar.get_pcfg()
    phen_tokens, tree_depth, gram_counter = grammar.mapping(probabilistic_distribution, ind['genotype'], mapping_values)
    phen = "".join(phen_tokens)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen_tokens
    ind['phenotype_string'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth
    ind['grammar_counter'] = gram_counter


def _initialize_grammar_and_distribution():
    global _genotype_distribution
    grammar.set_path(params['GRAMMAR'])
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])
    grammar.read_grammar(
        params['LEARNING_STRATEGY'],
        params['ALGORITHM_METHOD'],
        params.get('LEVELS_UP', 1),
        params.get('LEVELS_DOWN', 3),
    )
    
    if params['GENOTYPE_DISTRIBUTION'] == GenotypeDistribution.CMA_ES:
        print("Initializing CMA-ES-inspired genotype distribution")
        # Initialize CMA-ES-inspired genotype distribution
        # This allows learned sampling instead of uniform random initialization
        max_expansions = grammar.get_count_references_to_non_terminals()
        _genotype_distribution = DiagonalGaussianDistribution(
            non_terminals=grammar.get_non_terminals(),
            max_expansions=max_expansions,
            init_std=0.2
        )
    else:
        _genotype_distribution = None


def _restore_checkpoint(resume_from, generations_override=None):
    global _genotype_distribution

    restored, checkpoint_file = checkpoint.load_checkpoint(resume_from)
    restored_params = restored.get('params')
    if not isinstance(restored_params, dict):
        raise checkpoint.CheckpointError('Checkpoint parameters are missing or invalid')

    actual_run_folder = str(checkpoint_file.parent.parent)
    params.clear()
    params.update(copy.deepcopy(restored_params))
    params['RUN_FOLDER'] = actual_run_folder
    params['RESUME_FROM'] = str(resume_from)

    if checkpoint.parameters_sha256(params) != restored.get('parameters_sha256'):
        raise checkpoint.CheckpointError(
            'Checkpoint parameter fingerprint does not match its saved parameters'
        )

    if generations_override is not None:
        generations_override = int(generations_override)
        saved_generations = int(params['GENERATIONS'])
        if generations_override < saved_generations:
            raise checkpoint.CheckpointError(
                'A resumed run cannot reduce GENERATIONS from %d to %d'
                % (saved_generations, generations_override)
            )
        params['GENERATIONS'] = generations_override

    parameters_path = params.get('PARAMETERS')
    saved_parameters_hash = restored.get('parameters_file_sha256')
    if parameters_path and saved_parameters_hash:
        try:
            current_parameters_hash = checkpoint.file_sha256(parameters_path)
        except OSError as exc:
            raise checkpoint.CheckpointError(
                'The parameter file used by the checkpoint is unavailable: %s'
                % parameters_path
            ) from exc
        if current_parameters_hash != saved_parameters_hash:
            raise checkpoint.CheckpointError(
                'The parameter file has changed since this checkpoint was created'
            )

    _initialize_grammar_and_distribution()
    grammar_path = params['GRAMMAR']
    try:
        current_grammar_hash = checkpoint.file_sha256(grammar_path)
    except OSError as exc:
        raise checkpoint.CheckpointError(
            'The grammar file used by the checkpoint is unavailable: %s'
            % grammar_path
        ) from exc
    if current_grammar_hash != restored.get('grammar_sha256'):
        raise checkpoint.CheckpointError(
            'The grammar file has changed since this checkpoint was created'
        )
    try:
        grammar.set_pcfg(restored['grammar_pcfg'])
        distribution_state = restored.get('genotype_distribution')
        if distribution_state is not None:
            if _genotype_distribution is None:
                raise checkpoint.CheckpointError(
                    'Checkpoint contains a genotype distribution that is disabled by its parameters'
                )
            _genotype_distribution.set_state(distribution_state)
    except (KeyError, TypeError, ValueError) as exc:
        raise checkpoint.CheckpointError('Checkpoint model state is incompatible') from exc

    if len(restored['population']) != int(params['POPSIZE']):
        raise checkpoint.CheckpointError(
            'Checkpoint population size does not match the saved POPSIZE parameter'
        )
    checkpoint.restore_rng_state(restored['rng_state'])
    logger.prepare_recovery_logs(
        actual_run_folder,
        checkpoint_file,
        restored['completed_generation'],
        restored['next_generation'],
    )
    return restored


def setup(parameters_file_path=None, resume_from=None,
          generations_override=None):
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
        params['PARAMETERS'] = parameters_file_path
    set_parameters(sys.argv[1:])
    if generations_override is None and any(
            argument == '--generations' or argument.startswith('--generations=')
            for argument in sys.argv[1:]):
        generations_override = params['GENERATIONS']
    if resume_from is not None:
        params['RESUME_FROM'] = resume_from
    if params.get('RESUME_FROM'):
        return _restore_checkpoint(
            params['RESUME_FROM'], generations_override
        )

    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    params['EXPERIMENT_NAME'] += "/" + str(params['LEARNING_FACTOR'] * 100)
    logger.prepare_dumps()
    checkpoint.seed_random_generators(int(params['SEED']))
    _initialize_grammar_and_distribution()
    return None


def _checkpoint_state(completed_generation, next_generation, population,
                      previous_population, best, best_gen, flag):
    parameters_path = params.get('PARAMETERS')
    return {
        'completed_generation': int(completed_generation),
        'next_generation': int(next_generation),
        'population': population,
        'previous_population': previous_population,
        'best': best,
        'best_gen': best_gen,
        'flag': bool(flag),
        'grammar_pcfg': copy.deepcopy(grammar.get_pcfg()),
        'genotype_distribution': (
            _genotype_distribution.get_state()
            if _genotype_distribution is not None else None
        ),
        'params': copy.deepcopy(params),
        'parameters_sha256': checkpoint.parameters_sha256(params),
        'parameters_file_sha256': (
            checkpoint.file_sha256(parameters_path)
            if parameters_path else None
        ),
        'grammar_sha256': checkpoint.file_sha256(params['GRAMMAR']),
        'command': list(sys.argv),
        'rng_state': checkpoint.capture_rng_state(),
    }



def evolutionary_algorithm(evaluation_function=None, parameters_file=None,
                           resume_from=None, generations=None):
    restored = setup(
        parameters_file_path=parameters_file,
        resume_from=resume_from,
        generations_override=generations,
    )
    if restored is None:
        population = list(make_initial_population(params['POPSIZE']))
        flag = False    # alternate False - best overall
        best = None
        best_gen = None
        it = 0
        for i in tqdm(population):
            if i['fitness'] is None:
                evaluate(i, evaluation_function)
        previous_population = copy.deepcopy(population)
    else:
        population = restored['population']
        previous_population = restored['previous_population']
        best = restored['best']
        best_gen = restored['best_gen']
        flag = restored['flag']
        it = int(restored['next_generation'])

    while it <= params['GENERATIONS']:        

        population.sort(key=lambda x: x['fitness'])

        # best individual overall
        if not best:
            best = copy.deepcopy(population[0])
            best_gen = copy.deepcopy(best)
        elif population[0]['fitness'] <= best['fitness']:
            best = copy.deepcopy(population[0])

        if flag:
            update_distributions(params['LEARNING_STRATEGY'], [best_gen] + population, params['LEARNING_FACTOR'], params['N_BEST'])
            flag = not flag
        else:
            update_distributions(params['LEARNING_STRATEGY'], population, params['LEARNING_FACTOR'], params['N_BEST'])
            flag = not flag
     
        #     if params['ADAPTIVE_LF']:
        #         params['LEARNING_FACTOR'] += params['ADAPTIVE_INCREMENT']

     
        logger.evolution_progress(it, population, best, best_gen, grammar.get_pcfg(),previous_population)

        if has_zero_fitness(population):
            return best
        
        # Update genotype distributions based on elite individuals
        # This refines the Gaussian sampling for next generation
        if params['GENOTYPE_DISTRIBUTION'] == GenotypeDistribution.CMA_ES:
            _genotype_distribution.update(population, params['N_BEST_GENOTYPE'])
            logger.save_distribution(it, _genotype_distribution)

        if params['SEARCH_STRATEGY'] == SearchStrategy.EDA or (params['SEARCH_STRATEGY'] == SearchStrategy.HYBRID and it % 2 == 0):
            new_population = list(make_initial_population(params['POPSIZE'] - params['ELITISM']))

            for i in tqdm(new_population):
                evaluate(i, evaluation_function)
            new_population.sort(key=lambda x: x['fitness'])
            # best individual from the current generation
            best_gen = copy.deepcopy(new_population[0])

            if params['REMAP']:
                for i in tqdm(population[:params['ELITISM']]):
                    evaluate(i, evaluation_function)
            new_population += population[:params['ELITISM']]

        else:
            new_population = []
            while len(new_population) < params['POPSIZE'] - params['ELITISM']:
                if np.random.uniform() < params['PROB_CROSSOVER']:
                    p1 = tournament(population, params['TSIZE'])
                    p2 = tournament(population, params['TSIZE'])
                    ni = crossover(p1, p2)
                else:
                    ni = tournament(population, params['TSIZE'])
                
                if params['ALGORITHM_METHOD'] == AlgorithmMethod.COPSGE or params['ALGORITHM_METHOD'] == AlgorithmMethod.PSGE_COPSGE:
                    ni = grammar_mutation(ni, params['PROB_MUTATION_GRAMMAR'], params['NORMAL_DIST_SD'])
                
                if params['ADAPTIVE_MUTATION']:
                    # Adaptive Facilitated Mutation
                    ni = mutation_prob_mutation(ni)
                    ni = mutate_level(ni)
                else:
                    if params['ALGORITHM_METHOD'] == AlgorithmMethod.COPSGE or params['ALGORITHM_METHOD'] == AlgorithmMethod.PSGE_COPSGE:
                        ni = mutate(ni, params['PROB_MUTATION'], ni['pcfg'])
                    else:
                        ni = mutate(ni, params['PROB_MUTATION'], grammar.get_pcfg())

                new_population.append(ni)


            # new_population += population[:params['ELITISM']]
            for i in tqdm(new_population):
                evaluate(i, evaluation_function)
            new_population.sort(key=lambda x: x['fitness'])
            # best individual from the current generation
            best_gen = copy.deepcopy(new_population[0])

            if params['REMAP']:
                for i in tqdm(population[:params['ELITISM']]):
                    evaluate(i, evaluation_function)
            new_population += population[:params['ELITISM']]

        previous_population = copy.deepcopy(population)
        population = new_population
        it += 1

        checkpoint.save_checkpoint(
            params['RUN_FOLDER'],
            _checkpoint_state(
                completed_generation=it - 1,
                next_generation=it,
                population=population,
                previous_population=previous_population,
                best=best,
                best_gen=best_gen,
                flag=flag,
            ),
            keep=2,
        )

    return best


def resume_evolutionary_algorithm(evaluation_function, resume_from,
                                  generations=None):
    """Resume an experiment from a run folder or a specific checkpoint file."""
    return evolutionary_algorithm(
        evaluation_function=evaluation_function,
        resume_from=resume_from,
        generations=generations,
    )
