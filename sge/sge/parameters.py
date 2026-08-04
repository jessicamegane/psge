import argparse
import yaml
from distutils.util import strtobool
from enum import Enum
'''
This was adapted from PonyGE2: https://github.com/PonyGE/PonyGE2
Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E., and O'Neill, M. PonyGE2: Grammatical Evolution in Python. arXiv preprint, arXiv:1703.08535, 2017.
'''
""""Algorithm Parameters"""

class LearningStrategy(Enum):
    INDEPENDENT = 'independent'
    DEPTH_BASED = 'depth_based'
    SUBTREE_DEPENDENT = 'subtree_dependent'
    CONTEXT_AWARE = 'context_aware'
    CONTEXT_AWARE_DEPTH = 'context_aware_depth'
    CONTEXT_AWARE_PREVIOUS = 'context_aware_previous'
    NONE = 'none'

    @classmethod
    def from_string(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return None
        aliases = {
            'standard': cls.INDEPENDENT.value,
            'dependent': cls.DEPTH_BASED.value,
        }
        value = value.lower()
        value = aliases.get(value, value)
        try:
            return cls(value)
        except ValueError as exc:
            valid = ', '.join([item.value for item in cls])
            raise ValueError(f"Invalid learning strategy: {value}. Valid options are: {valid}") from exc

class SearchStrategy(Enum):
    STANDARD = 'standard'
    EDA = 'eda'
    HYBRID = 'hybrid'

    @classmethod
    def from_string(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return None
        try:
            return cls(value.lower())
        except ValueError as exc:
            valid = ', '.join([item.value for item in cls])
            raise ValueError(f"Invalid search strategy: {value}. Valid options are: {valid}") from exc

class AlgorithmMethod(Enum):
    PSGE = 'psge'
    COPSGE = 'copsge'
    SGEF = 'sgef'
    PSGE_COPSGE = 'psge_copsge'

    @classmethod
    def from_string(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return None
        try:
            return cls(value.lower())
        except ValueError as exc:
            valid = ', '.join([item.value for item in cls])
            raise ValueError(f"Invalid Algorithm Method: {value}. Valid options are: {valid}") from exc


class GenotypeDistribution(Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'
    CMA_ES = 'cma_es'
    VAE = 'vae'

    @classmethod
    def from_string(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return None
        try:
            return cls(value.lower())
        except ValueError as exc:
            valid = ', '.join([item.value for item in cls])
            raise ValueError(f"Invalid genotype distribution: {value}. Valid options are: {valid}") from exc

params = {'PARAMETERS': None,
          'POPSIZE': 100,
          'GENERATIONS': 100,
          'ELITISM': 10,                    # number of individuals that survive
          'PROB_CROSSOVER': 0.9,
          'PROB_MUTATION': 0.1,
          'MUTATION_STD': 0.5,
          'TSIZE': 3,
          'MIN_TREE_DEPTH': 6,
          'MAX_TREE_DEPTH': 17,
          'GRAMMAR': 'grammars/regression.pybnf',
          'EXPERIMENT_NAME': "dumps/Test",
          'SEED': None,
          'RUN': 1,
          'INCLUDE_GENOTYPE': True,
          'SAVE_STEP': 1,
          'VERBOSE': True,
          'LEARNING_FACTOR': 0.01,
          'ADAPTIVE_LF': False,
          'ADAPTIVE_INCREMENT': 0.0001,
          'REMAP': True,
          'ADAPTIVE_MUTATION': False,
          'PROB_MUTATION_PROBS': 0.3,
          'GAUSS_SD': 0.01,
          'GRAMMAR_PROBS': None,
          'N_BEST': 1,
          'LEVELS_UP': 1,
          'LEVELS_DOWN': 3,
          'SEARCH_STRATEGY': SearchStrategy.STANDARD,
          'LEARNING_STRATEGY': LearningStrategy.INDEPENDENT,
          'GENOTYPE_INIT': 'dynamic',  # 'FIXED' or 'DYNAMIC'
          'GENOTYPE_DISTRIBUTION': GenotypeDistribution.UNIFORM,
          'N_BEST_GENOTYPE': 100,
          'ALGORITHM_METHOD': AlgorithmMethod.PSGE,
          'PROB_MUTATION_GRAMMAR': 0.05,
          'NORMAL_DIST_SD': 0.5,
          'RESUME_FROM': None,
          }


def load_parameters(file_name=None):
    with open(file_name, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    params.update(cfg)
    if 'PROBS_UPDATE' in cfg and 'LEARNING_STRATEGY' not in cfg:
        params['LEARNING_STRATEGY'] = LearningStrategy.from_string(
            cfg['PROBS_UPDATE']
        )
    params.pop('PROBS_UPDATE', None)
    if 'LEARNING_STRATEGY' in params:
        params['LEARNING_STRATEGY'] = LearningStrategy.from_string(params['LEARNING_STRATEGY'])
    if 'SEARCH_STRATEGY' in params:
        params['SEARCH_STRATEGY'] = SearchStrategy.from_string(params['SEARCH_STRATEGY'])
    if 'ALGORITHM_METHOD' in params:
        params['ALGORITHM_METHOD'] = AlgorithmMethod.from_string(params['ALGORITHM_METHOD'])
    if 'GENOTYPE_DISTRIBUTION' in params:
        params['GENOTYPE_DISTRIBUTION'] = GenotypeDistribution.from_string(params['GENOTYPE_DISTRIBUTION'])


def set_parameters(arguments):
    # Initialise parser
    parser = argparse.ArgumentParser(
        usage=argparse.SUPPRESS,
        description="Welcome to PSGE code",
    )
    parser.add_argument('--parameters',
                        dest='PARAMETERS',
                        type=str,
                        help='Specifies the parameters file to be used. Must '
                             'include the full file extension. Full file path'
                             'does NOT need to be specified.')
    parser.add_argument('--resume',
                        dest='RESUME_FROM',
                        type=str,
                        help='Resume from a run folder or checkpoint file.')
    parser.add_argument('--popsize',
                        dest='POPSIZE',
                        type=int,
                        help='Specifies the population size.')
    parser.add_argument('--generations',
                        dest='GENERATIONS',
                        type=float,
                        help='Specifies the total number of generations.')
    parser.add_argument('--elitism',
                        dest='ELITISM',
                        type=int,
                        help='Specifies the total number of individuals that should survive in each generation.')
    parser.add_argument('--prob_crossover',
                        dest='PROB_CROSSOVER',
                        type=float,
                        help='Specifies the probability of crossover usage. Float required')
    parser.add_argument('--prob_mutation',
                        dest='PROB_MUTATION',
                        type=float,
                        help='Specifies the probability of mutation usage. Float required')
    parser.add_argument('--mutation_std',
                        dest='MUTATION_STD',
                        type=float,
                        help='Specifies the standard deviation for mutation. Float required')
    parser.add_argument('--tsize',
                        dest='TSIZE',
                        type=int,
                        help='Specifies the tournament size for parent selection.')
    parser.add_argument('--min_tree_depth',
                        dest='MIN_TREE_DEPTH',
                        type=int,
                        help='Specify the initialisation tree depth.')
    parser.add_argument('--max_tree_depth',
                        dest='MAX_TREE_DEPTH',
                        type=int,
                        help='Specify the initialisation tree depth.')
    parser.add_argument('--genotype_init',
                        dest='GENOTYPE_INIT',
                        type=str,
                        help='Specifies the method for generating the initial genotype: fixed or dynamic.')
    parser.add_argument('--grammar',
                        dest='GRAMMAR',
                        type=str,
                        help='Specifies the path to the grammar file.')
    def parse_learning_strategy(value):
        try:
            return LearningStrategy.from_string(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))

    def parse_search_strategy(value):
        try:
            return SearchStrategy.from_string(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))

    def parse_algorithm_method(value):
        try:
            return AlgorithmMethod.from_string(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    def parse_genotype_distribution(value):
        try:
            return GenotypeDistribution.from_string(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    parser.add_argument('--genotype_distribution',
                        dest='GENOTYPE_DISTRIBUTION',
                        type=parse_genotype_distribution,
                        help='Specifies the distribution for generating the initial genotype: uniform, normal, cma_es, vae.')
    parser.add_argument('--n_best_genotype',
                        dest='N_BEST_GENOTYPE',
                        type=int,
                        help='Specifies the number of best individuals to consider when updating the genotype distribution. Only used if --genotype_distribution is set to cma_es.')
    parser.add_argument('--learning_strategy',
                        dest='LEARNING_STRATEGY',
                        type=parse_learning_strategy,
                        help=('Learning strategy: independent, depth_based, '
                              'subtree_dependent, context_aware, '
                              'context_aware_depth, context_aware_previous, '
                              'or none.'))
    parser.add_argument('--probs_update',
                        dest='LEARNING_STRATEGY',
                        type=parse_learning_strategy,
                        help=argparse.SUPPRESS)
    parser.add_argument('--search_strategy',
                        dest='SEARCH_STRATEGY',
                        type=parse_search_strategy,
                        help='Search strategy: standard, eda, hybrid.')
    parser.add_argument('--algorithm_method',
                        dest='ALGORITHM_METHOD',
                        type=parse_algorithm_method,
                        help='Algorithm method: psge, copsge, sgef, psge_copsge.')
    parser.add_argument('--grammar_probs',
                        dest='GRAMMAR_PROBS',
                        type=str,
                        help='Path to json file, with list of probabilities to each production rule.')
    parser.add_argument('--learning_factor',
                        dest='LEARNING_FACTOR',
                        type=float,
                        help='Specifies the value of the learning factor used to update the probabilities.')
    parser.add_argument('--n_best',
                        dest='N_BEST',
                        type=int,
                        help='Specifies the number of individuals to consider in the update of probabilities.')
    parser.add_argument('--levels_up',
                        dest='LEVELS_UP',
                        type=int,
                        help=('Number of ancestors above an expanded node at '
                              'which to root its subtree context. Only used by '
                              'the subtree_dependent learning strategy.'))
    parser.add_argument('--levels_down',
                        dest='LEVELS_DOWN',
                        type=int,
                        help=('Maximum depth of the subtree context. Only used '
                              'by the subtree_dependent learning strategy.'))
    parser.add_argument('--adaptive_lf',
                        dest='ADAPTIVE_LF',
                        type=strtobool,
                        help='Specifies if it is supposed to run the adaptive version of PSGE, in which the learning factor updated based on the ADAPTIVE_INCREMENT defined.')
    parser.add_argument('--adaptive_increment',
                        dest='ADAPTIVE_INCREMENT',
                        type=float,
                        help='Specifies the value used to add to the learning factor each generation.')
    parser.add_argument('--remap',
                        dest='REMAP',
                        type=strtobool,
                        help='Specifies if the elitists are remapped each iteration.')
    parser.add_argument('--adaptive_mutation',
                        dest='ADAPTIVE_MUTATION',
                        type=strtobool,
                        help='Specifies if we want to use the traditional mutation or the Adaptive Facilitated Mutation.')
    parser.add_argument('--prob_mutation_probs',
                        dest='PROB_MUTATION_PROBS',
                        type=float,
                        help='Specifies the probability of occurring a mutation in the prob mutation. Option only if --adaptive_mutation is set to true.')
    parser.add_argument('--gauss_sd',
                        dest='GAUSS_SD',
                        type=float,
                        help='Specifies the value of the standard deviation used in the generation of a number with a normal distribution. Option only if --adaptive_mutation is set to true.')
    parser.add_argument('--prob_mutation_grammar',
                        dest='PROB_MUTATION_GRAMMAR',
                        type=float,
                        help='Specifies the probability of occurring a mutation in the individual grammar. Only available for algorithm methods COPSGE and PSGE_COPSGE.')
    parser.add_argument('--normal_dist_sd',
                        dest='NORMAL_DIST_SD',
                        type=float,
                        help='Specifies the value of the standard deviation used in the generation of a number with a normal distribution. Only used in the grammar mutation, for algorithm methods COPSGE and PSGE_COPSGE.')
    parser.add_argument('--experiment_name',
                        dest='EXPERIMENT_NAME',
                        type=str,
                        help='Specifies the name of the folder where stats are going to be stored.')
    parser.add_argument('--run',
                        dest='RUN',
                        type=int,
                        help='Specifies the run number.')
    parser.add_argument('--seed',
                        dest='SEED',
                        type=float,
                        help='Specifies the seed to be used by the random number generator.')
    parser.add_argument('--include_genotype',
                        dest='INCLUDE_GENOTYPE',
                        type=strtobool,
                        help='Specifies if the genotype is to be include in the log files.')
    parser.add_argument('--save_step',
                        dest='SAVE_STEP',
                        type=int,
                        help='Specifies how often stats are saved.')
    parser.add_argument('--verbose',
                        dest='VERBOSE',
                        type=strtobool,
                        help='Turns on the verbose output of the program.')

    # Parse command line arguments using all above information.
    args, _ = parser.parse_known_args(arguments)

    # All default args in the parser are set to "None".
    cmd_args = {key: value for key, value in vars(args).items() if value is
                not None}

    # Set "None" values correctly.
    for key in sorted(cmd_args.keys()):
        # Check all specified arguments.

        if type(cmd_args[key]) == str and cmd_args[key].lower() == "none":
            cmd_args[key] = None

    if 'PARAMETERS' in cmd_args:
        load_parameters(cmd_args['PARAMETERS'])
    params.update(cmd_args)

    if 'LEARNING_STRATEGY' in params:
        params['LEARNING_STRATEGY'] = LearningStrategy.from_string(params['LEARNING_STRATEGY'])
    if 'SEARCH_STRATEGY' in params:
        params['SEARCH_STRATEGY'] = SearchStrategy.from_string(params['SEARCH_STRATEGY'])
    if 'ALGORITHM_METHOD' in params:
        params['ALGORITHM_METHOD'] = AlgorithmMethod.from_string(params['ALGORITHM_METHOD'])
    if 'GENOTYPE_DISTRIBUTION' in params:
        params['GENOTYPE_DISTRIBUTION'] = GenotypeDistribution.from_string(params['GENOTYPE_DISTRIBUTION'])
