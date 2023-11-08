import argparse
import yaml
'''
This was adapted from PonyGE2: https://github.com/PonyGE/PonyGE2
Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E., and O'Neill, M. PonyGE2: Grammatical Evolution in Python. arXiv preprint, arXiv:1703.08535, 2017.
'''
""""Algorithm Parameters"""
params = {'PARAMETERS': None,
          'POPSIZE': 10,
          'GENERATIONS': 10,
          'ELITISM': 10,                    # number of individuals that survive
          'PROB_CROSSOVER': 0.9,
          'PROB_MUTATION': 0.1,
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
          'PROBS_UPDATE': 'standard',
          'LEARNING_FACTOR': 0.01,
          'ADAPTIVE_LF': False,
          'ADAPTIVE_INCREMENT': 0.0001,
          'REMAP': True,
          'ADAPTIVE_MUTATION': False,
          'PROB_MUTATION_PROBS': 0.3,
          'GAUSS_SD': 0.01,
          }


def load_parameters(file_name=None):
    with open(file_name, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    params.update(cfg)


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
    parser.add_argument('--grammar',
                        dest='GRAMMAR',
                        type=str,
                        help='Specifies the path to the grammar file.')
    parser.add_argument('--probs_update',
                        dest='PROBS_UPDATE',
                        type=str,
                        help='Specifies the type of update of probabilities. Currtently is implemented \'standard\' and \'dependent\'.')    
    parser.add_argument('--learning_factor',
                        dest='LEARNING_FACTOR',
                        type=float,
                        help='Specifies the value of the learning factor used to update the probabilities.')
    parser.add_argument('--adaptive_lf',
                        dest='ADAPTIVE_LF',
                        type=bool,
                        help='Specifies if it is supposed to run the adaptive version of PSGE, in which the learning factor updated based on the ADAPTIVE_INCREMENT defined.')
    parser.add_argument('--adaptive_increment',
                        dest='ADAPTIVE_INCREMENT',
                        type=float,
                        help='Specifies the value used to add to the learning factor each generation.')
    parser.add_argument('--remap',
                        dest='REMAP',
                        type=bool,
                        help='Specifies if the elitists are remapped each iteration.')
    parser.add_argument('--adaptive_mutation',
                        dest='ADAPTIVE_MUTATION',
                        type=bool,
                        help='Specifies if we want to use the traditional mutation or the Adaptive Facilitated Mutation.')
    parser.add_argument('--prob_mutation_probs',
                        dest='PROB_MUTATION_PROBS',
                        type=float,
                        help='Specifies the probability of occurring a mutation in the prob mutation. Option only if --adaptive_mutation is set to true.')
    parser.add_argument('--gauss_sd',
                        dest='GAUSS_SD',
                        type=float,
                        help='Specifies the value of the standard deviation used in the generation of a number with a normal distribution. Option only if --adaptive_mutation is set to true.')
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
                        type=bool,
                        help='Specifies if the genotype is to be include in the log files.')
    parser.add_argument('--save_step',
                        dest='SAVE_STEP',
                        type=int,
                        help='Specifies how often stats are saved.')
    parser.add_argument('--verbose',
                        dest='VERBOSE',
                        type=bool,
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

