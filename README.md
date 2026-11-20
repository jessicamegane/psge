# Probabilistic Structured Grammatical Evolution (PSGE) framework

A Python 3 framework for Probabilistic Structured Grammatical Evolution and related approaches.

#### Base algorithm

SGE with a floating-point genotype uses the same representation as the other approaches and keeps the grammar probabilities fixed. Select it with `--algorithm_method sgef`. [SGE implementation](https://github.com/nunolourenco/sge3).

**Adaptive bias using probabilistic grammars.** Canonical PSGE learns one production probability distribution per non-terminal. This framework also supports individual grammars and distributions that depend on the derivation context.

#### Univariate probabilistic grammar approaches

**PSGE** updates a shared probabilistic context-free grammar (PCFG) each generation using the production frequencies observed in selected individuals. The original method alternates between the best individual found so far and the best individual in the current generation. With remapping enabled, the updated probabilities can map the same genotype to a different phenotype. Select it with `--algorithm_method psge --learning_strategy independent`.

**Co-PSGE** gives each individual its own grammar and genotype. The individual's grammar maps its genotype into a program, and both the grammar and genotype are subject to variation during evolution. Select it with `--algorithm_method copsge`. [Co-PSGE implementation](https://github.com/jessicamegane/co-psge) · [GECCO 2022 paper](https://jessicamegane.pt/files/gecco_copsge.pdf)

**Hybrid PSGE–Co-PSGE** combines a shared grammar with an individual grammar for each member of the population. During mapping, each production's probability is the arithmetic mean of its probabilities in the two grammars. Select it with `--algorithm_method psge_copsge`.

#### Bivariate and multivariate probabilistic grammar approaches

Use these learning strategies with `--algorithm_method psge`. Each strategy maintains production probability distributions per non-terminal, conditioned on the following information.

| Learning strategy | What the distribution depends on |
| --- | --- |
| `previous_rule` | The rule used the last time the same non-terminal was expanded in the current derivation. The first expansion has its own context. |
| `depth_based` | The depth of the current expansion in the derivation tree. |
| `context_aware` | The most recent ancestor productions along the path to the current node. Set the number of ancestor productions with `--context_window` (default 1). |
| `previous_rule_depth` | Both the previous rule for the same non-terminal and the current derivation depth. |
| `context_aware_depth` | Both the ancestor production history and the current derivation depth. |
| `subtree_dependent` | A portion of the partially expanded derivation tree, excluding the node being expanded and its descendants. `--levels_up` sets how far to climb toward the root (default 1), and `--levels_down` limits the number of levels kept from that ancestor (default 3). |

## Getting Started

### Prerequisites

- Python 3.11 with `venv` and `bz2` support. Check that `python3.11 --version` works before starting.
- Poetry 2.x to install from `poetry.lock`. The lock file uses PyTorch builds for CUDA 12.1. GPU execution requires a compatible NVIDIA GPU and driver.

### Installation

#### Create the environment with `venv`

Run these commands from the repository root. The install command explicitly targets the project's `.venv`.

```bash
python3.11 -m venv .venv &&
VIRTUAL_ENV="$PWD/.venv" POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --no-root &&
source .venv/bin/activate &&
python -c 'import sys, numpy; print(sys.executable); print("NumPy", numpy.__version__)'
```

The final command should print an interpreter path ending in `.venv/bin/python` and the NumPy version. In a new terminal, run `source .venv/bin/activate` from the repository root before running the examples.

### Basic Usage

PSGE requires two components to solve a problem:
1. A **grammar** file (see `grammars/` folder)
2. A **fitness function** (see `examples/` folder)

#### Quick Start Example

Run symbolic regression from the repository root with the environment active. Run the later examples from the `sge/` directory as well.

```bash
cd sge
python -m examples.symreg pagiepolynomial --grammar grammars/regression.pybnf
```

## Configuration

### Parameters File

Create a YAML parameters file or use one from `parameters/`:

```bash
python -m examples.symreg pagiepolynomial --grammar grammars/regression.pybnf --parameters parameters/standard.yml
```

### Command Line Arguments

| argument | type | description |
| --------------- | ----------- | ------------ |
| --parameters | str | Specifies the parameters file to be used. Must include the full file extension. | 
| --popsize | int | Specifies the population size. |
| --generations | int | Specifies the total number of generations.
| --elitism | int | Specifies the total number of individuals that should survive in each generation. |
| --prob_crossover | float | Specifies the probability of crossover usage. Float required. |
| --prob_mutation | float | Specifies the probability of mutation usage. Float required. |
| --tsize | int | Specifies the tournament size for parent selection. |
| --min_tree_depth | int | Specifies the initialisation tree depth. |
| --max_tree_depth | int | Specifies the maximum tree depth. |
| --grammar | str | specifies the path to the grammar file. |
| --learning_factor | float | Specifies the value of the learning factor used to update the probabilities. |
| --adaptive_lf | bool | Specifies if it is supposed to run the adaptive version of PSGE, in which the learning factor updated based on the ADAPTIVE_INCREMENT defined. |
| --adaptive_increment | float | Specifies the value used to add to the learning factor each generation. |
| --remap | bool | Specifies if the elitists are remapped each iteration. |
| --adaptive_mutation | bool | Specifies if we want to use the traditional mutation or the Adaptive Facilitated Mutation. |
| --prob_mutation_probs | float | Specifies the probability of occurring a mutation in the prob mutation. Option only if --adaptive_mutation is set to true. |
| --gauss_sd | float | Specifies the value of the standard deviation used in the generation of a number with a normal distribution. Option only if --adaptive_mutation is set to true. |
| --search_strategy | str | Specifies the search strategy: standard, eda, hybrid. |
| --algorithm_method | str | Specifies the base algorithm: psge, copsge, sgef, psge_copsge. |
| --learning_strategy | str | Specifies the learning strategy: independent, depth_based, subtree_dependent, context_aware, context_aware_depth, previous_rule, previous_rule_depth, or none. |
| --context_window | int | Number of ancestor productions used by `context_aware` and `context_aware_depth`. |
| --levels_up | int | Number of ancestors to climb to find the root of the subtree context. |
| --levels_down | int | Maximum number of levels retained from the root of the subtree context. |
| --genotype_distribution | str | Specifies the distribution for generating the initial genotype: uniform, normal, cma_es, vae. |
| --experiment_name | str | Specifies the name of the folder where stats are going to be stored. |
| --prob_mutation_grammar | float | Probability of mutating an individual's grammar with `copsge` or `psge_copsge`. |
| --normal_dist_sd | float | Standard deviation of the normal distribution used for grammar mutation with `copsge` or `psge_copsge`. |
| --run | int | Specifies the run number. |
| --seed | float | Specifies the seed to be used by the random number generator. |
| --include_genotype | bool | Specifies if the genotype is to be included in the log files |
| --save_step | int | Specifies how often stats are saved. |
| --verbose | bool | Turns on the verbose output of the program. |
| --resume | str | Resumes an interrupted run from a run folder or a specific checkpoint file. |

**Note:** Override parameter file settings by adding arguments to the command line. For example:

```bash
python -m examples.symreg pagiepolynomial --grammar grammars/regression.pybnf --parameters parameters/standard.yml --seed 123
```

For full parameter documentation:

```bash
python -m examples.symreg pagiepolynomial --help
```


### Checkpoints and recovery

The main evolutionary algorithm writes a complete recovery checkpoint after every
generation. A checkpoint includes the full population, learned grammar and genotype
distributions, resolved parameters and seed, best individuals, and random-generator
states. Only the two newest checkpoints are retained. After a run finishes normally,
the checkpoint files are removed; all normal experiment logs remain available.

To continue an interrupted experiment, run the same example and problem arguments and
pass the interrupted run folder:

```
python -m examples.symreg pagiepolynomial --resume dumps/Test/1.0/run_1_123456789
```

The newest valid checkpoint is selected automatically. If it is incomplete or corrupt,
the previous checkpoint is used. A specific checkpoint file can also be supplied.
Recovery never modifies or deletes existing logs. Continued output is written under the
run's `recovery_logs/` directory, and `recovery_manifest.json` records which log segments
form the canonical run history.

Programmatic recovery is also available:

```python
import sge

sge.resume_evolutionary_algorithm(
    evaluation_function=eval_func,
    resume_from="dumps/Test/1.0/run_1_123456789",
)
```

Use `sge.get_canonical_log_segments(run_folder)` to resolve the original and recovery
log directories and their valid generation ranges, or
`sge.read_canonical_progress(run_folder)` to read the canonical progress rows. Since
checkpoints use Python serialization, only load checkpoint files produced locally by a
trusted experiment. Exact recovery also requires unchanged code, grammar, input data,
and evaluator behavior.


## Mutation Strategies

PSGE supports two mutation types:

### Standard Mutation

Changes genotype values randomly based on `--prob_mutation` parameter.

### Adaptive Facilitated Mutation (AFM)

Enable with `--adaptive_mutation true`. Each individual has mutation probabilities for each non-terminal grammar rule. These probabilities:
- Start with equal values (set by `--prob_mutation`)
- Update each generation based on `--prob_mutation_probs` and `--gauss_sd`

This adaptive approach was published at EuroGP 2023. See the [full paper](https://jessicamegane.pt/files/eurogp_afm.pdf) and cite it if used.

## Project Structure

- **`examples/`** - Benchmark problems (symbolic regression, feature engineering, etc.)
- **`grammars/`** - Grammar files in PYBNF, BNF, and TXT formats
- **`parameters/`** - Example configuration files
- **`resources/`** - Datasets for benchmark problems
- **`sge/`** - Core PSGE implementation

### Citation

If you use PSGE, please cite the following work:

```bibtex
@inproceedings{megane2022cec,
    author={Mégane, Jessica and Lourenço, Nuno and Machado, Penousal},
    booktitle={2022 IEEE Congress on Evolutionary Computation (CEC)}, 
    title={Probabilistic Structured Grammatical Evolution}, 
    year={2022},
    pages={1-9},
    doi={10.1109/CEC55065.2022.9870397}}
```

For a detailed explanation and performance analysis, see the [original paper](https://arxiv.org/pdf/2205.10685) published at IEEE CEC 2022.


## Support

For questions or suggestions, contact:
- Jessica Mégane: [jessicac@dei.uc.pt](mailto:jessicac@dei.uc.pt)

## References

- O'Neill, M. and Ryan, C. (2003). **Grammatical Evolution: Evolutionary Automatic Programming in an Arbitrary Language**. Kluwer Academic Publishers.

- Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E., and O'Neill, M. (2017). **PonyGE2: Grammatical Evolution in Python**. arXiv preprint, arXiv:1703.08535.

- Mégane, J., Lourenço, N., and Machado, P. (2022). **Co-evolutionary Probabilistic Structured Grammatical Evolution**. Proceedings of the Genetic and Evolutionary Computation Conference (GECCO ’22), 991–999. [DOI](https://doi.org/10.1145/3512290.3528833)

- Lourenço, N., Assunção, F., Pereira, F. B., Costa, E., and Machado, P.. **Structured Grammatical Evolution: A Dynamic Approach**. In Handbook of Grammatical Evolution. Springer Int, 2018.

- Mégane, J., Lourenço, N., and Machado, P.. **Probabilistic Grammatical Evolution**. In Genetic Programming, Ting Hu, Nuno Lourenço, and Eric Medvet (Eds.). Springer International Publishing, Cham, 198–213, 2021.

- Carvalho, P., Mégane, J., Lourenço, N., Machado, P. (2023). **Context Matters: Adaptive Mutation for Grammars**. In: Pappa, G., Giacobini, M., Vasicek, Z. (eds) Genetic Programming. EuroGP 2023. Lecture Notes in Computer Science, vol 13986. Springer, Cham.
