# Probabilistic Structured Grammatical Evolution (PSGE)

A Python3 implementation of Probabilistic Structured Grammatical Evolution.

## Overview

Probabilistic Structured Grammatical Evolution (PSGE) combines the representation of Structured Grammatical Evolution (SGE) with the mapping mechanism of Probabilistic Grammatical Evolution (PGE).

**Genotype representation:** PSGE uses a set of dynamic lists of real values, with one list per grammar non-terminal. Each element (codon) represents the probability of choosing a production rule, updating based on the best individual's phenotype each generation using PGE's mapping mechanism.

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



## Getting Started

### Prerequisites

- Python 3.5 or newer
- Dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r sge/requirements.txt
```

### Basic Usage

PSGE requires two components to solve a problem:
1. A **grammar** file (see `grammars/` folder)
2. A **fitness function** (see `examples/` folder)

#### Quick Start Example

Run symbolic regression:

```bash
python3 -m examples.symreg --grammar grammars/regression.pybnf
```

## Configuration

### Parameters File

Create a YAML parameters file or use one from `parameters/`:

```bash
python3 -m examples.symreg --grammar grammars/regression.pybnf --parameters parameters/standard.yml
```

### Command Line Arguments

| --parameters | str | YAML parameters file (must include extension) | 
| --grammar | str | Path to grammar file (required) |
| --popsize | int | Population size |
| --generations | int | Number of generations |
| --elitism | int | Number of individuals to survive each generation |
| --prob_crossover | float | Crossover probability (0.0-1.0) |
| --prob_mutation | float | Mutation probability (0.0-1.0) |
| --tsize | int | Tournament size for parent selection |
| --min_tree_depth | int | Initial tree depth |
| --max_tree_depth | int | Maximum tree depth |
| --learning_factor | float | Probability update learning factor ((0.0-1.0)) |
| --n_best | int | Number of individuals for the updating mechanism |
| --adaptive_lf | bool | Enable adaptive learning factor |
| --adaptive_increment | float | Learning factor increment per generation |
| --remap | bool | Remap elites each iteration |
| --adaptive_mutation | bool | Use Adaptive Facilitated Mutation (AFM) |
| --prob_mutation_probs | float | AFM mutation probability (requires `--adaptive_mutation`) |
| --gauss_sd | float | AFM Gaussian standard deviation (requires `--adaptive_mutation`) |
| --experiment_name | str | Output folder name for statistics |
| --run | int | Run number (for tracking) |
| --seed | int | Random seed |
| --include_genotype | bool | Save genotype in log files |
| --save_step | int | Statistics save frequency |
| --verbose | bool | Verbose output |

**Note:** Override parameter file settings by adding arguments to the command line. For example:

```bash
python3 -m examples.symreg --grammar grammars/regression.pybnf --parameters parameters/standard.yml --seed 123
```

For full parameter documentation:

```bash
python3 -m examples.symreg --help
```

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

## Support

For questions or suggestions, contact:
- Jessica Mégane: [jessicac@dei.uc.pt](mailto:jessicac@dei.uc.pt)

## References

- O'Neill, M. and Ryan, C. (2003). **Grammatical Evolution: Evolutionary Automatic Programming in an Arbitrary Language**. Kluwer Academic Publishers.

- Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E., and O'Neill, M. (2017). **PonyGE2: Grammatical Evolution in Python**. arXiv preprint, arXiv:1703.08535.

Lourenço, N., Assunção, F., Pereira, F. B., Costa, E., and Machado, P.. **Structured Grammatical Evolution: A Dynamic Approach**. In Handbook of Grammatical Evolution. Springer Int, 2018.

Mégane, J., Lourenço, N., and Machado, P.. **Probabilistic Grammatical Evolution**. In Genetic Programming, Ting Hu, Nuno Lourenço, and Eric Medvet (Eds.). Springer International Publishing, Cham, 198–213, 2021.

Carvalho, P., Mégane, J., Lourenço, N., Machado, P. (2023). **Context Matters: Adaptive Mutation for Grammars**. In: Pappa, G., Giacobini, M., Vasicek, Z. (eds) Genetic Programming. EuroGP 2023. Lecture Notes in Computer Science, vol 13986. Springer, Cham.