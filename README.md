# Probabilistic Structured Grammatical Evolution python3 code

Probabilistic Structured Grammatical Evolution (PSGE) results from the combination of the representation of Structured Grammatical Evolution (SGE) and the mapping mechanism of Probabilistic Grammatical Evolution (PGE).

The genotype in PSGE is a set of dynamic lists of real values, with a list for each non-terminal of the grammar. Each element of the list (i.e., codon) represents the probability of choosing a production rule from the grammar, which are updated based on the phenotype of the best individual at the end of each generation, using the same mapping mechanism proposed by PGE.

A more in-depth explanation of the method and an analysis of its performance can be found in the [article](https://arxiv.org/pdf/2205.10685), published at the 2022 IEEE Congress on Evolutionary Computation (IEEE CEC 2022). If you use this algorithm, a reference to the following work would be appreciated:

```
@inproceedings{megane2022cec,
    author={Mégane, Jessica and Lourenço, Nuno and Machado, Penousal},
    booktitle={2022 IEEE Congress on Evolutionary Computation (CEC)}, 
    title={Probabilistic Structured Grammatical Evolution}, 
    year={2022},
    pages={1-9},
    doi={10.1109/CEC55065.2022.9870397}}
```



### Requirements
This code needs python3.5 or a newer version. More detail on the required libraries can be found in the `requirements.txt` file.


### Execution

Like all grammar-based Evolutionary Algorithms, to run the algorithm to solve a problem you need a **grammar** and a **fitness function**.
The folder `examples/` contains the code for some benchmark problems used in Genetic Programming, and the folder ``grammars/`` contain the respective grammars. To run, for example, a Symbolic Regression problem, you can use the following command:

```
python3 -m examples.symreg --grammar grammars/regression.pybnf
```

#### Parameters

The folder `parameters/` contains an example of standard parameters to run. You can define the parameters on a file and specify them when executing the code. For example:

```
python3 -m examples.symreg --grammar grammars/regression.pybnf --parameters parameters/standard.yml
```

You can also add manually more parameters when calling the code without changing the parameter file. Here is an example where we define the seed:

```
python3 -m examples.symreg --grammar grammars/regression.pybnf --parameters parameters/standard.yml --seed 123
```

If you need to know the possible parameters, you can use the flag ``--help``. For example:
```
python -m examples.symreg --help
```

Here is the list of possible parameters, and how to call them.

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
| --experiment_name | str | Specifies the name of the folder where stats are going to be stored. |
| --run | int | Specifies the run number. |
| --seed | float | Specifies the seed to be used by the random number generator. |
| --include_genotype | bool | Specifies if the genotype is to be included in the log files |
| --save_step | int | Specifies how often stats are saved. |
| --verbose | bool | Turns on the verbose output of the program. |
| --resume | str | Resumes an interrupted run from a run folder or a specific checkpoint file. |

### Checkpoints and recovery

The main evolutionary algorithm writes a complete recovery checkpoint after every
generation. A checkpoint includes the full population, learned grammar and genotype
distributions, resolved parameters and seed, best individuals, and random-generator
states. Only the two newest checkpoints are retained. After a run finishes normally,
the checkpoint files are removed; all normal experiment logs remain available.

To continue an interrupted experiment, run the same example and problem arguments and
pass the interrupted run folder:

```
python3 -m examples.symreg --resume dumps/Test/1.0/run_1_123456789
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

### Mutation

This code supports two types of mutations.

- The **standard mutation** consists in changing the values in the genotype according to the ``PROB_MUTATION`` parameter.
- The **adaptive facilitated mutation** can be enabled by setting the ``ADAPTIVE_MUTATION`` parameter to True. This mutation evolves different probabilities of mutation for each non-terminal of the grammar. Each individual contains a different array with probabilities of mutation for each non-terminal. The array starts with equal values for each non-terminal, pre-defined with the parameter ``PROB_MUTATION``. They update each genaration based on the ``PROB_MUTATION_PROBS`` parameter, which defines the likelihood of the values suffering an alteration, and the **GAUSS_SD** parameter which defines the impact of those changes.
The Adaptive Facilitated Mutation was publised and presented in the EuroGP 2023 conference, you can read the full paper [here](https://jessicamegane.pt/files/eurogp_afm.pdf). If you use this mutation please cite our paper.


### Support

Any questions, comments or suggestion should be directed to Jessica Mégane ([jessicac@dei.uc.pt](mailto:jessicac@dei.uc.pt)) or Nuno Lourenço ([naml@dei.uc.pt](mailto:naml@dei.uc.pt)).


## References

O'Neill, M. and Ryan, C. "Grammatical Evolution: Evolutionary Automatic Programming in an Arbitrary Language", Kluwer Academic Publishers, 2003.

Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E., and O'Neill, M. PonyGE2: Grammatical Evolution in Python. arXiv preprint, arXiv:1703.08535, 2017.

Lourenço, N., Assunção, F., Pereira, F. B., Costa, E., and Machado, P.. Structured Grammatical Evolution: A Dynamic Approach. In Handbook of Grammatical Evolution. Springer Int, 2018.

Mégane, J., Lourenço, N., and Machado, P.. Probabilistic Grammatical Evolution. In Genetic Programming, Ting Hu, Nuno Lourenço, and Eric Medvet (Eds.). Springer International Publishing, Cham, 198–213, 2021.

Carvalho, P., Mégane, J., Lourenço, N., Machado, P. (2023). Context Matters: Adaptive Mutation for Grammars. In: Pappa, G., Giacobini, M., Vasicek, Z. (eds) Genetic Programming. EuroGP 2023. Lecture Notes in Computer Science, vol 13986. Springer, Cham.
