# Probabilistic Structured Grammatical Evolution python3 code

Probabilistic Structured Grammatical Evolution (PSGE) results from the combination of the representation of Structured Grammatical Evolution (SGE) and the mapping mechanism of Probabilistic Grammatical Evolution (PGE).

the genotype in PSGE is a set of dynamic lists of real values, with a list for each non-terminal of the grammar. Each element of the list (i.e., codon) represents the probability of choosing a production rule from the grammar, which are updated based on the phenotype of the best individual at the end of each generation, using the same mapping mechanism proposed by PGE.

A more in-depth explanation of the method and an analysis of its performance can be found in the [article](https://arxiv.org/pdf/2205.10685), published at the 2022 IEEE Congress on Evolutionary Computation (IEEE CEC 2022). 



### Requirements
This code needs python3.5 or a newer version. More detail on the required libraries can be found in the `requirements.txt` file.

### Execution

The folder `examples/` contains the code for some benchmark problems used in GP. To run, for example, Symbolic Regression, you can use the following command:

```python3 -m examples.symreg --experiment_name dumps/example --seed 791021 --parameters parameters/standard.yml --grammars/regression.pybnf```

The parameters such as the learning factor, mutation rate, etc., can be set on the parameters file.

### Support

Any questions, comments or suggestion should be directed to Jessica Mégane ([jessicac@dei.uc.pt](mailto:jessicac@dei.uc.pt)) or Nuno Lourenço ([naml@dei.uc.pt](mailto:naml@dei.uc.pt)).


## References

O'Neill, M. and Ryan, C. "Grammatical Evolution: Evolutionary Automatic Programming in an Arbitrary Language", Kluwer Academic Publishers, 2003.

Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E., and O'Neill, M. PonyGE2: Grammatical Evolution in Python. arXiv preprint, arXiv:1703.08535, 2017.

Lourenço, N., Assunção, F., Pereira, F. B., Costa, E., and Machado, P.. Structured Grammatical Evolution: A Dynamic Approach. In Handbook of Grammatical Evolution. Springer Int, 2018.

Mégane, J., Lourenço, N., and Machado, P.. Probabilistic Grammatical Evolution. In Genetic Programming, Ting Hu, Nuno Lourenço, and Eric Medvet (Eds.). Springer International Publishing, Cham, 198–213, 2021.
