#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=500MB
#SBATCH --job-name=bt_sending_loop
#SBATCH --output=logs/sending_loop%j.log


export PATH=$HOME/.local/bin:$PATH
module load Python
pip install numpy
pip install PyYAML
pip install tqdm

python -m examples.symreg --experiment_name /project/p288427/megalomania/first_run --seed 71920 --parameters parameters/standard.yml --grammars/regression.pybnf --gauss_sd $1 --prob_mutation_probs $2 --mutate_grammar $3