#!/bin/bash
#SBATCH --time=12:10:00
#SBATCH --mem=500MB
#SBATCH --job-name=psge_job
#SBATCH --output=logs/psge_job%j.log


export PATH=$HOME/.local/bin:$PATH
module load Python
pip install numpy
pip install PyYAML
pip install tqdm

python -m examples.symreg --experiment_name /scratch/p288427/megalomania/psge --run $5 --seed $5 --parameters parameters/standard.yml --grammars/regression.pybnf --gauss_sd $1 --prob_mutation_probs $2 --delay $3 --remap $4 --prob_mutation $6
