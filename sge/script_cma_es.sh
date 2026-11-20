#!/bin/bash

MAX_JOBS=10   # change this to control parallelism

# SEEDS=(1234 4321 1111 2222 3333 4444 5555 6666 7777 8888 9999 1010 2020 3030 4040 5050 6060 7070 8080 9090 1122 3344 5566 7788 9900 1357 2468 3691 4812 5793 1212 1313 1414 1515 1616 1717 1818 1919 2323 2424 2525 2626 2727 2828 2929 3434 3535 3636 3737 3838 3939 4545 4646 4747 4848 4949 5656 5757 5858 5959 6767 6868 6969 7878 7979 8989 9091 9191 9292 9393 9494 9595 9696 9797 9898 1235 2345 3456 4567 5678 6789 7890 8901 9012 1023 2134 3245 4356 5467 6578 7689 8790 9801 1092 2103 3214 4325 5436 6547 7658)

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

printf '%s\n' "${SEEDS[@]}" | parallel -j $MAX_JOBS \
    python -m examples.bostonhousing_pytorch \
    --grammar grammars/bostonhousing_torch.pybnf \
    --parameters parameters/auto_params.yaml \
    --seed {} --run {#} \
    --tsize 3 \
    --elitism 1 \
    --search_strategy eda \
    --learning_strategy none \
    --algorithm_method sgef \
    --remap False \
    --genotype_distribution cma_es \
    --genotype_init fixed \
    --n_best_genotype 250 \
    --experiment_name experiments_500_gen/cma_es_n_best_250_sgef_eda_elite_1_sigmoid/bostonhousing 


printf '%s\n' "${SEEDS[@]}" | parallel -j $MAX_JOBS \
    python -m examples.bostonhousing_pytorch \
    --grammar grammars/bostonhousing_torch.pybnf \
    --parameters parameters/auto_params.yaml \
    --seed {} --run {#} \
    --tsize 3 \
    --elitism 1 \
    --search_strategy eda \
    --learning_strategy independent \
    --algorithm_method psge \
    --learning_factor 0.01 \
    --n_best 1 \
    --remap False \
    --genotype_distribution cma_es \
    --genotype_init fixed \
    --n_best_genotype 250 \
    --experiment_name experiments_500_gen/cma_es_n_best_250_psge_eda_elite_1_sigmoid_nbest1/bostonhousing 


printf '%s\n' "${SEEDS[@]}" | parallel -j $MAX_JOBS \
    python -m examples.bostonhousing_pytorch \
    --grammar grammars/bostonhousing_torch.pybnf \
    --parameters parameters/auto_params.yaml \
    --seed {} --run {#} \
    --tsize 3 \
    --elitism 1 \
    --search_strategy eda \
    --learning_strategy depth_based \
    --algorithm_method psge \
    --learning_factor 0.01 \
    --n_best 1 \
    --remap False \
    --genotype_distribution cma_es \
    --genotype_init fixed \
    --n_best_genotype 250 \
    --experiment_name experiments_500_gen/cma_es_n_best_250_psge_depth_based_eda_elite_1_sigmoid_nbest1/bostonhousing 

