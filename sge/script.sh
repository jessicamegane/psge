
parallel_processes=16
num_runs=30

benchmarks=(
    # pagiepolynomial
    # nguyen4polynomial
    nguyen5polynomial
    # quarticpolynomial
    # koza2polynomial
    # koza3polynomial
)

learning_factors=(0.01)
n_best_values=(1)
# elite_values=(1)
# learning_factors=(0.01)
# n_best_values=(100)
# remap_values=(true false)

for benchmark in "${benchmarks[@]}"; do
    for learning_factor in "${learning_factors[@]}"; do
        for n_best in "${n_best_values[@]}"; do
            # for elite in "${elite_values[@]}"; do

            # first batch
            seq "$num_runs" | parallel -j "$parallel_processes" \
            python -m examples.symreg_pytorch "$benchmark" \
                --grammar grammars/regression_torch.pybnf \
                --parameters parameters/auto_params.yaml \
                --n_best "$n_best" \
                --remap "false" \
                --learning_factor "$learning_factor" \
                --elitism 50 \
                --pop_size 500 \
                --seed {} --run {} --experiment_name /media/storage/jessica/search_strategy/full_experiments_fix/standard_no_remap_nbest"$n_best"_elite_50_pop_500/"$benchmark"/ \
                --search_strategy "standard"
            
            # grid batch
            seq "$num_runs" | parallel -j "$parallel_processes" \
            python -m examples.symreg_pytorch "$benchmark" \
                --grammar grammars/regression_torch.pybnf \
                --parameters parameters/auto_params.yaml \
                --n_best "$n_best" \
                --remap "false" \
                --learning_factor "$learning_factor" \
                --elitism 50 \
                --pop_size 500 \
                --seed {} --run {} --experiment_name /media/storage/jessica/search_strategy/full_experiments_fix/eda_no_remap_nbest"$n_best"_elite_50_pop_500/"$benchmark"/ \
                --search_strategy "eda"


            # first batch
            seq "$num_runs" | parallel -j "$parallel_processes" \
            python -m examples.symreg_pytorch "$benchmark" \
                --grammar grammars/regression_torch.pybnf \
                --parameters parameters/auto_params.yaml \
                --n_best "$n_best" \
                --remap "false" \
                --learning_factor "$learning_factor" \
                --elitism 100 \
                --max_tree_depth 15 \
                --seed {} --run {} --experiment_name /media/storage/jessica/search_strategy/full_experiments_fix/standard_no_remap_nbest"$n_best"_elite_100_pop_1000_depth_15/"$benchmark"/ \
                --search_strategy "standard"

            # grid batch
            seq "$num_runs" | parallel -j "$parallel_processes" \
            python -m examples.symreg_pytorch "$benchmark" \
                --grammar grammars/regression_torch.pybnf \
                --parameters parameters/auto_params.yaml \
                --n_best "$n_best" \
                --remap "false" \
                --learning_factor "$learning_factor" \
                --elitism 100 \
                --max_tree_depth 15 \
                --seed {} --run {} --experiment_name /media/storage/jessica/search_strategy/full_experiments_fix/eda_no_remap_nbest"$n_best"_elite_100_pop_1000_depth_15/"$benchmark"/ \
                --search_strategy "eda"
        
            # done

        done

    done

done



# # Set the number of parallel processes you want to run
# parallel_processes=16  # Adjust this based on your system's capacity

# num_runs=30

# benchmarks=(
#     nguyen4polynomial
#     nguyen5polynomial
#     pagiepolynomial
#     quarticpolynomial
#     koza2polynomial
#     koza3polynomial
# )

# learning_factors=(0.01 0.05)

# n_best_values=(1 100 250 500)


# for benchmark in "${benchmarks[@]}"; do
#     seq $num_runs | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch '"$benchmark"' --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 1 --learning_factor 0.01 --probs_update "standard" --search_strategy "standard" --experiment_name "/media/storage/jessica/search_strategy/full_experiments/standard/'"$benchmark"'/"" --run $i'
#     for learning_factor in "${learning_factors[@]}"; do
#         for n_best in "${n_best_values[@]}"; do
#             seq $num_runs | parallel -j $parallel_processes '
#             i={}
#             python -m examples.symreg_pytorch '"$benchmark"' --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best '"$n_best"' --learning_factor '"$learning_factor"' --probs_update "standard" --search_strategy "eda" --experiment_name "/media/storage/jessica/search_strategy/full_experiments/eda_remap_nbest'"$n_best"'/'"$benchmark"'/"" --run $i'

#         done
#     done

# done



# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 1 --learning_factor 0.01 --probs_update "standard" --experiment_name "/media/storage/jessica/search_strategy/results/standard_alternate_bests/quartic/" --run $i'

# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 1 --learning_factor 0.01 --probs_update "standard" --experiment_name "/media/storage/jessica/search_strategy/results_better_map/standard/quartic/" --run $i'



# seq $num_runs | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch nguyen4polynomial --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 1 --learning_factor 0.01 --probs_update "standard" --search_strategy "standard" --experiment_name "/media/storage/jessica/search_strategy/full_experiments/standard/nguyen4/" --run $i'

# seq $num_runs | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch nguyen4polynomial --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 1 --learning_factor 0.01 --probs_update "standard" --search_strategy "eda" --experiment_name "/media/storage/jessica/search_strategy/full_experiments/eda_remap_nbest1_elitism/nguyen4/" --run $i'


# seq $num_runs | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch nguyen4polynomial --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 100 --learning_factor 0.01 --probs_update "standard" --search_strategy "eda" --experiment_name "/media/storage/jessica/search_strategy/full_experiments/eda_remap_nbest100/nguyen4/" --run $i'

# seq $num_runs | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch nguyen4polynomial --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 500 --learning_factor 0.01 --probs_update "standard" --search_strategy "eda" --experiment_name "/media/storage/jessica/search_strategy/full_experiments/eda_remap_nbest500/nguyen4/" --run $i'

# seq $num_runs | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch nguyen4polynomial --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 1 --learning_factor 0.05 --probs_update "standard" --search_strategy "eda" --experiment_name "/media/storage/jessica/search_strategy/full_experiments/eda_remap_nbest1_elitism/nguyen4/" --run $i'

# seq $num_runs | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch nguyen4polynomial --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 100 --learning_factor 0.05 --probs_update "standard" --search_strategy "eda" --experiment_name "/media/storage/jessica/search_strategy/full_experiments/eda_remap_nbest100/nguyen4/" --run $i'

# seq $num_runs | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch nguyen4polynomial --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 500 --learning_factor 0.05 --probs_update "standard" --search_strategy "eda" --experiment_name "/media/storage/jessica/search_strategy/full_experiments/eda_remap_nbest500/nguyen4/" --run $i'




# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.parity_5 --grammar grammars/5_bit_parity_grammar.txt --parameters parameters/auto_params.yaml --seed $i --n_best 500 --learning_factor 0.01 --probs_update "standard" --experiment_name "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_500_best/5parity/" --run $i'

# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.parity_5 --grammar grammars/5_bit_parity_grammar.txt --parameters parameters/auto_params.yaml --seed $i --n_best 500 --learning_factor 0.05 --probs_update "standard" --experiment_name "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_500_best/5parity/" --run $i'


# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 500 --learning_factor 0.01 --probs_update "standard" --experiment_name "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_500_best/quartic/" --run $i'


# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --n_best 500 --learning_factor 0.05 --probs_update "standard" --experiment_name "/media/storage/jessica/search_strategy/results/eda_elitism_remap_update_500_best/quartic/" --run $i'


# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.parity_5 --grammar grammars/5_bit_parity_grammar.txt --parameters parameters/auto_params.yaml --seed $i --probs_update "autoPSGE" --epochs 100 --batch_size 128 --train_interval 50 --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/5parity/independent/genotype_init_0/probs_sum_mu_best_softmax/Nadam_CosineAnn/" --run $i'



# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --probs_update "autoPSGE" --epochs 100 --batch_size 64 --train_interval 25 --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/pagie/independent/probs_sum_mu_best_softmax/Nadam/" --run $i'

# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --probs_update "autoPSGE" --epochs 100 --batch_size 64 --train_interval 50 --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/pagie/independent/probs_sum_mu_best_softmax/Nadam/" --run $i'

# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --probs_update "autoPSGE" --epochs 100 --batch_size 128 --train_interval 25 --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/pagie/independent/probs_sum_mu_best_softmax/Nadam/" --run $i'

# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --probs_update "autoPSGE" --epochs 100 --batch_size 128 --train_interval 50 --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/pagie/independent/probs_sum_mu_best_softmax/Nadam/" --run $i'



# seq 15 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --probs_update "autoPSGE" --epochs 100 --batch_size 128 --train_interval 25 --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/pagie/independent/probs_sum_mu_best/" --run $i'


# seq 15 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --probs_update "autoPSGE" --epochs 100 --batch_size 128 --train_interval 50 --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/pagie/independent/probs_sum_mu_best/" --run $i'



# parallel_processes=10

# seq 15 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.parity_5 --grammar grammars/5_bit_parity_grammar.txt --parameters parameters/auto_params.yaml --seed $i --probs_update "standard" --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/5parity/independent/standard_train/" --run $i --save_step 100 --generations 101'



# seq 15 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.parity_5 --grammar grammars/5_bit_parity_grammar.txt --parameters parameters/auto_params.yaml --seed $i --probs_update "autoPSGE" --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/autopsge/5parity/independent/train25_probs_mu_best_softmax/" --run $i'


# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg_pytorch --grammar grammars/regression_torch.pybnf --parameters parameters/auto_params.yaml --seed $i --probs_update "standard" --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/standard/pagie/" --run $i'


# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.parity_5 --grammar grammars/5_bit_parity_grammar.txt --parameters parameters/auto_params.yaml --seed $i --probs_update "standard" --experiment_name "/media/cdv/nvme980pro/jessica/vae_experiments/standard/5parity/" --run $i'



# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.symreg --grammar grammars/regression.pybnf --parameters parameters/pagie_dependency.yml --seed $i --probs_update "common_subtree" --experiment_name "common_subtree/pagie/dependent_force_best_different_worst_different/" --learning_factor 0.01 --run $i'

# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.multiplexer_11 --grammar grammars/mux11_grammar.txt --parameters parameters/pagie_dependency.yml --seed $i --probs_update "common_subtree" --experiment_name "common_subtree/11multiplexer/previous_rule_only/" --learning_factor 0.01 --run $i'

# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.multiplexer_11 --grammar grammars/mux11_grammar.txt --parameters parameters/pagie_dependency.yml --seed $i --probs_update "standard" --experiment_name "standard/11multiplexer/" --learning_factor 0.01 --run $i'




# seq 30 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.parity_5 --grammar grammars/5_bit_parity_grammar.txt --parameters parameters/pagie_dependency.yml --seed $i --probs_update "common_subtree" --experiment_name "common_subtree/5parity/dependent_force_best_different_worst_different/1error/" --learning_factor 0.01 --run $i'



# seq 50 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.progsys_mirror_image --grammar grammars/progsys/Mirror\ Image.bnf --parameters parameters/params_progsys.yml --seed $i --probs_update "standard" --experiment_name "standard/progsys/Mirror-Image/" --learning_factor 0.01 --run $i'

# seq 50 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.progsys_mirror_image --grammar grammars/progsys/Mirror\ Image.bnf --parameters parameters/params_progsys.yml --seed $i --probs_update "subtree_dependent" --experiment_name "subtree_dependent/progsys/Mirror-Image/nbest_5/up_2/down_3/" --learning_factor 0.001 --run $i --n_best 5 --levels_up 2 --levels_down 3'

# seq 50 | parallel -j $parallel_processes  '
#     i={}
#     python -m examples.progsys_mirror_image --grammar grammars/progsys/Mirror\ Image.bnf --parameters parameters/params_progsys.yml --seed $i --probs_update "subtree_dependent" --experiment_name "subtree_dependent/progsys/Mirror-Image/nbest_5/up_2/down_4/" --learning_factor 0.001 --run $i --n_best 5 --levels_up 2 --levels_down 4'

# seq 50 | parallel -j $parallel_processes '
#     i={}
#     python -m examples.progsys_mirror_image --grammar grammars/progsys/Mirror\ Image.bnf --parameters parameters/params_progsys.yml --seed $i --probs_update "subtree_dependent" --experiment_name "subtree_dependent/progsys/Mirror-Image/nbest_5/up_2/down_5/" --learning_factor 0.001 --run $i --n_best 5 --levels_up 2 --levels_down 5'

# seq 50 | parallel -j $parallel_processes  '
#     i={}
#     python -m examples.progsys_mirror_image --grammar grammars/progsys/Mirror\ Image.bnf --parameters parameters/params_progsys.yml --seed $i --probs_update "subtree_dependent" --experiment_name "subtree_dependent/progsys/Mirror-Image/nbest_5/up_2/down_6/" --learning_factor 0.001 --run $i --n_best 5 --levels_up 2 --levels_down 6'

# seq 50 | parallel -j $parallel_processes  '
#     i={}
#     python -m examples.progsys_mirror_image --grammar grammars/progsys/Mirror\ Image.bnf --parameters parameters/params_progsys.yml --seed $i --probs_update "subtree_dependent" --experiment_name "subtree_dependent/progsys/Mirror-Image/nbest_5/up_2/down_7/" --learning_factor 0.001 --run $i --n_best 5 --levels_up 2 --levels_down 7'
