#!/usr/bin/env bash

set -euo pipefail

MAX_JOBS="${MAX_JOBS:-16}"
NUM_RUNS="${NUM_RUNS:-30}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments_hybrid_1000_gen}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v parallel >/dev/null 2>&1; then
    echo "GNU Parallel is required but was not found in PATH." >&2
    exit 1
fi

PARAMETERS_FILE="parameters/hybrid_psge_copsge_1000.yaml"
MAX_DEPTHS=(10 15)
SUBTREE_LEVELS_UP="${SUBTREE_LEVELS_UP:-1}"
SUBTREE_LEVELS_DOWN="${SUBTREE_LEVELS_DOWN:-3}"

# Format: name|algorithm method|search strategy|learning strategy|elitism
CONFIGURATIONS=(
    "hybrid_psge_copsge_standard|psge_copsge|standard|independent|100"
    "psge_independent_eda|psge|eda|independent|1"
    "psge_independent_standard|psge|standard|independent|100"
    "psge_depth_based_eda|psge|eda|depth_based|1"
    "psge_depth_based_standard|psge|standard|depth_based|100"
    "psge_subtree_dependent_eda|psge|eda|subtree_dependent|1"
    "psge_subtree_dependent_standard|psge|standard|subtree_dependent|100"
    "psge_context_aware_eda|psge|eda|context_aware|1"
    "psge_context_aware_standard|psge|standard|context_aware|100"
    "psge_context_aware_depth_eda|psge|eda|context_aware_depth|1"
    "psge_context_aware_depth_standard|psge|standard|context_aware_depth|100"
    "psge_context_aware_previous_eda|psge|eda|context_aware_previous|1"
    "psge_context_aware_previous_standard|psge|standard|context_aware_previous|100"
    "copsge_standard|copsge|standard|none|100"
)

# Format: result name|Python module|grammar|optional positional benchmark
PROBLEMS=(
    "quartic_polynomial|examples.symreg|grammars/regression_1var.pybnf|quarticpolynomial"
    "boston_housing|examples.bostonhousing|grammars/bostonhousing.bnf|__NONE__"
    "pagie_polynomial|examples.symreg|grammars/regression.pybnf|pagiepolynomial"
    "nguyen5_polynomial|examples.symreg|grammars/regression_1var.pybnf|nguyen5polynomial"
    "multiplexer_11|examples.multiplexer_11|grammars/mux11_grammar.txt|__NONE__"
    "5parity|examples.parity_5|grammars/5_bit_parity_grammar.txt|__NONE__"
)

run_one() {
    local module="$1"
    local grammar="$2"
    local benchmark="$3"
    local configuration="$4"
    local algorithm_method="$5"
    local search_strategy="$6"
    local learning_strategy="$7"
    local elitism="$8"
    local problem="$9"
    local max_depth="${10}"
    local seed="${11}"

    local command=(python -m "$module")
    if [[ "$benchmark" != "__NONE__" ]]; then
        command+=("$benchmark")
    fi
    command+=(
        --parameters "$PARAMETERS_FILE"
        --grammar "$grammar"
        --seed "$seed"
        --run "$seed"
        --algorithm_method "$algorithm_method"
        --search_strategy "$search_strategy"
        --learning_strategy "$learning_strategy"
        --elitism "$elitism"
        --max_tree_depth "$max_depth"
        --levels_up "$SUBTREE_LEVELS_UP"
        --levels_down "$SUBTREE_LEVELS_DOWN"
        --n_best 1
        --experiment_name "$RESULTS_ROOT/$configuration/depth_$max_depth/$problem"
    )
    "${command[@]}"
}

export -f run_one
export PARAMETERS_FILE RESULTS_ROOT SUBTREE_LEVELS_UP SUBTREE_LEVELS_DOWN

for problem_spec in "${PROBLEMS[@]}"; do
    IFS='|' read -r problem module grammar benchmark <<< "$problem_spec"

    for configuration_spec in "${CONFIGURATIONS[@]}"; do
        IFS='|' read -r configuration algorithm_method search_strategy \
            learning_strategy elitism <<< "$configuration_spec"

        for max_depth in "${MAX_DEPTHS[@]}"; do
            echo "Running $configuration on $problem at depth $max_depth ($NUM_RUNS runs)"
            seq 1 "$NUM_RUNS" | parallel \
                --jobs "$MAX_JOBS" \
                --halt soon,fail=1 \
                run_one \
                "$module" \
                "$grammar" \
                "$benchmark" \
                "$configuration" \
                "$algorithm_method" \
                "$search_strategy" \
                "$learning_strategy" \
                "$elitism" \
                "$problem" \
                "$max_depth" \
                {}
        done
    done
done
