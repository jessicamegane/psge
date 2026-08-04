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

# Format: result name|Python module|grammar|optional positional benchmark
PROBLEMS=(
    "quartic_polynomial|examples.symreg_pytorch|grammars/regression_torch.pybnf|quarticpolynomial"
    "boston_housing|examples.bostonhousing_pytorch|grammars/bostonhousing_torch.pybnf|__NONE__"
    "pagie_polynomial|examples.symreg_pytorch|grammars/regression_torch.pybnf|pagiepolynomial"
    "nguyen5_polynomial|examples.symreg_pytorch|grammars/regression_torch.pybnf|nguyen5polynomial"
    "multiplexer_11|examples.multiplexer_11|grammars/mux11_grammar.txt|__NONE__"
    "5parity|examples.parity_5|grammars/5_bit_parity_grammar.txt|__NONE__"
)

run_one() {
    local module="$1"
    local grammar="$2"
    local benchmark="$3"
    local problem="$4"
    local max_depth="$5"
    local seed="$6"

    local command=(python -m "$module")
    if [[ "$benchmark" != "__NONE__" ]]; then
        command+=("$benchmark")
    fi
    command+=(
        --parameters "$PARAMETERS_FILE"
        --grammar "$grammar"
        --seed "$seed"
        --run "$seed"
        --max_tree_depth "$max_depth"
        --experiment_name "$RESULTS_ROOT/depth_$max_depth/$problem"
    )
    "${command[@]}"
}

export -f run_one
export PARAMETERS_FILE RESULTS_ROOT

for problem_spec in "${PROBLEMS[@]}"; do
    IFS='|' read -r problem module grammar benchmark <<< "$problem_spec"

    for max_depth in "${MAX_DEPTHS[@]}"; do
        echo "Running hybrid PSGE/Co-PSGE on $problem at depth $max_depth ($NUM_RUNS runs)"
        seq 1 "$NUM_RUNS" | parallel \
            --jobs "$MAX_JOBS" \
            --halt soon,fail=1 \
            run_one \
            "$module" \
            "$grammar" \
            "$benchmark" \
            "$problem" \
            "$max_depth" \
            {}
    done
done
