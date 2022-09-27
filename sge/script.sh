#!/bin/bash

seeds=(3601294368 683917412 1208919537 1629436337 779238096 1216155998 2120024179 3600272074 4118287191 1643356525 4182889281 3445083937 1754613280 1796245274 2904271034 3559738049 217686711 2861030558 1856221505 2160446191 1460319434 2416604709 414233177 2455725808 172153083 1962782296 1521279429 1310373479 2712692902 483329035)

run=1
echo $seeds
for s in ${seeds[*]}; do
    python3 -m examples.a4a_dt_hybrid --experiment_name a4a/psge/ --seed $s --parameters parameters/standard_a4a.yml  --grammar grammars/a4a_dt_hybrid.pybnf --run $run
    run=$((run + 1))
done
