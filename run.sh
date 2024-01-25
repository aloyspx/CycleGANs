#!/bin/bash

elements=(1 2 5 10)

for LAMBDA_SEG_A in "${elements[@]}"; do
    for LAMBDA_SEG_B in "${elements[@]}"; do
        sbatch run_experiments.sh $LAMBDA_SEG_A $LAMBDA_SEG_B
    done
done
