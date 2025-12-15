#!/bin/bash
# Script to run multiple imputation training runs
# Usage: ./multiple_runs.sh [parameters_file] [num_runs]

PARAMS_FILE=${1:-"./datasets/breast/parameters.json"}
NUM_RUNS=${2:-50}

echo "Running $NUM_RUNS imputation runs with parameters: $PARAMS_FILE"

for run in $(seq 1 $NUM_RUNS)
do
    echo "Running run = $run / $NUM_RUNS"
    gainpro gain --parameters "$PARAMS_FILE"
done

echo "Completed all $NUM_RUNS runs"
