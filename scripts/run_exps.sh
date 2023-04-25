#!/bin/env bash

# run experiments
WORK_DIR="$(dirname "$0")/../results/$(date +%y%m%d-%H%M%S)"
WORK_DIR=$(realpath "${WORK_DIR}")
mkdir -p "${WORK_DIR}"
echo "Saving results to ${WORK_DIR}"
for config in "OVER_PEAKINESS" "DOUBLY_STOCHASTIC_BETAS" "ABSORBING_CENTER_M1" "ABSORBING_CENTER_M5" "GAUSSIAN_BETAS" 
do
    python -m d3exp ${config} --seed 42 --work_dir "${WORK_DIR}"
done