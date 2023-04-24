#!/bin/env bash

# run experiments
for config in "OVER_PEAKINESS" "DOUBLY_STOCHASTIC_BETAS" "ABSORBING_CENTER_M1" "ABSORBING_CENTER_M5" "GAUSSIAN_BETAS" 
do
    python -m d3exp ${config} --seed 42
done