#!/bin/bash

# Array of seeds to use
# seeds=(42 123 456 789 101112 5 17 8 99 10)
# seeds=(43 124 457 790 101113 6 18 9 100 11)
seeds=(20 21 22 23 24 25 26 27 28 29)
# Run the Python script with each seed
for seed in "${seeds[@]}"
do
    echo "Running experiment with seed $seed"
    python accuracy_retraining.py --seed $seed
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed run with seed $seed"
    else
        echo "ERROR: Run with seed $seed failed"
    fi
    echo "------------------------"
done