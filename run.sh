#!/bin/bash

elements=("t1" "t1ce" "t2" "flair")

for SOURCE_MODALITY in "${elements[@]}"; do
    for TARGET_MODALITY in "${elements[@]}"; do
        if [ "$SOURCE_MODALITY" != "$TARGET_MODALITY" ]; then
            sbatch run_experiments.sh $SOURCE_MODALITY $TARGET_MODALITY
        fi
    done
done
