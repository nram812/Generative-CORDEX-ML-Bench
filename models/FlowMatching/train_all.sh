#!/bin/bash

# Configuration
regions=("ALPS" "SA" "NZ")
experiments=("Emul_hist_future" "ESD_pseudo_reality")

for region in "${regions[@]}"; do
    for experiment in "${experiments[@]}"; do
        CONFIG_DIR="/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/${region}/configs/${experiment}"
        LOG_DIR="/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/${region}/DECAY_LOGS_flow"

        if [ ! -d "$CONFIG_DIR" ]; then
            continue
        fi

        mkdir -p "$LOG_DIR"

        for config_file in "$CONFIG_DIR"/*_orog.json; do
            [ -e "$config_file" ] || continue

            # Skip no_orog — glob *_orog.json also catches *_no_orog.json
            if [[ "$config_file" == *"_no_orog"* ]]; then
                continue
            fi

            config_name=$(basename "$config_file" .json)
            echo "Submitting job: $config_name"

            qsub -v CONFIG_FILE="$config_file",CONFIG_NAME="$config_name",LOG_DIR="$LOG_DIR" \
                 /esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/FlowMatching/train_model.sh
        done
    done
done

echo "All orog jobs submitted."