#!/bin/bash

# Configuration
regions=("ALPS" "SA" "NZ")
experiments=("Emul_hist_future" "ESD_pseudo_reality")

# Loop through regions and experiments
for region in "${regions[@]}"; do
    for experiment in "${experiments[@]}"; do
        CONFIG_DIR="/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/${region}/configs/${experiment}"
        LOG_DIR="/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/${region}/DM_LOGS"
        echo "CONFIG_DIR: $CONFIG_DIR"

        # Check if CONFIG_DIR exists
        if [ ! -d "$CONFIG_DIR" ]; then
            echo "Warning: CONFIG_DIR does not exist: $CONFIG_DIR"
            echo "Skipping region: $region, experiment: $experiment"
            continue
        fi

        # Create log directory if it doesn't exist
        mkdir -p "$LOG_DIR"

        # Loop through all config files
        for config_file in "$CONFIG_DIR"/*.json; do
            # Skip if no files match the pattern
            [ -e "$config_file" ] || continue

            # Extract filename without path and extension for naming
            config_name=$(basename "$config_file" .json)
            echo "Submitting job for config: $config_name (region: $region, experiment: $experiment)"

            # Submit job with config file path, name, and log directory as environment variables
            qsub -v CONFIG_FILE="$config_file",CONFIG_NAME="$config_name",LOG_DIR="$LOG_DIR" \
                 /esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaDDPM/pbs_test_job.sh
        done
    done
done

echo "All jobs submitted."