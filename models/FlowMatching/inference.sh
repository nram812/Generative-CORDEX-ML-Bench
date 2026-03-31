#!/bin/bash
# submit_inference_jobs.sh
# Submits one PBS job per region x experiment_type combination

REGIONS=("NZ" "ALPS" "SA")
EXPERIMENT_TYPES=("ESD" "Hist_future")

SCRIPT="/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/FlowMatching/run_inference.sh"
LOG_BASE="/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/logs/inference"

mkdir -p "${LOG_BASE}"

for region in "${REGIONS[@]}"; do
    for experiment in "${EXPERIMENT_TYPES[@]}"; do

        job_name="flow_${region}_${experiment}"
        log_dir="${LOG_BASE}/${region}_${experiment}"
        mkdir -p "${log_dir}"

        echo "Submitting: region=${region} experiment=${experiment}"
        
        qsub \
            -N "${job_name}" \
            -o "${log_dir}/stdout.log" \
            -e "${log_dir}/stderr.log" \
            -v "LOG_DIR=${log_dir},REGION=${region},EXPERIMENT=${experiment}" \
            "${SCRIPT}"

    done
done

echo "All jobs submitted."