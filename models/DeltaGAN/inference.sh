#!/bin/bash
#PBS -N GPU_job_test
#PBS -l walltime=00:59:30
#PBS -A niwa03712
#PBS -q a100_devq
#PBS -l select=1:ncpus=10:ngpus=1:mem=120GB:nodepool=a100p

# LOG_DIR is passed from submission script via -v flag
# Redirect stdout and stderr to named log files
exec > "${LOG_DIR}/${CONFIG_NAME}_output.log" 2> "${LOG_DIR}/${CONFIG_NAME}_error.log"


echo "Running on node: $(hostname)"
echo "Config file: $CONFIG_FILE"
echo "Log directory: $LOG_DIR"
echo "Loaded modules:"
echo "NVIDIA-SMI output:"
nvidia-smi

cd /esi/project/niwa03712/rampaln/pixi_environments/pytorch_env/

# 7 hr 51 for 229 epochs.
# Load modules
.  /opt/niwa/profile/spack_v0.23_2025.05.1.sh

# Set up CUDA environment
export LD_LIBRARY_PATH=/home/rampaln/.conda/envs/tf25/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/home/rampaln/.conda/envs/tf25/lib
export PATH=$CUDA_HOME/bin:$PATH

export CUDA_HOME=/home/rampaln/.conda/envs/tf25
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Run Python with config file argument
/home/rampaln/.conda/envs/tf25/bin/python "/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaGAN/model_inference.py" "$CONFIG_FILE"