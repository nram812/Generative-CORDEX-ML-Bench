#!/bin/bash
#PBS -N flow_inference
#PBS -l walltime=23:59:30
#PBS -A niwa03712
#PBS -q a100q
#PBS -l select=1:ncpus=10:ngpus=1:mem=120GB:nodepool=a100p
#PBS -koed

echo "Running on node: $(hostname)"
echo "Region: $1"
echo "Experiment: $2"
echo "Log directory: $LOG_DIR"
nvidia-smi

cd /esi/project/niwa03712/rampaln/pixi_environments/pytorch_env/

. /opt/niwa/profile/spack_v0.23_2025.05.1.sh

export CUDA_HOME=/home/rampaln/.conda/envs/tf25
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

/home/rampaln/.conda/envs/tf25/bin/python \
    "/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/FlowMatching/model_inference.py" \
    "$REGION" "$EXPERIMENT"

#    /home/rampaln/.conda/envs/tf25/bin/python \
#    ".../model_inference.py" \
#    "$REGION" "$EXPERIMENT"