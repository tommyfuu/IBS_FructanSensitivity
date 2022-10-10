#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 15:00
#
# Set output file
#BSUB -o  basicModels.%I.out
#
# Set error file
#BSUB -eo basicModels.%I.stderr
#
# Specify node group
#BSUB -m "ly-gpu"
#BSUB -q gpuqueue
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=24]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "basicModels"
source ~/.bashrc
module load cuda/10.1
conda activate vir-env
# TODO: ask madison about the syntax and where on documentation to find it
python runModels.py 
