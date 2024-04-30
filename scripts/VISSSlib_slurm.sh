#!/bin/bash

# sample script to process VISSS data on slurm cluster

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=pq2

#$1 settings file
#$2 nDays to look back 

cd ~/slurm

# all products but metarotation
echo "python -m VISSSlib scripts.loopCreateBatch  $1 $2 1"
python -m VISSSlib scripts.loopCreateBatch  $1 $2 1

# metarotation
echo "python3 -m VISSSlib scripts.loopCreateMetaRotation  $1 $2 1"
python3 -m VISSSlib scripts.loopCreateMetaRotation  $1 $2 1







