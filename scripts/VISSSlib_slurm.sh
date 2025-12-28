#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=pq2



# everything but metarotation
echo "python -m VISSSlib products.submitAll /projekt1/ag_maahn/VISSS_config/$1 $2 /projekt6/ag_maahn/visss_task_queue_1.2"
python -m VISSSlib products.submitAll /projekt1/ag_maahn/VISSS_config/$1 $2 /projekt6/ag_maahn/visss_task_queue_1.2
