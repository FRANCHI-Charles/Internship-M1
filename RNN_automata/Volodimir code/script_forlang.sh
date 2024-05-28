#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -p LONG,LONG2
#SBATCH -D /home/miv09159/ForLang
#SBATCH -o output_%j.out
#SBATCH -J sparql_llm
source /home_expes/tools/python/python3102_0_gpu_base/bin/activate
srun python3 cluster_main.py

