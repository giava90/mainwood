#!/bin/bash
# Stage 2 on Euler: SorSim assortments -> summaries_for_plots/<region>_<scenario>.csv + figures.
#
# Interactive equivalent (a sample of 100 files is enough to check a new region):
# python summarize_and_create_plots.py Entlebuch BAU ../data 4 100
# python summarize_and_create_plots.py Entlebuch BAU ../data 4 False alive

# Load necessary modules
module load stack/2024-06 python/3.12.8  # Load Python 3.12.8 from the specified stack

# Define the number of cores to use
n_cores=1  # Adjust this value as needed
mem_per_cpu=20000 #megabytes per cpu
case_study='All'
management_scenario='ALL'
cohort='dead'      # 'dead' (harvested trees) or 'alive' (standing stock)
folder_data='/cluster/scratch/giacomov/mainwood/'
walltime='8:00:00'

use_sample=100 # 'False' or an integer value
# Submit the job to SLURM using sbatch
# -c specifies the number of CPU cores
# --wrap allows executing a command within sbatch
# --mem-per-cpu specifies the memory per cpu in megabytes
sbatch -c $n_cores --time=$walltime --mail-type=BEGIN,END,FAIL  --mem-per-cpu=$mem_per_cpu --wrap="python summarize_and_create_plots.py $case_study $management_scenario $folder_data $n_cores $use_sample $cohort"
