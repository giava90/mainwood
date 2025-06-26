#!/bin/bash

# Load necessary modules
module load stack/2024-06 python/3.12.8  # Load Python 3.12.8 from the specified stack

# python summarize_and_create_plots.py All ALL /cluster/scratch/giacomov/mainwood/ 1
# Define the number of cores to use
n_cores=1  # Adjust this value as needed
mem_per_cpu=20000 #megabytes per cpu
case_study='All'
management_scenario='ALL'
folder_data='/cluster/scratch/giacomov/mainwood/'
walltime='8:00:00'
# Submit the job to SLURM using sbatch
# -c specifies the number of CPU cores
# --wrap allows executing a command within sbatch
# --mem-per-cpu specifies the memory per cpu in megabytes
sbatch -c $n_cores --time=$walltime --mem-per-cpu=$mem_per_cpu --wrap="python summarize_and_create_plots.py $case_study $management_scenario $folder_data $n_cores"
