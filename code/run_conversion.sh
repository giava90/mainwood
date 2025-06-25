#!/bin/bash

# python convert_data.py WOOD True 3 False Entlebuch
# python convert_data.py All True 1 False All
# Load necessary modules
module load stack/2024-06 python/3.12.8  # Load Python 3.12.8 from the specified stack
module load stack/2024-06 openjdk/21.0.3_9  # Load OpenJDK 21.0.3_9 from the specified stack
# Define the number of cores to use
n_cores=5  # Adjust this value as needed
mem_per_cpu=3000 #megabytes per cpu
management_scenario='WOOD'
case_study='Entlebuch'
use_sample='True'
save_intermediate='False'
walltime='8:00:00'
# Submit the job to SLURM using sbatch
# -c specifies the number of CPU cores
# --wrap allows executing a command within sbatch
# --mem-per-cpu specifies the memory per cpu in megabytes
sbatch -c $n_cores --time=$walltime --mem-per-cpu=$mem_per_cpu --wrap="python convert_data.py $management_scenario $use_sample $n_cores $save_intermediate $case_study"
