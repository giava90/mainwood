#!/bin/bash
# Stage 2 on Euler: SorSim assortments -> summaries_for_plots/ + figures.
#
# Usage:  ./run_analysis.sh <case_study> <scenario> [cohort] [sample_size] [format]
#
#   ./run_analysis.sh Entlebuch WOOD                   # dead, all files, parquet
#   ./run_analysis.sh Entlebuch WOOD alive             # alive cohort
#   ./run_analysis.sh All ALL dead False csv           # everything, CSV output
#   ANALYSIS_MEM_PER_CPU=40000 ./run_analysis.sh Surselva ALL
#
# folder_data comes from MAINWOOD_DATA_ROOT in code/local.env, so this file is
# identical on the laptop and on Euler.
set -euo pipefail

cd "$(dirname "$0")"

# Per-machine settings. Environment wins over local.env, matching code/paths.py,
# so a one-off override on the command line is honoured.
# shellcheck disable=SC1091
. ./load_env.sh
load_local_env

# Arguments first, so a typo fails here rather than after the modules load.
case_study="${1:?case study required: Entlebuch | Vaud | Surselva | Misox | All}"
management_scenario="${2:?scenario required: BAU | WOOD | HYBRID | BIO | ALL}"
cohort="${3:-dead}"          # dead | alive
use_sample="${4:-False}"     # False (all files) or an integer
summary_format="${5:-parquet}"

case "$cohort" in
    dead|alive) ;;
    *) echo "cohort must be 'dead' or 'alive', got '$cohort'" >&2; exit 2 ;;
esac

module load stack/2024-06 python/3.12.8

folder_data="${MAINWOOD_DATA_ROOT:-../data}"
n_cores="${N_CORES:-${ANALYSIS_CORES:-1}}"
mem_per_cpu="${ANALYSIS_MEM_PER_CPU:-20000}"   # megabytes per cpu
walltime="${ANALYSIS_WALLTIME:-8:00:00}"
mail_type="${MAIL_TYPE:-BEGIN,END,FAIL}"

echo "stage 2: $case_study / $management_scenario / $cohort cohort"
echo "  reading $folder_data"
echo "  ${n_cores} cores, ${mem_per_cpu} MB/cpu, walltime ${walltime}"

sbatch -c "$n_cores" \
       --time="$walltime" \
       --mail-type="$mail_type" \
       --mem-per-cpu="$mem_per_cpu" \
       --job-name="summ-${case_study}-${management_scenario}-${cohort}" \
       --output="slurm-summ-${case_study}-${management_scenario}-${cohort}-%j.out" \
       --wrap="python summarize_and_create_plots.py $case_study $management_scenario $folder_data $n_cores $use_sample $cohort $summary_format"
