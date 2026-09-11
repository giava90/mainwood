#!/bin/bash
# Stage 1 on Euler: ForClim output -> SorSim tree lists -> SorSim assortments.
#
# Usage:  ./run_conversion.sh <scenario> <case_study> [cohort] [use_sample] [save_intermediate]
#
#   ./run_conversion.sh WOOD Entlebuch                 # dead cohort, all files
#   ./run_conversion.sh WOOD Entlebuch alive           # alive cohort, all files
#   ./run_conversion.sh WOOD Entlebuch dead True       # 50-file smoke test
#   N_CORES=8 CONVERT_WALLTIME=24:00:00 ./run_conversion.sh ALL All
#
# Nothing in this file needs editing to change machine, region or scenario:
# paths live in code/local.env (git-ignored), everything else is an argument.
set -euo pipefail

cd "$(dirname "$0")"

# Per-machine settings. Exported so the Python entry point sees the same paths.
# Per-machine settings. Environment wins over local.env, matching code/paths.py,
# so a one-off override on the command line is honoured.
# shellcheck disable=SC1091
. ./load_env.sh
load_local_env

# Arguments first, so a typo fails here rather than after the modules load.
management_scenario="${1:?scenario required: BAU | WOOD | HYBRID | BIO | ALL}"
case_study="${2:?case study required: Entlebuch | Vaud | Surselva | Misox | All}"
cohort="${3:-dead}"               # dead (harvested trees) | alive (standing stock)
use_sample="${4:-False}"          # True = first 50 files only
save_intermediate="${5:-False}"   # True = keep the tree list as a .zip

case "$cohort" in
    dead|alive) ;;
    *) echo "cohort must be 'dead' or 'alive', got '$cohort'" >&2; exit 2 ;;
esac

module load stack/2024-06 python/3.12.8
module load stack/2024-06 openjdk/21.0.3_9   # SorSim is a Java jar

n_cores="${N_CORES:-${CONVERT_CORES:-4}}"
mem_per_cpu="${CONVERT_MEM_PER_CPU:-3000}"   # megabytes per cpu
walltime="${CONVERT_WALLTIME:-8:00:00}"
mail_type="${MAIL_TYPE:-BEGIN,END,FAIL}"

echo "stage 1: $case_study / $management_scenario / $cohort cohort"
echo "  ${n_cores} cores, ${mem_per_cpu} MB/cpu, walltime ${walltime}"

sbatch -c "$n_cores" \
       --time="$walltime" \
       --mail-type="$mail_type" \
       --mem-per-cpu="$mem_per_cpu" \
       --job-name="conv-${case_study}-${management_scenario}-${cohort}" \
       --output="slurm-conv-${case_study}-${management_scenario}-${cohort}-%j.out" \
       --wrap="python convert_data.py $management_scenario $use_sample $n_cores $save_intermediate $case_study $cohort"
