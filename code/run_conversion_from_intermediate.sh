#!/bin/bash
# Re-run SorSim from saved tree lists, without redoing the ForClim conversion.
#
# Use this when intermediate/ is not empty -- either save_intermediate was True, or
# a stage 1 job died between the two pool steps. The cohort is read back from each
# tree list name, so a folder holding both dead and alive lists is handled in one pass.
#
# Usage:  ./run_conversion_from_intermediate.sh <scenario> <case_study> [use_sample] [save_intermediate]
#
#   ./run_conversion_from_intermediate.sh WOOD Surselva
#   N_CORES=16 ./run_conversion_from_intermediate.sh ALL All
set -euo pipefail

cd "$(dirname "$0")"

# Arguments first, so a typo fails here rather than after the modules load.
management_scenario="${1:?scenario required: BAU | WOOD | HYBRID | BIO | ALL}"
case_study="${2:?case study required: Entlebuch | Vaud | Surselva | Misox | Jurapark | All}"
use_sample="${3:-False}"
save_intermediate="${4:-False}"

# Per-machine settings. Environment wins over local.env, matching code/paths.py.
# shellcheck disable=SC1091
. ./load_env.sh
load_local_env

module load stack/2024-06 python/3.12.8
module load stack/2024-06 openjdk/21.0.3_9   # SorSim is a Java jar

n_cores="${N_CORES:-${CONVERT_CORES:-4}}"
mem_per_cpu="${CONVERT_MEM_PER_CPU:-3000}"
walltime="${CONVERT_WALLTIME:-8:00:00}"
mail_type="${MAIL_TYPE:-BEGIN,END,FAIL}"

echo "re-run SorSim: $case_study / $management_scenario"
echo "  reading ${MAINWOOD_INTERMEDIATE_TEMPLATE:-<default ../data/{case_study}/intermediate/{scenario}/>}"
echo "  ${n_cores} cores, ${mem_per_cpu} MB/cpu, walltime ${walltime}"

sbatch -c "$n_cores" \
       --time="$walltime" \
       --mail-type="$mail_type" \
       --mem-per-cpu="$mem_per_cpu" \
       --job-name="redo-${case_study}-${management_scenario}" \
       --output="slurm-redo-${case_study}-${management_scenario}-%j.out" \
       --wrap="python convert_data_from_intermediate.py $management_scenario $use_sample $n_cores $save_intermediate $case_study"
