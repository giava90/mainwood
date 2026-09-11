#!/bin/bash
# Submit a stage 1 scaling grid: every (cores x samples) combination as its own job.
#
# Usage:  ./scaling_benchmark.sh <scenario> <case_study> [cohort]
#
#   ./scaling_benchmark.sh BAU Jurapark
#   SCALING_CORES="5 10 20" SCALING_SAMPLES="40 80 200" ./scaling_benchmark.sh BAU Jurapark
#   SCALING_CORES="5 20" ./scaling_benchmark.sh BAU Jurapark alive
#
# Defaults are 3 core counts x 3 sample sizes = 9 jobs, each with an 8 hour
# walltime. Every point gets its own output tree, because nine concurrent stage 1
# jobs writing into one outputs/<scenario>/ would overwrite each other's files and
# time each other's I/O -- that would measure the collision, not the code.
#
# When they have all finished:
#   python plot_scaling.py
set -euo pipefail

cd "$(dirname "$0")"

# Arguments first, so a typo fails here rather than after the modules load.
management_scenario="${1:?scenario required: BAU | WOOD | HYBRID | BIO}"
case_study="${2:?case study required: Entlebuch | Vaud | Surselva | Misox | Jurapark}"
cohort="${3:-dead}"

case "$cohort" in
    dead|alive) ;;
    *) echo "cohort must be 'dead' or 'alive', got '$cohort'" >&2; exit 2 ;;
esac

# Per-machine settings. Environment wins over local.env, matching code/paths.py.
# shellcheck disable=SC1091
. ./load_env.sh
load_local_env

module load stack/2024-06 python/3.12.8
module load stack/2024-06 openjdk/21.0.3_9   # SorSim is a Java jar

cores_list="${SCALING_CORES:-5 10 20}"
samples_list="${SCALING_SAMPLES:-40 80 200}"
walltime="${SCALING_WALLTIME:-8:00:00}"
mem_per_cpu="${CONVERT_MEM_PER_CPU:-3000}"
mail_type="${MAIL_TYPE:-FAIL}"

# Kept off the summary folder so a benchmark never sits among the deliverables.
default_root="${MAINWOOD_DATA_ROOT:-../data}/scaling_${case_study}_${management_scenario}_${cohort}"
bench_root="${SCALING_ROOT:-$default_root}"

mkdir -p "$bench_root/results"

echo "scaling grid: $case_study / $management_scenario / $cohort cohort"
echo "  cores    : $cores_list"
echo "  samples  : $samples_list"
echo "  walltime : $walltime per job"
echo "  results  : $bench_root/results"
echo

submitted=0
for cores in $cores_list; do
    for samples in $samples_list; do
        job_name="scale-${case_study}-${cores}c-${samples}s"
        sbatch -c "$cores" \
               --time="$walltime" \
               --mem-per-cpu="$mem_per_cpu" \
               --mail-type="$mail_type" \
               --job-name="$job_name" \
               --output="${bench_root}/results/${job_name}-%j.out" \
               --wrap="python scaling_run.py $management_scenario $case_study $cohort $cores $samples $bench_root"
        submitted=$((submitted + 1))
    done
done

echo
echo "$submitted job(s) submitted. Watch them with:  squeue -u \$USER"
echo "When they have all finished:"
echo "  python plot_scaling.py --root '$bench_root'"
