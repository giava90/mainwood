"""Re-run SorSim from tree lists that are already in ``intermediate/<scenario>/``.

Same as :mod:`convert_data` but skips the ForClim conversion step -- use it when a
run was interrupted after the tree lists were written (``save_intermediate=True``),
or when SorSim itself failed and the conversion does not need repeating.

Usage:
    python convert_data_from_intermediate.py <scenario> <use_sample> <n_cores> <save_intermediate> <case_study>

The cohort is taken from each tree list name, so no cohort argument is needed.
"""
import os
import subprocess
import zipfile
import sys
import datetime as dt
import time
from multiprocessing import Pool

import exclusions
import paths
import regions
from naming import (
    COHORTS,
    intermediate_filename,
    parse_intermediate_filename,
    sorsim_output_filename,
)

def get_files_in_folder(folder_path):
    """Returns a list of file names in the specified folder."""
    print("Looking for files in folder path", folder_path)
    return [entry.name for entry in os.scandir(folder_path) if entry.is_file()]
    #return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def run_command(command):
    """Executes a shell command and handles errors."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing command:")
        print("Return Code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def run_sorsim(file, output_folder_path, case_study, management_scenario, save_intermediate=False):
    """Runs SorSim on an intermediate tree list that already exists on disk.

    The cohort is read back from the tree list name, so a folder holding both
    ``deadCohorts*`` and ``aliveCohorts*`` files is processed correctly in one go.
    """
    # No shared-state membership test here: it cost one Manager round trip per
    # file, and the manager spawns a thread per connection. See process_files.
    parsed = parse_intermediate_filename(file)
    if parsed is None:
        print(f"Skipping {file}: not an intermediate SorSim tree list.")
        return
    cohort, stand, simtype = parsed
    intermediate_path = (f"{output_folder_path}/intermediate/{management_scenario}/"
                         f"{intermediate_filename(stand, simtype, cohort)}")
    command = (f"python ../minimal/run_sorsim.py ../minimal/sorsim/SorSim4Python.jar "
               f"{intermediate_path} "
               f"{output_folder_path}/outputs/{management_scenario}/"
               f"{sorsim_output_filename(stand, simtype, cohort)} 6 True")
    if not run_command(command):
        return file          # collected by the caller
    elif save_intermediate == "True":
        compress_file(stand, simtype, output_folder_path, management_scenario, cohort)
    else:
        os.remove(intermediate_path)

def process_files(files, input_folder_path, output_folder_path, case_study, management_scenario, num_cores=4, sample=False, save_intermediate = False):
    """Converts ForClim output and runs SorSim for each file in parallel using multiprocessing.
    
    Args:        
        files (list): List of file names to process.
        input_folder_path (str): Path to the input folder containing ForClim output files.
        output_folder_path (str): Path to the output folder for SorSim results.
        case_study (str): Case study to be used.
        management_scenario (str): Management scenario to be used in processing.
        num_cores (int): Number of CPU cores to use for parallel processing.
        sample (bool): Whether to use a sample of the files for processing.
        save_intermediate (bool): Whether to save intermediate files as zip.
    Returns:
        list: List of files that failed to process."""

    if sample == "True":
        files = files[: min(len(files), paths.sample_size())]

    # Failures come back as return values. The Manager list this replaced cost
    # one remote call per file and spawns a thread per connection; at 48 workers
    # over 60,870 files that hit the thread limit outright and every worker
    # blocked forever. See convert_data.process_files.
    sorsim_start = time.perf_counter()
    with Pool(processes=num_cores) as pool:
        results = pool.starmap(run_sorsim, [(file, output_folder_path, case_study, management_scenario, save_intermediate) for file in files])
    sorsim_s = time.perf_counter() - sorsim_start
    print(f"Phase 2 (SorSim): {sorsim_s:,.1f} s for {len(files)} files", flush=True)

    return [f for f in results if f is not None]



def parse_filename(filename, case_study=None, management_scenario=None):
    """Extracts ``(stand, simtype)`` from an intermediate tree list name.

    Kept for backwards compatibility; :func:`naming.parse_intermediate_filename`
    also returns the cohort and is what the pipeline uses.
    """
    parsed = parse_intermediate_filename(filename)
    if parsed is None:
        return None
    _cohort, stand, simtype = parsed
    return stand, simtype

def compress_file(stand, simtype, output_folder_path, management_scenario, cohort="dead"):
    """Compresses and removes the intermediate CSV file."""
    file_path = (f"{output_folder_path}/intermediate/{management_scenario}/"
                 f"{intermediate_filename(stand, simtype, cohort)}")
    zip_path = file_path + ".zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))
    os.remove(file_path)

def process_combination(cs, ms, input_folder_path, output_folder_path, num_cores, use_sample, save_intermediate, start):
    """
    Processes files for a given case study and management scenario combination.

    Parameters
    ----------
    cs : str
        Case study name.
    ms : str
        Management scenario name.
    input_folder_path : str
        path for folder inputs 
    output_folder_path : str
        path for folder outputs 
    num_cores : int
        Number of CPU cores to use during processing.
    use_sample : bool
        Whether to process only a sample of the files.
    save_intermediate : bool
        Whether to save intermediate processing results.
    start : datetime
        Timestamp marking the start of the processing (used for timing output).

    Returns
    -------
    List[str]
        A list of file names that failed to process.
    """
    
    files = get_files_in_folder(input_folder_path)
    print("We have", len(files), "files to be processed (if sample == True, then only 40)")

    # Apply the same exclusion rule as stage 1. Tree lists written before the rule
    # existed can still contain stands that cannot be computed, and re-running
    # SorSim over them would put exactly the rows back that stage 1 now leaves out.
    # The cohort lives in each tree list name, so the report is written per cohort.
    matched = len(files)
    areas = exclusions.load_stand_areas(paths.stand_details_path(cs))
    for cohort in COHORTS:
        def stand_of(name, cohort=cohort):
            parsed = parse_intermediate_filename(name)
            if parsed is None or parsed[0] != cohort:
                return None
            return (parsed[1], parsed[2])

        of_cohort = [f for f in files if stand_of(f) is not None]
        if not of_cohort:
            continue
        kept, rows = exclusions.partition(of_cohort, stand_of, areas, cs, ms, cohort)
        dropped = set(of_cohort) - set(kept)
        if dropped:
            files = [f for f in files if f not in dropped]
        summary = exclusions.summarise(rows, len(kept), len(of_cohort))
        if summary:
            print(summary)
        written = exclusions.write_report(
            exclusions.report_path(output_folder_path, cs, ms, cohort), rows
        )
        if written:
            print("Excluded stands written to", written)
    if len(files) != matched:
        print(f"{len(files)} of {matched} tree lists will be re-run.")

    failed_files = process_files(
        files,
        input_folder_path,
        output_folder_path,
        cs,
        ms,
        num_cores=num_cores,
        sample=use_sample,
        save_intermediate=save_intermediate
    )
    print("The following files were not processed correctly...")
    for f in failed_files:
        print(f)
    print("Time taken:", dt.datetime.now() - start)


if __name__ == "__main__":
    start_time = dt.datetime.now()
    # Parse arguments
    management_scenario = sys.argv[1]
    use_sample = sys.argv[2]
    num_cores = int(sys.argv[3])
    save_intermediate = sys.argv[4]
    case_study = sys.argv[5]
    # folder_path = sys.argv[7]
    print("Processing data for management scenario ", management_scenario)
    print("Case study ", case_study)
    print("Number of cores used ", num_cores)
    print("Using a sample ", use_sample)
    print("Save intermediate files as zip ", save_intermediate)
    # check that the argument is valid
    regions.check_case_study(case_study)
    regions.check_scenario(management_scenario)
    if use_sample not in ["True", "False"]:
        raise ValueError("Invalid argument for use_sample. Please provide True or False.")
    if use_sample == "True":
        print("Using sample data...")
    
    # Select what to run
    case_studies_to_run = regions.resolve_case_studies(case_study)
    scenarios_to_run = regions.resolve_scenarios(management_scenario)

    local_env = paths.load_local_env()

    for cs in case_studies_to_run:
        for ms in scenarios_to_run:
            # Same configuration as stage 1 -- see code/paths.py.
            input_folder_path = paths.intermediate_folder(cs, ms, local_env=local_env)
            output_folder_path = paths.output_folder(cs, ms, local_env=local_env)
            print(f"Re-running SorSim from {input_folder_path} into {output_folder_path}")
            paths.ensure_output_tree(output_folder_path, ms)
            failed = process_combination(
                cs, ms, 
                input_folder_path, 
                output_folder_path,
                num_cores=num_cores,
                use_sample=use_sample,
                save_intermediate=save_intermediate,
                start=start_time
            )
