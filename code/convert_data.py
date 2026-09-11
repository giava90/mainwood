"""Stage 1 of the pipeline: ForClim output -> SorSim tree list -> SorSim assortments.

For every ForClim file in ``../data/<case_study>/inputs/<scenario>/`` this script

  1. converts it into a SorSim tree list in ``../data/<case_study>/intermediate/<scenario>/``
  2. runs SorSim on that list, writing into ``../data/<case_study>/outputs/<scenario>/``
  3. deletes (or zips) the intermediate file.

Usage:
    python convert_data.py <scenario> <use_sample> <n_cores> <save_intermediate> <case_study> [cohort]

    scenario          BAU | WOOD | HYBRID | BIO | ALL
    use_sample        True (first 50 files only) | False
    n_cores           int, processes for the multiprocessing pool
    save_intermediate True (zip the tree list) | False (delete it)
    case_study        Entlebuch | Vaud | Surselva | Misox | All
    cohort            dead (default) | alive

Example:
    python convert_data.py WOOD True 4 False Entlebuch
    python convert_data.py BAU False 8 False Entlebuch alive
"""

import os
import subprocess
import zipfile
import sys
import datetime as dt
from multiprocessing import Pool, Manager

import paths
from naming import (
    COHORTS,
    intermediate_filename,
    parse_forclim_filename,
    sorsim_output_filename,
)


def get_files_in_folder(folder_path):
    """Returns a list of file names in the specified folder."""
    print("Looking for files in folder path", folder_path)
    return [entry.name for entry in os.scandir(folder_path) if entry.is_file()]


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


def convert_forclim(file, input_folder_path, output_folder_path, case_study, management_scenario, failed, cohort="dead"):
    """Converts one ForClim output file into a SorSim tree list.

    The cohort decides both which column of the ForClim file holds the number of
    trees (``dtrees`` for dead, ``trees`` for alive) and the name of the
    intermediate file, so that the alive and dead runs of the same stand never
    overwrite each other.
    """
    parsed = parse_forclim_filename(file, case_study, cohort)
    if parsed is None:
        print(f"Skipping {file}: does not match the {cohort}-cohort naming convention.")
        failed.append(file)
        return
    stand, simtype = parsed
    dead_cohorts = "True" if cohort == "dead" else "False"
    command = (f"python ../minimal/output_input_converter.py {input_folder_path} {file} "
               f"{output_folder_path}/intermediate/{management_scenario}/ "
               f"{intermediate_filename(stand, simtype, cohort)} {dead_cohorts}")
    if not run_command(command):
        failed.append(file)  # Store failed file


def run_sorsim(file, output_folder_path, case_study, management_scenario, failed, save_intermediate=False, cohort="dead"):
    """Runs SorSim on the tree list produced by :func:`convert_forclim`.

    Args:
        file (str): The ForClim file name this tree list came from.
        output_folder_path (str): ``../data/<case_study>/``.
        case_study (str): Region name.
        management_scenario (str): BAU / WOOD / HYBRID / BIO.
        failed (list): Shared list of files that failed an earlier step.
        save_intermediate (str): "True" to zip the tree list, otherwise delete it.
        cohort (str): "dead" or "alive".
    """
    if file in failed:
        return  # Skip failed files
    parsed = parse_forclim_filename(file, case_study, cohort)
    if parsed is None:
        return  # already reported by convert_forclim
    stand, simtype = parsed
    intermediate_path = (f"{output_folder_path}/intermediate/{management_scenario}/"
                         f"{intermediate_filename(stand, simtype, cohort)}")
    command = (f"python ../minimal/run_sorsim.py ../minimal/sorsim/SorSim4Python.jar "
               f"{intermediate_path} "
               f"{output_folder_path}/outputs/{management_scenario}/"
               f"{sorsim_output_filename(stand, simtype, cohort)} 6 True")
    if not run_command(command):
        failed.append(file)  # Store failed file
    elif save_intermediate == "True":
        compress_file(stand, simtype, output_folder_path, management_scenario, cohort)
    else:
        os.remove(intermediate_path)


def process_files(files, input_folder_path, output_folder_path, case_study, management_scenario, num_cores=4, sample=False, save_intermediate=False, cohort="dead"):
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
        cohort (str): "dead" or "alive" cohort of the ForClim simulation.

    Returns:
        list: List of files that failed to process."""
    with Manager() as manager:
        failed = manager.list()

        if sample == "True":
            sample_size = min(len(files), 50)
            files = files[:sample_size]

        # Step 1: Convert ForClim Output in Parallel
        with Pool(processes=num_cores) as pool:
            pool.starmap(convert_forclim, [(file, input_folder_path, output_folder_path, case_study, management_scenario, failed, cohort) for file in files])
        # Step 2: Run SorSim in Parallel
        with Pool(processes=num_cores) as pool:
            pool.starmap(run_sorsim, [(file, output_folder_path, case_study, management_scenario, failed, save_intermediate, cohort) for file in files])

        return list(failed)


def parse_filename(filename, case_study, management_scenario=None, cohort="dead"):
    """Extracts stand and simtype from a ForClim file name.

    Thin wrapper kept so older call sites keep working; the naming rules live in
    :mod:`naming`. ``management_scenario`` is ignored -- BIO file names match the
    same pattern as every other scenario, and the old BIO-only branch could not
    cope with planted variants.
    """
    return parse_forclim_filename(filename, case_study, cohort)


def compress_file(stand, simtype, output_folder_path, management_scenario, cohort="dead"):
    """Compresses and removes the intermediate CSV file."""
    file_path = (f"{output_folder_path}/intermediate/{management_scenario}/"
                 f"{intermediate_filename(stand, simtype, cohort)}")
    zip_path = file_path + ".zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))
    os.remove(file_path)


def process_combination(cs, ms, input_folder_path, output_folder_path, num_cores, use_sample, save_intermediate, start, cohort="dead"):
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
    cohort : str
        "dead" or "alive" cohort of the ForClim simulation.

    Returns
    -------
    List[str]
        A list of file names that failed to process.
    """

    files = get_files_in_folder(input_folder_path)
    print("We have", len(files), "files in the input folder (if sample == True, only 50 are used)")

    # keep only the files that belong to this cohort and region: a leftover .zip
    # or an alive file sitting next to the dead ones would otherwise be handed to
    # SorSim and fail halfway through the run
    files = [f for f in files if parse_forclim_filename(f, cs, cohort) is not None]
    print("Of those,", len(files), f"match the {cohort}-cohort naming convention")

    failed_files = process_files(
        files,
        input_folder_path,
        output_folder_path,
        cs,
        ms,
        num_cores=num_cores,
        sample=use_sample,
        save_intermediate=save_intermediate,
        cohort=cohort,
    )
    print("The following files were not processed correctly...")
    for f in failed_files:
        print(f)
    print("Time taken:", dt.datetime.now() - start)
    return failed_files


if __name__ == "__main__":
    start_time = dt.datetime.now()
    # Parse arguments
    management_scenario = sys.argv[1]
    use_sample = sys.argv[2]
    num_cores = int(sys.argv[3])
    save_intermediate = sys.argv[4]
    case_study = sys.argv[5]
    cohort = sys.argv[6] if len(sys.argv) > 6 else "dead"
    print("Processing data for management scenario ", management_scenario)
    print("Case study ", case_study)
    print("Cohort ", cohort)
    print("Number of cores used ", num_cores)
    print("Using a sample ", use_sample)
    print("Save intermediate files as zip ", save_intermediate)
    # check that the argument is valid
    valid_management_scenarios = ["BAU", "WOOD", "HYBRID", "ALL", "BIO"]
    valid_case_studies = ["Entlebuch", "Vaud", "Surselva", "All", "Misox"]
    if case_study not in valid_case_studies:
        raise ValueError(f"Invalid case study. Please provide a valid case study {valid_case_studies}.")
    if management_scenario not in valid_management_scenarios:
        raise ValueError(f"Invalid management scenario. Please provide a valid management scenario {valid_management_scenarios}.")
    if use_sample not in ["True", "False"]:
        raise ValueError("Invalid argument for use_sample. Please provide True or False.")
    if cohort not in COHORTS:
        raise ValueError(f"Invalid cohort. Please provide one of {list(COHORTS)}.")
    if use_sample == "True":
        print("Using sample data...")

    # Select what to run
    case_studies_to_run = [cs for cs in valid_case_studies if cs != "All"] if case_study == "All" else [case_study]
    scenarios_to_run = [ms for ms in valid_management_scenarios if ms != "ALL"] if management_scenario == "ALL" else [management_scenario]

    local_env = paths.load_local_env()

    for cs in case_studies_to_run:
        for ms in scenarios_to_run:
            # Paths come from the environment / code/local.env, never from an edit
            # to this file -- see code/paths.py and docs/03-runbook.md section 0.
            input_folder_path = paths.input_folder(cs, ms, cohort, local_env)
            output_folder_path = paths.output_folder(cs, ms, cohort, local_env)
            print(f"Paths for {cs}/{ms}/{cohort}:")
            print(paths.describe(cs, ms, cohort))
            paths.ensure_output_tree(output_folder_path, ms)
            failed = process_combination(
                cs, ms,
                input_folder_path,
                output_folder_path,
                num_cores=num_cores,
                use_sample=use_sample,
                save_intermediate=save_intermediate,
                start=start_time,
                cohort=cohort,
            )
