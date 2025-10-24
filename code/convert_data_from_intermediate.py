import os
import subprocess
import zipfile
import sys
import datetime as dt
from multiprocessing import Pool, Manager
import re

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

def convert_forclim(file, input_folder_path, output_folder_path, case_study, management_scenario, failed):
    stand, simtype = parse_filename(file, case_study, management_scenario)
    command = (f"python ../minimal/output_input_converter.py {input_folder_path} {file} "
               f"{output_folder_path}/intermediate/{management_scenario}/ "
               f"deadCohorts{stand}_{simtype}.csv True")
    if not run_command(command):
        failed.append(file)  # Store failed file

def run_sorsim(file, output_folder_path, case_study, management_scenario, failed, save_intermediate=False):
    """ Running the SorSim script.
    Args:

    """
    if file in failed:
        return  # Skip failed files
    stand, simtype = parse_filename(file, case_study, management_scenario)
    command = (f"python ../minimal/run_sorsim.py ../minimal/sorsim/SorSim4Python.jar "
               f"{output_folder_path}/intermediate/{management_scenario}/deadCohorts{stand}_{simtype}.csv "
               f"{output_folder_path}/outputs/{management_scenario}/sorsim_output{stand}_{simtype}.csv 6 True")
    if not run_command(command):
        failed.append(file)  # Store failed file
    elif save_intermediate == "True":
        compress_file(stand, simtype, management_scenario)
    else:
        file_path = f"{output_folder_path}/intermediate/{management_scenario}/deadCohorts{stand}_{simtype}.csv"
        os.remove(file_path)

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
    with Manager() as manager:
        failed = manager.list()

        if sample == "True":
            sample_size = min(len(files), 50)
            files = files[:sample_size]
        
        # Step 2: Run SorSim in Parallel
        with Pool(processes=num_cores) as pool:
            pool.starmap(run_sorsim, [(file, output_folder_path, case_study, management_scenario, failed, save_intermediate) for file in files])

        return list(failed)



def parse_filename(filename, case_study, management_scenario):
    """Extracts stand and simtype from the filename."""
    if management_scenario == "BIO":
        stand, simtype = filename.split("_")
        stand = stand.replace("deadCohorts", "")  # Extract numeric stand ID
        simtype = simtype.replace(".csv", "")   # Extract first letter of simtype
        return stand, simtype
    else:
        parts = filename.split("_") 
        stand = parts[0].replace("deadCohorts", "")
        simtype = ""
        for p in parts[1:]:
            simtype += p+"_"
        simtype = simtype.replace(".csv_", "")
        return stand, simtype
  
def compress_file(stand, simtype, output_folder_path, management_scenario):
    """Compresses and removes the intermediate CSV file."""
    file_path = f"{output_folder_path}/intermediate/{management_scenario}/deadCohorts{stand}_{simtype}.csv"
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
    
    #keep only files that have the 

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
    valid_management_scenarios = ["BAU", "WOOD", "HYBRID", "ALL", "BIO"]
    valid_case_studies = ["Entlebuch", "Vaud", "Surselva", "All"]
    if case_study not in valid_case_studies:
        raise ValueError(f"Invalid case study. Please provide a valid case study {valid_case_studies}.")
    if management_scenario not in valid_management_scenarios:
        raise ValueError(f"Invalid management scenario. Please provide a valid management scenario {valid_management_scenarios}.")
    if use_sample not in ["True", "False"]:
        raise ValueError("Invalid argument for use_sample. Please provide True or False.")
    if use_sample == "True":
        print("Using sample data...")
    
    # Select what to run
    case_studies_to_run = [cs for cs in valid_case_studies if cs != "All"] if case_study == "All" else [case_study]
    scenarios_to_run = [ms for ms in valid_management_scenarios if ms != "ALL"] if management_scenario == "ALL" else [management_scenario]

    for cs in case_studies_to_run:
        for ms in scenarios_to_run:
            #input_folder_path = f"/cluster/work/climate/amauri/{cs}/Results/mgmt_{ms}/dead.trees/"
            #output_folder_path = f"/cluster/scratch/giacomov/mainwood/{cs}/"
            input_folder_path = f"../data/{cs}/intermediate/{ms}/"
            output_folder_path = f"../data/{cs}/"
            failed = process_combination(
                cs, ms, 
                input_folder_path, 
                output_folder_path,
                num_cores=num_cores,
                use_sample=use_sample,
                save_intermediate=save_intermediate,
                start=start_time
            )
