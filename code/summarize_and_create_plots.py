import io

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
import datetime as dt
from multiprocessing import Pool
from collections import defaultdict

from summary_io import SUMMARY_FORMATS, write_summary
import paths
import regions

import pdb


from matplotlib.lines import Line2D

def parse_filename(filename: str, scenario: str) -> tuple:
    """
    Parse a SorSim output filename into the metadata carried by that file.

    Parameters:
    - filename: str, e.g. "sorsim_output240_7_planted_01.csv"
                     or "sorsim_alive_output240_7_planted_01.csv"
    - scenario: str, one of "WOOD", "BAU", "HYBRID", "BIO"

    Returns:
    - tuple: (stand, simtype, planted_species, plantation, cohort)
             planted_species is "999" for BIO and for the no-planting variant;
             plantation is True for the single-species plantation naming
             (e.g. "sorsim_output59_PMen_1.csv");
             cohort is "dead" or "alive".
    """
    parts = filename.replace(".csv", "").split("_")

    # Alive-cohort files carry an "alive" flag right after the "sorsim" prefix.
    # Strip it, and everything downstream sees the historical dead-cohort layout.
    cohort = "dead"
    if len(parts) > 1 and parts[1] == "alive":
        cohort = "alive"
        parts = [parts[0]] + parts[2:]

    # Expecting something like: ['sorsim', 'output240', '7', 'planted', '01']
    stand = parts[1].replace("output", "")
    simtype = parts[2]

    if scenario == "BIO":
        return stand, simtype, "999", False, cohort
    else:
        planted_species = parts[-1] if "planted" in parts else None # this case is panted
        if planted_species is None:
            # we are considering plantations with only one species for which the naming is different
            simtype = parts[-1]
            planted_species =  parts[2]
            return stand, simtype, planted_species, True, cohort
        else:
            return stand, simtype, planted_species, False, cohort

def file_cohort(filename: str, scenario: str):
    """Return the cohort a SorSim output belongs to, or None if the name is unreadable.

    Used to filter a folder that holds both cohorts without letting one oddly
    named leftover file abort the whole run.
    """
    try:
        return parse_filename(filename, scenario)[4]
    except (IndexError, ValueError):
        print(f"Warning: cannot read cohort from {filename}, skipping.")
        return None


def load_data(folder_path, management_scenario):
    """
    Reads and combines data from CSV files in the specified folder.

    Args:
        folder_path (str): The path to the folder containing the CSV files.

    Returns:
        pandas.DataFrame: A combined DataFrame containing data from all valid CSV files,
                          or an empty DataFrame if no valid files are found.
    """
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if "assortments_summaries.csv" in files:
        files.remove("assortments_summaries.csv")

    df = pd.DataFrame()
    for file in files:
        filename = os.path.join(folder_path, file)
        if os.path.isfile(filename):
            try:
                stand, simtype, planted_species, plantation, cohort = parse_filename(file, management_scenario)
                # Read the CSV file
                data = pd.read_csv(filename, sep=";", low_memory=False)
                # Identify the row with '#Gruppierungsmerkmal' and set it as the new header
                cut_point = data[data["#ID"] == "#Gruppierungsmerkmal"]
                if not cut_point.empty:
                    cut_point_index = cut_point.index[0]
                    data = data[cut_point_index:]
                    data.columns = data.iloc[0]
                    data = data[1:]
                    data["simtype"] = simtype
                    data["stand"] = stand
                    data["planted_species"] = planted_species
                    data["planting"] = True if planted_species != "999" else False
                    data["plantation"] = plantation
                    data["cohort"] = cohort
                    df = pd.concat([df, data], ignore_index=True)
                else:
                    print(f"Warning: '#Gruppierungsmerkmal' row not found in {file}. Skipping.")
            except ValueError:
                print(f"Warning: Skipping file {file} due to naming convention mismatch.")
            except Exception as e:
                print(f"Warning: Error reading file {file}: {e}")
    return df

def load_data_parallel(folder_path, management_scenario, sample=False, num_cores=1, batch_size=500, chunk_size=None, cohort="dead"):
    """
    Efficiently reads and combines data from CSV files in parallel using multiprocessing.

    This function utilizes all available CPU cores to speed up the reading and processing
    of CSV files. Each file is processed in parallel, and only valid DataFrames are collected.
    Files without the '#Gruppierungsmerkmal' marker or that raise errors are skipped.
    Files named 'assortments_summaries.csv' are also excluded.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
        management_scenario (str): A string used to extract metadata from filenames.
        sample (int): The number of files to sample at random to test. Default is False, meaning that all files are parsed
        num_cores (int): The number of cores used for the multiprocessing. Default is 1.
        cohort (str): "dead" or "alive". Only the SorSim outputs of that cohort are read,
            so a folder holding both cohorts yields one summary per cohort.

    Returns:
        pandas.DataFrame: A combined DataFrame containing processed data from all valid CSV files,
                          or an empty DataFrame if no valid files are found.
    """
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f)) and f != "assortments_summaries.csv"]
    # keep only the requested cohort: dead-cohort names have no flag, alive ones
    # carry "alive" right after the "sorsim" prefix
    files = [f for f in files if file_cohort(f, management_scenario) == cohort]
    print(f"Found {len(files)} files for the {cohort} cohort")
    if sample:
        np.random.seed(42)
        sample_size = min(sample, len(files))
        files = np.random.choice(files, size=sample_size, replace=False)

    args = [(folder_path, f, management_scenario) for f in files]
    if chunk_size is None:
        chunk_size = min(int(len(files)/num_cores), 200)
    with Pool(num_cores) as pool:
        results_iterator = pool.imap_unordered(process_file, args, chunksize=chunk_size)

        #  results = [df for df in results if df is not None]
        data_frames = []
        batch=[]
        for i, df in enumerate(results_iterator, start=1):
            if df is not None:
                batch.append(df)
                # When batch full, concatenate and clear
                if i % batch_size == 0:
                    data_frames.append(pd.concat(batch, ignore_index=True))
                    batch.clear()

        # concatenate any leftovers
        if batch:
            data_frames.append(pd.concat(batch, ignore_index=True))
  
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

#: SorSim writes a long per-tree block, then this marker, then the aggregated block.
#: Only the aggregated block is used, and it is about 2.5% of the file.
AGGREGATED_BLOCK_MARKER = b"\n#Gruppierungsmerkmal"


def process_file(file_path_and_name):
    """
    Processes a single CSV file to extract and transform relevant data.

    The aggregated block is located with a raw byte scan and only that block is
    handed to the CSV parser. Parsing the whole file and slicing afterwards --
    which is what this did until 2026-09 -- cost about 17x more time and forced
    every column to object dtype, because the per-tree block above the marker has
    a different, wider layout.

    The file is decoded as UTF-8 with replacement. SorSim has already written the
    replacement characters into the file itself, so this reproduces the column
    names and species spellings the rest of the pipeline expects (``F�hre``
    -> ``Foehre`` in :func:`preprocess_data`). Decoding as cp1252 would corrupt
    them further.

    Args:
        file_path_and_name (tuple): A tuple containing:
            - file_path (str): The folder path where the file is located.
            - file_name (str): The name of the file to process.
            - management_scenario (str): A string used to extract metadata from the filename.

    Returns:
        pandas.DataFrame or None: A processed DataFrame if successful, or None if the file is invalid.
    """
    file_path, file_name, management_scenario = file_path_and_name
    try:
        stand, simtype, planted_species, plantation, cohort = parse_filename(file_name, management_scenario)
        with open(os.path.join(file_path, file_name), "rb") as handle:
            blob = handle.read()
        position = blob.find(AGGREGATED_BLOCK_MARKER)
        if position == -1:
            return None
        df = pd.read_csv(
            io.BytesIO(blob[position + 1:]),
            sep=";",
            encoding="utf-8",
            encoding_errors="replace",
        )
        df["simtype"] = simtype
        df["stand"] = stand
        df["planted_species"] = planted_species
        df["planting"] =  True if planted_species != "999" else False
        df["plantation"] =  plantation
        df["cohort"] = cohort
        return df
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None


def preprocess_data(df, management_scenario):
    """
    Preprocesses the combined DataFrame by removing unnecessary columns,
    handling missing values, renaming columns with umlauts, and casting volume columns to float.
    Also, if stand_simtype_count is not None, we divide each (stand,simtype) pair by the count observed.
    This division is to normalize per number of simulations per (stand,simtype) pair.

    Args:
        df (pandas.DataFrame): The DataFrame to preprocess.
        management_scenario (str): The management scenario, used to set weights.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty, no preprocessing performed.")
        return df

    # Remove the first character from 'Gruppierungsmerkmal'
    df["Gruppierungsmerkmal"] = df["#Gruppierungsmerkmal"].apply(lambda x: x[1:])

    # Drop the old '#Gruppierungsmerkmal' column
    summaries = df.drop(columns=["#Gruppierungsmerkmal"])
    # Drop rows with NaN in 'Baumart'
    summaries = summaries.dropna(subset=["Baumart"])

    # Drop columns that have nan as column name
    summaries = summaries.loc[:, ~summaries.columns.isna()]

    # Rename columns with umlauts
    summaries.rename({"L�ngenklasse": "Laengenklasse", 'St�rkenklasse': "Staerkenklasse"}, axis=1, inplace=True)

    # Rename rows in 'Baumart' with umlauts and specific replacements
    summaries["Baumart"] = (summaries["Baumart"]
                            .str.replace("�", "oe", regex=False)
                            .str.replace("oebriges", "Ubrige", regex=False))

    # Cast volume and value columns to float
    cols_to_float = ["Volumen IR [m3]", "Volumen OR [m3]", "Wert [CHF]"]
    for col in cols_to_float:
        if col in summaries.columns:
            # assign the whole column, not .loc[:, col]: the latter writes the
            # numbers back into the existing object-dtype column, so every later
            # multiplication and groupby runs element-by-element in Python
            summaries[col] = pd.to_numeric(summaries[col], errors='coerce') # Handle potential conversion errors
        else:
            print(f"Warning: Column '{col}' not found, skipping float conversion.")
    

    # rename the column of summaries "Gruppierungsmerkmal" as "year"
    summaries.rename(columns={"Gruppierungsmerkmal": "year"}, inplace=True)
    # cast years as int
    summaries["year"] = summaries["year"].astype(int)
    summaries = summaries[summaries["year"] >=2020 ]
    
    # we know create a weighting columns to rescale the assortments based on the rules used in ForClim simulations
    if management_scenario == "BIO":
        # eight = 1.0 as there is no planting in BIO
        summaries["weight"] = 1
    if management_scenario != "BIO":
        # add columns called weight with value 0.9 if planting is False, 0.1 otherwise
        summaries["weight"] = np.where(summaries["planting"], 0.1, 0.9)
        # count the number of unique species per (stand, simtype) pair - we could also use the defaultdict
        species_count = (
            summaries.groupby(['stand', 'simtype'])['planted_species']
            .nunique()
            .reset_index(name='species_count')
        )
        # merge the species_count back to summaries
        summaries = summaries.merge(species_count, on=['stand', 'simtype'], how='left', copy=False)
        # If the species count was exactly 1, we put weight 1  
        summaries.loc[summaries['species_count']==1, 'weight'] = 1
        # If the species count is larger than 1 and planting is True, we divide 0.1 by species_count-1
        mask = summaries['planting']
        summaries.loc[mask*summaries['species_count']>1, 'weight'] /= summaries.loc[mask, 'species_count']-1
        # If the species count is larger than 1 and planting is False, we keep 0.9 initialized
        # do nothing
        # if plantation is True, we simple put weights to 1 and then divide by the species count
        mask = summaries['plantation']
        summaries.loc[mask, "weight"]  = 1.0
        summaries.loc[mask, "weight"] /= summaries.loc[mask, "species_count"]
        # drop the species_count column
        summaries.drop(columns=['species_count'], inplace=True)
    # drop the planting column
    summaries = summaries.drop(columns=["planting"])    
    
    # multiply the weight by the value in the "Volumen OR [m3]"  and "Volumen IR [m3]" columns
    summaries["Volumen OR [m3]"] = summaries["Volumen OR [m3]"] * summaries["weight"]
    summaries["Volumen IR [m3]"] = summaries["Volumen IR [m3]"] * summaries["weight"]
    # multiply the value in the "Wert [CHF]" column by the weight
    summaries["Wert [CHF]"] = summaries["Wert [CHF]"] * summaries["weight"]
    # drop the weight column
    summaries = summaries.drop(columns=["weight"])   

    return summaries

#: Low-cardinality text columns worth storing as categories. ``diameter_class`` is
#: deliberately absent: the plots group by it, and a categorical group key makes
#: pandas emit the full product of categories instead of the observed rows.
CATEGORICAL_COLUMNS = ("Baumart", "Laengenklasse", "Staerkenklasse", "stand",
                       "simtype", "planted_species", "cohort")


def compact_dtypes(summaries):
    """Shrinks the summary frame by storing repeated text as categories.

    The frame is one row per (year x species x length class x diameter class x
    stand x planting variant), so a handful of short strings are repeated tens of
    millions of times. Held as Python objects that was ~20 GB for a region the
    size of Surselva, which is why the analysis job asks for 20 GB on one core.

    Call this *after* the per-file frames have been concatenated and after
    :func:`preprocess_data`: concatenating frames with different categories falls
    back to object anyway, and ``preprocess_data`` groups by ``stand``/``simtype``,
    which is slower on categoricals.

    Args:
        summaries (pandas.DataFrame): The preprocessed summaries.

    Returns:
        pandas.DataFrame: The same data with narrower dtypes.
    """
    if summaries.empty:
        return summaries
    before = summaries.memory_usage(deep=True).sum() / 1e6
    for column in CATEGORICAL_COLUMNS:
        if column in summaries.columns and summaries[column].dtype == object:
            summaries[column] = summaries[column].astype("category")
    if "year" in summaries.columns:
        summaries["year"] = summaries["year"].astype("int16")
    after = summaries.memory_usage(deep=True).sum() / 1e6
    print(f"Frame compacted from {before:.0f} MB to {after:.0f} MB")
    return summaries


def augment_with_stand_data(summaries, stand_data):
    """
    Augments the summaries DataFrame with stand information (altitude and area).

    Args:
        summaries (pandas.DataFrame): The preprocessed summaries DataFrame containing a 'stand' column.
        stand_data (pandas.DataFrame): DataFrame containing stand details with 'fsID', 'Above1000m', 'n.patches', and 'area' columns.

    Returns:
        pandas.DataFrame: The summaries DataFrame augmented with 'Above1000m', 'sim_area (m2)', and 'area' columns.
    """
    if summaries.empty or stand_data.empty:
        print("Warning: One or both input DataFrames are empty, no stand data augmentation performed.")
        return summaries

    # Check for missing stands (optional, as the original code did)
    for stand in summaries["stand"].unique():
        if int(stand) not in stand_data["fsID"].values:
            print(f"Stand {stand} not found in stand data")
    #stand_to_else = stand_data[["fsID", "Above1000m", "n.patches", "area"]].copy()
    col_to_keep = ["fsID", "area_ha"]
    if "Above1000m" in  stand_data.columns:
        col_to_keep = col_to_keep + ["Above1000m"]
    stand_to_else = stand_data[col_to_keep].copy()
    stand_to_else.index = stand_to_else["fsID"].astype(str)
    stand_to_else = stand_to_else.drop(columns=["fsID"])
    stand_to_else_dict = stand_to_else.to_dict()

    # Adding area and altitude to summaries
    if "Above1000m" in  stand_data.columns:
        summaries["Above1000m"] = summaries["stand"].astype(str).map(stand_to_else_dict["Above1000m"])
    # computing the simulated area
    #summaries["sim_area (m2)"] = summaries["stand"].astype(str).map(stand_to_else_dict["n.patches"]) * 625
    # the simulated area is always 100 patches of 625m2 each
    summaries["sim_area (m2)"] = 100 * 625
    summaries["area"] = summaries["stand"].astype(str).map(stand_to_else_dict["area_ha"])
    # rescaling the volume according to the actual size of the stand
    # in the simulations, we have 100 patches of 625 m2 each
    # however, the actual area of the stand is different and saved in "area" column
    # hence, we have to divide the "Volumen OR [m3]" by 100 * 625 and multiply by the actual area
    # we have saved the simulated area in "sim_area (m2)" column
    # summaries["stand_productivity"] = summaries["Volumen OR [m3]"] / summaries["sim_area (m2)"]
    summaries["Volumen OR [m3]"] /= summaries["sim_area (m2)"] 
    summaries["Volumen OR [m3]"] *= summaries["area"]*10000
    summaries["Volumen IR [m3]"] /= summaries["sim_area (m2)"] 
    summaries["Volumen IR [m3]"] *= summaries["area"]*10000

    return summaries

def add_sawmill_diameter_info(summaries):
    """
    Adds a boolean column indicating whether the diameter class ('Staerkenklasse')
    is typically used by sawmills.

    Args:
        summaries (pandas.DataFrame): The DataFrame containing a 'Staerkenklasse' column.

    Returns:
        pandas.DataFrame: The DataFrame with an added 'is_for_sawmills_diameter' column.
    """
    if summaries.empty:
        print("Warning: Input DataFrame is empty, no sawmill diameter information added.")
        return summaries
    # the following categorization is more about quality that sawmills.
    # sawmills accept diameter usually diameter above 15 or 18 (see Blumer-Lehmann or tschopp) up to 50 or 60.
    # this categorization could be improved.
    staerken2sawmills_use = {"1a": False, "1b": False, "2a": False, "2b": False, "3a": False, "3b": False,
                             "4": True, "5": True, "6": True, "7": True, "8": True,
                             "Restholz": False}
    diameter_class_mapping = {"1a": "<20cm", "1b": "<20cm", "2a": "20-40cm", "2b": "20-40cm", "3a": "20-40cm", "3b": "20-40cm",
                             "4": ">40cm", "5": ">40cm", "6": ">40cm", "7": ">40cm", "8": ">40cm",
                             "Restholz": "<20cm"}
    # .map over the distinct values instead of .apply over every row. An unknown
    # class still stops the run -- SorSim writes "Restholz 1", "Restholz 2", ...
    # in the per-tree block and only plain "Restholz" in the aggregated one, so a
    # value we have never seen means the file layout changed and the volumes
    # would silently land in the wrong bucket.
    klasse = summaries["Staerkenklasse"]
    unknown = set(klasse.dropna().unique()) - set(staerken2sawmills_use)
    if unknown:
        raise KeyError(f"Unknown Staerkenklasse values: {sorted(unknown)}")
    summaries["is_for_sawmills_diameter"] = klasse.map(staerken2sawmills_use).astype(bool)
    summaries["diameter_class"] = klasse.map(diameter_class_mapping)
    return summaries


def load_quality_data(file_path):
    """
    Loads and preprocesses wood quality data from an Excel file.

    Args:
        file_path (str): The path to the Excel file containing quality data.

    Returns:
        pandas.DataFrame: The preprocessed quality DataFrame.
    """
    try:
        quality_df = pd.read_excel(file_path)
        # Keep only the first 8 columns
        quality_df = quality_df.iloc[:, :8].copy()
        # Rename the columns
        quality_df.columns = ["Baumart", "Wertklasse", "Stammholz-Faktor", "A", "B", "C", "D", "Endnutzung"]
        # When the Baumart is NaN, fill with the value from the previous row
        quality_df["Baumart"] = quality_df["Baumart"].ffill()
        # Fill remaining NaN values with 0
        quality_df = quality_df.fillna(0)
        # Keep only the rows that have Wertklasse == 2
        quality_df = quality_df[quality_df["Wertklasse"] == 2].copy()
        # Drop the columns Wertklasse, Stammholz-Faktor
        quality_df = quality_df.drop(columns=["Wertklasse", "Stammholz-Faktor"])
        # Add a column that sums the values of quality classes A, B, C
        quality_df["For Sawmills"] = quality_df["A"] + quality_df["B"] + quality_df["C"]
        return quality_df
    except FileNotFoundError:
        print(f"Error: Quality data file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading or processing quality data: {e}")
        return pd.DataFrame()

def create_quality_mapping(quality_df):
    """
    Creates a dictionary that maps 'Baumart' (tree species) to the fraction
    of wood suitable for sawmills based on the quality data.

    Args:
        quality_df (pandas.DataFrame): The preprocessed quality DataFrame.

    Returns:
        dict: A dictionary mapping tree species to the fraction for sawmills (values between 0 and 1).
    """
    if quality_df.empty:
        print("Warning: Input quality DataFrame is empty, cannot create quality mapping.")
        return {}

    baumart2fraction = quality_df.set_index("Baumart")["For Sawmills"].to_dict()
    baumart2fraction = {k: v / 100 for k, v in baumart2fraction.items()}
    return baumart2fraction

def map_species_for_quality(df, species_quality_mapping):
    """
    Adds a column 'baumart_for_quality' to the DataFrame by mapping the
    'Baumart' column to a standardized species name used in the quality data.

    Args:
        df (pandas.DataFrame): The DataFrame to add the new column to.
        species_quality_mapping (dict): A dictionary mapping the species names in the DataFrame
                                       to the species names in the quality data.

    Returns:
        pandas.DataFrame: The DataFrame with the added 'baumart_for_quality' column.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty, no species mapping for quality performed.")
        return df

    # map + fillna keeps the "default to the original name" behaviour without a
    # Python call per row
    mapped = df["Baumart"].map(species_quality_mapping)
    df["baumart_for_quality"] = mapped.fillna(df["Baumart"])
    return df

def calculate_biomass_for_sawmills(x, baumart2fraction, biomass_column="Volumen OR [m3]"):
    """
    Calculates the biomass fraction suitable for sawmills based on diameter class and wood quality.

    Args:
        x (pandas.Series): A row of the DataFrame.
        baumart2fraction (dict): A dictionary mapping tree species to the fraction for sawmills.
        biomass_column (str, optional): The name of the column containing biomass.
                                         Defaults to "Volumen OR [m3]".

    Returns:
        float: The biomass (from the specified column) multiplied by the fraction
               suitable for sawmills if the diameter class is relevant for sawmills,
               otherwise 0.
    """
    if x["is_for_sawmills_diameter"]:
        fraction = baumart2fraction.get(x["baumart_for_quality"], 0) # Default to 0 if species not found
        return x[biomass_column] * fraction
    else:
        return 0

def calculate_biomass_not_for_sawmills(x, baumart2fraction, biomass_column="Volumen OR [m3]"):
    """
    Calculates the biomass fraction not suitable for sawmills.

    Args:
        x (pandas.Series): A row of the DataFrame.
        baumart2fraction (dict): A dictionary mapping tree species to the fraction for sawmills.
        biomass_column (str, optional): The name of the column containing biomass.
                                         Defaults to "Volumen OR [m3]".

    Returns:
        float: The biomass (from the specified column) multiplied by (1 - fraction for sawmills)
               if the diameter class is relevant for sawmills, otherwise the original biomass.
    """
    if x["is_for_sawmills_diameter"]:
        fraction = baumart2fraction.get(x["baumart_for_quality"], 0) # Default to 0 if species not found
        return x[biomass_column] * (1 - fraction)
    else:
        return x[biomass_column]

def calculate_sawmill_split(summaries, baumart2fraction, biomass_column="Volumen OR [m3]"):
    """Splits the volume into the share that can go to a sawmill and the rest.

    Vectorised replacement for ``summaries.apply(calculate_biomass_for_sawmills,
    axis=1)``, which built a Series per row and took minutes on a large region.
    The result is identical: only rows in a sawmill diameter class contribute,
    scaled by the species' A+B+C quality fraction, and a species missing from the
    quality table contributes nothing.

    Args:
        summaries (pandas.DataFrame): Needs ``is_for_sawmills_diameter``,
            ``baumart_for_quality`` and ``biomass_column``.
        baumart2fraction (dict): Species -> fraction suitable for sawmills (0-1).
        biomass_column (str): Column holding the volume to split.

    Returns:
        pandas.DataFrame: With the two ``_for_sawmills`` / ``_not_for_sawmills`` columns.
    """
    fraction = summaries["baumart_for_quality"].map(baumart2fraction).fillna(0.0).to_numpy(dtype=float)
    volume = summaries[biomass_column].to_numpy(dtype=float)
    for_sawmills = np.where(summaries["is_for_sawmills_diameter"].to_numpy(dtype=bool), volume * fraction, 0.0)
    summaries[f"{biomass_column}_for_sawmills"] = for_sawmills
    summaries[f"{biomass_column}_not_for_sawmills"] = volume - for_sawmills
    return summaries


def split_by_soft_hard(summaries, soft_species, hard_species):
    """
    Splits the summaries DataFrame into soft and hardwood categories.

    Args:
        summaries (pandas.DataFrame): The DataFrame containing a 'Baumart' column.
        soft_species (list): A list of softwood species names.
        hard_species (list): A list of hardwood species names.

    Returns:
        pandas.DataFrame: The input DataFrame with two new boolean columns: 'is_soft' and 'is_hard'.
    """
    if summaries.empty:
        print("Warning: Input DataFrame is empty, cannot split by soft/hard wood.")
        return summaries

    summaries["is_soft"] = summaries["Baumart"].isin(soft_species)
    summaries["is_hard"] = summaries["Baumart"].isin(hard_species)
    return summaries

def rolling_stats(df, time_window=5): # Added default time_window
    """
    Calculates the rolling mean and standard deviation of a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame with a time-based index.
        time_window (int, optional): The size of the rolling window. Defaults to 5.

    Returns:
        tuple: A tuple containing two DataFrames: the rolling mean and the rolling standard deviation.
               Returns (None, None) if the input DataFrame is empty.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty, cannot calculate rolling statistics.")
        return None, None
    return df.rolling(window=time_window, center=True, min_periods=1).mean(), \
           df.rolling(window=time_window, center=True, min_periods=1).std()

def plot_biomass(df_soft_1, df_hard_1, df_soft_7, df_hard_7, show=False, save=True):
    """
    Generates and displays plots of total biomass for softwood and hardwood under two RCP scenarios.

    Args:
        df_soft_1 (pandas.DataFrame): Softwood data for RCP 8.5.
        df_hard_1 (pandas.DataFrame): Hardwood data for RCP 8.5.
        df_soft_7 (pandas.DataFrame): Softwood data for RCP 4.5.
        df_hard_7 (pandas.DataFrame): Hardwood data for RCP 4.5.
        show (bool, optional): Whether to show the plot. Defaults to False.
        save (bool, optional): Whether to save the plot. Defaults to True.
    """
    y_mins = [df_soft_1["Volumen OR [m3]"].min(), df_hard_1["Volumen OR [m3]"].min(),
                df_soft_7["Volumen OR [m3]"].min(), df_hard_7["Volumen OR [m3]"].min()]
    y_maxs = [df_soft_1["Volumen OR [m3]"].max(), df_hard_1["Volumen OR [m3]"].max(),
                df_soft_7["Volumen OR [m3]"].max(), df_hard_7["Volumen OR [m3]"].max()]
    y_min = np.nanmin(y_mins) * 0.9
    y_max = np.nanmax(y_maxs) * 1.1
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.ylabel('Wood (m3)')
    plt.plot(df_soft_1.groupby(['year'])["Volumen OR [m3]"].sum().index,
             df_soft_1.groupby(['year'])["Volumen OR [m3]"].sum().values, label="softwood")
    plt.plot(df_hard_1.groupby(['year'])["Volumen OR [m3]"].sum().index,
             df_hard_1.groupby(['year'])["Volumen OR [m3]"].sum().values, label="hardwood")
    plt.xlabel('Year')
    plt.yscale('log')
    plt.ylim(y_min, y_max)

    plt.title('RCP 8.5 - Total Biomass')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df_soft_7.groupby(['year'])["Volumen OR [m3]"].sum().index,
             df_soft_7.groupby(['year'])["Volumen OR [m3]"].sum().values, label="softwood")
    plt.plot(df_hard_7.groupby(['year'])["Volumen OR [m3]"].sum().index,
             df_hard_7.groupby(['year'])["Volumen OR [m3]"].sum().values, label="hardwood")
    plt.yscale('log')
    plt.ylim(y_min, y_max)
    plt.xlabel('Year')
    plt.title('RCP 4.5 - Total Biomass')
    plt.legend()

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig("../figures/biomass_plot_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')

def plot_normalized_biomass_for_sawmill_categories(df_biomass_soft_1, df_biomass_hard_1,
                                                   df_biomass_soft_7, df_biomass_hard_7,
                                                   total_area, show=False, save=True):
    """
    Plots the normalized (by total area) biomass suitable and not suitable for sawmills
    for softwood and hardwood under two RCP scenarios, showing rolling mean and standard deviation.

    Args:
        df_biomass_soft_1 (pandas.DataFrame): Softwood biomass data for RCP 8.5.
        df_biomass_hard_1 (pandas.DataFrame): Hardwood biomass data for RCP 8.5.
        df_biomass_soft_7 (pandas.DataFrame): Softwood biomass data for RCP 4.5.
        df_biomass_hard_7 (pandas.DataFrame): Hardwood biomass data for RCP 4.5.
        total_area (float): The total area for normalization.
        show (bool, optional): Whether to show the plot. Defaults to False.
        save (bool, optional): Whether to save the plot. Defaults to True.
    """
    # Normalizing the volume by total area in ha
    norm_factor = total_area 
    df_biomass_soft_1_norm = df_biomass_soft_1 / norm_factor
    df_biomass_hard_1_norm = df_biomass_hard_1 / norm_factor
    df_biomass_soft_7_norm = df_biomass_soft_7 / norm_factor
    df_biomass_hard_7_norm = df_biomass_hard_7 / norm_factor

    # Apply rolling mean and std
    df_biomass_soft_1_mean, df_biomass_soft_1_std = rolling_stats(df_biomass_soft_1_norm)
    df_biomass_hard_1_mean, df_biomass_hard_1_std = rolling_stats(df_biomass_hard_1_norm)
    df_biomass_soft_7_mean, df_biomass_soft_7_std = rolling_stats(df_biomass_soft_7_norm)
    df_biomass_hard_7_mean, df_biomass_hard_7_std = rolling_stats(df_biomass_hard_7_norm)

    x1 = df_biomass_soft_1.index
    x2 = df_biomass_soft_7.index
    x3 = df_biomass_hard_1.index
    x4 = df_biomass_hard_7.index

    colors = sns.color_palette("tab10", n_colors=2)
    ls_styles = '-'

    plt.figure(figsize=(8, 6))

    # Softwood - For Sawmills
    plt.subplot(2, 2, 1)
    # plt.xlim(2025, 2305)
    plt.xticks()
    if df_biomass_soft_1_mean is not None and df_biomass_soft_1_std is not None:
        plt.plot(x1, df_biomass_soft_1_mean.iloc[:, 0], label="RCP 8.5", lw=2, ls=ls_styles[0], color=colors[0])
        plt.fill_between(x1, (df_biomass_soft_1_mean.iloc[:, 0] - df_biomass_soft_1_std.iloc[:, 0]),
                         (df_biomass_soft_1_mean.iloc[:, 0] + df_biomass_soft_1_std.iloc[:, 0]), color=colors[0], alpha=0.3)
    if df_biomass_soft_7_mean is not None and df_biomass_soft_7_std is not None:
        plt.plot(x2, df_biomass_soft_7_mean.iloc[:, 0], label="RCP 4.5", lw=2, ls=ls_styles, color=colors[1])
        plt.fill_between(x2, (df_biomass_soft_7_mean.iloc[:, 0] - df_biomass_soft_7_std.iloc[:, 0]),
                         (df_biomass_soft_7_mean.iloc[:, 0] + df_biomass_soft_7_std.iloc[:, 0]), color=colors[1], alpha=0.3)
    plt.ylabel('Wood (m3 / ha)')
    plt.title("Softwood", fontsize=18)

    # Hardwood - For Sawmills
    ax = plt.subplot(2, 2, 2)
    # plt.xlim(2025, 2305)
    plt.xticks()
    if df_biomass_hard_1_mean is not None and df_biomass_hard_1_std is not None:
        plt.plot(x3, df_biomass_hard_1_mean.iloc[:, 0], label="RCP 8.5", lw=2, ls=ls_styles, color=colors[0])
        plt.fill_between(x3, (df_biomass_hard_1_mean.iloc[:, 0] - df_biomass_hard_1_std.iloc[:, 0]),
                         (df_biomass_hard_1_mean.iloc[:, 0] + df_biomass_hard_1_std.iloc[:, 0]), color=colors[0], alpha=0.3)
    if df_biomass_hard_7_mean is not None and df_biomass_hard_7_std is not None:
        plt.plot(x4, df_biomass_hard_7_mean.iloc[:, 0], label="RCP 4.5", lw=2, ls=ls_styles, color=colors[1])
        plt.fill_between(x4, (df_biomass_hard_7_mean.iloc[:, 0] - df_biomass_hard_7_std.iloc[:, 0]),
                         (df_biomass_hard_7_mean.iloc[:, 0] + df_biomass_hard_7_std.iloc[:, 0]), color=colors[1], alpha=0.3)
    plt.ylabel('High quality\n(A,B/C)', labelpad=30, fontsize=18)
    ax.yaxis.set_label_position("right")
    plt.title("Hardwood", fontsize=18)

    # Softwood - Not For Sawmills
    plt.subplot(2, 2, 3)
    # plt.xlim(2025, 2305)
    plt.xticks()
    if df_biomass_soft_1_mean is not None and df_biomass_soft_1_std is not None:
        plt.plot(x1, df_biomass_soft_1_mean.iloc[:, 1], label="RCP 8.5", lw=2, ls=ls_styles, color=colors[0])
        plt.fill_between(x1, (df_biomass_soft_1_mean.iloc[:, 1] - df_biomass_soft_1_std.iloc[:, 1]),
                         (df_biomass_soft_1_mean.iloc[:, 1] + df_biomass_soft_1_std.iloc[:, 1]), color=colors[0], alpha=0.3)
    if df_biomass_soft_7_mean is not None and df_biomass_soft_7_std is not None:
        plt.plot(x2, df_biomass_soft_7_mean.iloc[:, 1], label="RCP 4.5", lw=2, ls=ls_styles, color=colors[1])
        plt.fill_between(x2, (df_biomass_soft_7_mean.iloc[:, 1] - df_biomass_soft_7_std.iloc[:, 1]),
                         (df_biomass_soft_7_mean.iloc[:, 1] + df_biomass_soft_7_std.iloc[:, 1]), color=colors[1], alpha=0.3)
    plt.ylabel('Wood (m3 / ha)')
    plt.xlabel('Year')

    # Hardwood - Not For Sawmills
    ax = plt.subplot(2, 2, 4)
    # plt.xlim(2025, 2305)
    plt.xticks()
    if df_biomass_hard_1_mean is not None and df_biomass_hard_1_std is not None:
        plt.plot(x3, df_biomass_hard_1_mean.iloc[:, 1], label="RCP 8.5", lw=2, ls=ls_styles, color=colors[0])
        plt.fill_between(x3, (df_biomass_hard_1_mean.iloc[:, 1] - df_biomass_hard_1_std.iloc[:, 1]),
                         (df_biomass_hard_1_mean.iloc[:, 1] + df_biomass_hard_1_std.iloc[:, 1]), color=colors[0], alpha=0.3)
    if df_biomass_hard_7_mean is not None and df_biomass_hard_7_std is not None:
        plt.plot(x4, df_biomass_hard_7_mean.iloc[:, 1], label="RCP 4.5", lw=2, ls=ls_styles, color=colors[1])
        plt.fill_between(x4, (df_biomass_hard_7_mean.iloc[:, 1] - df_biomass_hard_7_std.iloc[:, 1]),
                         (df_biomass_hard_7_mean.iloc[:, 1] + df_biomass_hard_7_std.iloc[:, 1]), color=colors[1], alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Low quality', labelpad=30, fontsize=18)
    ax.yaxis.set_label_position("right")

    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.2))
    if show:
        plt.show()
    if save:
        plt.savefig("../figures/biomass_normalized_for_sawmills_category_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')

import pandas as pd
import matplotlib.pyplot as plt

def plot_normalized_biomass_for_sawmill_categories_and_altitues(s, show=False, save=True, fname="all", normalize_by_area_and_year=False):
    """
    Create a 2x2 grid of line plots showing wood volume ('Volumen OR [m3]')
    by year, for softwood vs hardwood and above/below 1000 m elevation.

    Parameters
    ----------
    summaries : pd.DataFrame
        DataFrame containing at least:
        ['year', 'diameter_class', 'is_soft', 'is_hard', 'Above1000m', 'Volumen OR [m3]', 'area']
    show : bool, optional
        If True, display the plot interactively. Default is False.
    save : bool, optional
        If True, save the plot as 'wood_volume_2x2.png'. Default is True.
    """
    df = s.copy()
    if normalize_by_area_and_year:
        if 'area' not in df.columns:
            raise ValueError("Column 'area' is required when normalize_by_area_and_year=True.")
        df['Volumen OR [m3]'] = df['Volumen OR [m3]'] / df['area']
        year_max = df['year'].max()
        year_min = df['year'].min()
        df['Volumen OR [m3]'] /= year_max - year_min
        ylabel = "Volume density (m³/ha/year)"
        fname += "_density"
    else:
        ylabel = "Volume (m³)"
    # --- 1Group data ---
    df = (
        df.groupby(['year', 'Above1000m', 'is_soft', 'is_hard', 'diameter_class'])[['Volumen OR [m3]']]
        .sum()
        .reset_index()
    )

    # ---  Set up subplots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    # --- Define colors for diameter classes ---
    all_colors = { 'soft' :{
                            '<20cm': '#b2182b',
                            '20-40cm': '#ef8a62',
                            '>40cm': '#fddbc7'
                            },
                    'hard' :{
                            '<20cm': "#2166ac",
                            '20-40cm': "#67a9cf",
                            '>40cm': "#d1e5f0"
                    }
    }
    # --- Plot each combination ---
    for i, above in enumerate([False, True]):  # Row: below / above 1000 m
        for j, wood_type in enumerate(['soft', 'hard']):  # Column: soft / hard
            ax = axes[i, j]

            # Filter by elevation & wood type
            if wood_type == 'soft':
                subset = df[(df['is_soft']) & (~df['is_hard']) & (df['Above1000m'] == above)]
            else:
                subset = df[(~df['is_soft']) & (df['is_hard']) & (df['Above1000m'] == above)]

            # Plot each diameter class line
            colors = all_colors[wood_type]
            for diam, color in colors.items():
                data = subset[subset['diameter_class'] == diam]
                ax.plot(data['year'], data['Volumen OR [m3]'], label=diam, color=color, linewidth=2)

            # Titles and labels
            elev_label = "Above 1000 m" if above else "Below 1000 m"
            wood_label = "Softwood" if wood_type == 'soft' else "Hardwood"
            ax.set_title(f"{wood_label} – {elev_label}", fontsize=12)
            ax.set_xlabel("Year")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=":", alpha=0.6)

    # --- Add legend once ---
    # Create handles and labels separately for soft and hard
    handles_soft = [mlines.Line2D([0], [0], color=c, lw=2) for c in all_colors['soft'].values()]
    labels_soft = [f"Softwood - {d}" for d in all_colors['soft'].keys()]

    handles_hard = [mlines.Line2D([0], [0], color=c, lw=2) for c in all_colors['hard'].values()]
    labels_hard = [f"Hardwood - {d}" for d in all_colors['hard'].keys()]

    # Combine them in order: soft first row, hard second row
    handles = handles_soft + handles_hard
    labels = labels_soft + labels_hard
    order = [0,3,1,4,2,5]
    handles = [handles[i] for i in order]  # Ensure order
    labels = [labels[i] for i in order]  # 
    # Use ncol to make two rows
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # fig.suptitle("Wood Volume by Diameter Class, Elevation, and Wood Type", fontsize=14, y=1.02)

    # --- Save or show ---
    if save:
        fname = "../figures/wood_volume_with_altitude_"+str(fname)+".png"
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print("Saved figure as ", fname)
    if show:
        plt.show()

    plt.close(fig)

def plot_normalized_biomass_for_sawmill_categories_and_altitues_old(df_soft_1, df_hard_1, df_soft_7, df_hard_7, areas, show=False, save=True):
    """
    Plots the normalized (by total area) biomass suitable and not suitable for sawmills
    for softwood and hardwood under two RCP scenarios, showing rolling mean and standard deviation,
    categorized by altitude.

    Args:
        df_soft_1 (pandas.DataFrame): Softwood biomass data for RCP 8.5.
        df_hard_1 (pandas.DataFrame): Hardwood biomass data for RCP 8.5.
        df_soft_7 (pandas.DataFrame): Softwood biomass data for RCP 4.5.
        df_hard_7 (pandas.DataFrame): Hardwood biomass data for RCP 4.5.
        areas (list): A list containing the areas for normalization.
        show (bool, optional): Whether to show the plot. Defaults to False.
        save (bool, optional): Whether to save the plot. Defaults to True.
    """
    # Get all expected combinations
    all_groups = df_soft_1['year'].unique()
    above1000m_values = [1.0, 0.0]

    # Create full index
    full_index = pd.MultiIndex.from_product([all_groups, above1000m_values], names=['year', 'Above1000m'])

    df_biomass_soft_1 = df_soft_1.groupby(['year', "Above1000m"])[["Volumen OR [m3]_for_sawmills", "Volumen OR [m3]_not_for_sawmills"]].sum()
    df_biomass_soft_1 = df_biomass_soft_1.reindex(full_index, fill_value=0)
    df_biomass_soft_1 = df_biomass_soft_1.fillna(0)
    df_biomass_soft_1 = df_biomass_soft_1.unstack()

    df_biomass_hard_1 = df_hard_1.groupby(['year', "Above1000m" ])[["Volumen OR [m3]_for_sawmills", "Volumen OR [m3]_not_for_sawmills"]].sum()
    df_biomass_hard_1 = df_biomass_hard_1.reindex(full_index, fill_value=0)
    df_biomass_hard_1 = df_biomass_hard_1.fillna(0)
    df_biomass_hard_1 = df_biomass_hard_1.unstack()

    df_biomass_soft_7 = df_soft_7.groupby(['year', "Above1000m"])[["Volumen OR [m3]_for_sawmills", "Volumen OR [m3]_not_for_sawmills"]].sum()
    df_biomass_soft_7 = df_biomass_soft_7.reindex(full_index, fill_value=0)
    df_biomass_soft_7 = df_biomass_soft_7.fillna(0)
    df_biomass_soft_7 = df_biomass_soft_7.unstack()

    df_biomass_hard_7 = df_hard_7.groupby(['year', "Above1000m"])[["Volumen OR [m3]_for_sawmills", "Volumen OR [m3]_not_for_sawmills"]].sum()
    df_biomass_hard_7 = df_biomass_hard_7.reindex(full_index, fill_value=0)
    df_biomass_hard_7 = df_biomass_hard_7.fillna(0)
    df_biomass_hard_7= df_biomass_hard_7.unstack()


    # normalizing the volumen by total area in ha
    norm_factor = np.array([areas[0], areas[1], areas[0], areas[1]]) 
    try:
        df_biomass_soft_1 = df_biomass_soft_1 / norm_factor
        df_biomass_hard_1 = df_biomass_hard_1 / norm_factor
        df_biomass_soft_7 = df_biomass_soft_7 / norm_factor
        df_biomass_hard_7 = df_biomass_hard_7 / norm_factor
    except ValueError:
        print("Warning: Unable to normalize - m3/ha values could be corrupted in biomass_for_sawmill_categories_and_altitues")
        return None
    # Apply rolling mean and std
    df_biomass_soft_1_mean, df_biomass_soft_1_std = rolling_stats(df_biomass_soft_1)
    df_biomass_hard_1_mean, df_biomass_hard_1_std = rolling_stats(df_biomass_hard_1)
    df_biomass_soft_7_mean, df_biomass_soft_7_std = rolling_stats(df_biomass_soft_7)
    df_biomass_hard_7_mean, df_biomass_hard_7_std = rolling_stats(df_biomass_hard_7)

    #
    x1 = df_biomass_soft_1.index
    x2 = df_biomass_soft_7.index
    x3 = df_biomass_hard_1.index
    x4 = df_biomass_hard_7.index
    # we use blue and orange for the "RCP 8.5" and "RCP 4.5" for consistency
    colors = sns.color_palette("tab10", n_colors=2)
    # we use two line styles for below 1000 and above 1000
    ls_styles = [":","-"]
    #making the plot
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    # plt.xlim(2025,2305)
    # plt.xticks([2030, 2100, 2200, 2300])
    plt.plot(x1, df_biomass_soft_1_mean.iloc[:, 0], label="RCP 8.5 - BELOW ", lw=2, ls=ls_styles[0], color=colors[0])
    plt.fill_between(x1, (df_biomass_soft_1_mean.iloc[:, 0] - df_biomass_soft_1_std.iloc[:, 0]) ,
                    (df_biomass_soft_1_mean.iloc[:, 0] + df_biomass_soft_1_std.iloc[:, 0]) , color="lightgrey", alpha=0.3)
    plt.plot(x2, df_biomass_soft_7_mean.iloc[:, 0] , label="RCP 4.5 - BELOW ", lw=2, ls=ls_styles[0], color=colors[1])
    plt.fill_between(x2, (df_biomass_soft_7_mean.iloc[:, 0] - df_biomass_soft_7_std.iloc[:, 0]) ,
                    (df_biomass_soft_7_mean.iloc[:, 0] + df_biomass_soft_7_std.iloc[:, 0]) , color="lightgrey", alpha=0.3)
    plt.plot(x1, df_biomass_soft_1_mean.iloc[:, 1] , label="RCP 8.5 - ABOVE ", lw=2, ls=ls_styles[1], color=colors[0])
    plt.fill_between(x1, (df_biomass_soft_1_mean.iloc[:, 1] - df_biomass_soft_1_std.iloc[:, 1]) ,
                    (df_biomass_soft_1_mean.iloc[:, 1] + df_biomass_soft_1_std.iloc[:, 1]) , color="lightgrey", alpha=0.3)
    plt.plot(x2, df_biomass_soft_7_mean.iloc[:, 1] , label="RCP 4.5 - ABOVE ", lw=2, ls=ls_styles[1], color=colors[1])
    plt.fill_between(x2, (df_biomass_soft_7_mean.iloc[:, 1] - df_biomass_soft_7_std.iloc[:, 1]) ,
                    (df_biomass_soft_7_mean.iloc[:, 1] + df_biomass_soft_7_std.iloc[:, 1]) , color="lightgrey", alpha=0.3)
    plt.title("Softwood", fontsize=18)
    plt.ylabel('Wood (m3/ha/year)')
    plt.title("Softwood", fontsize = 18)
    ax = plt.subplot(2, 2, 2)
    # plt.xlim(2025,2305)
    # plt.xticks([2030, 2100, 2200, 2300])
    plt.plot(x3, df_biomass_hard_1_mean.iloc[:, 0], label="RCP 8.5 - BELOW ", lw=2, ls=ls_styles[0], color=colors[0])
    plt.fill_between(x3, (df_biomass_hard_1_mean.iloc[:, 0] - df_biomass_hard_1_std.iloc[:, 0]) ,
                    (df_biomass_hard_1_mean.iloc[:, 0] + df_biomass_hard_1_std.iloc[:, 0]) , color="lightgrey", alpha=0.3)
    plt.plot(x4, df_biomass_hard_7_mean.iloc[:, 0] , label="RCP 4.5 - BELOW ", lw=2, ls=ls_styles[0], color=colors[1])
    plt.fill_between(x4, (df_biomass_hard_7_mean.iloc[:, 0] - df_biomass_hard_7_std.iloc[:, 0]) ,
                    (df_biomass_hard_7_mean.iloc[:, 0] + df_biomass_hard_7_std.iloc[:, 0]) , color="lightgrey", alpha=0.3)
    plt.plot(x3, df_biomass_hard_1_mean.iloc[:, 1] , label="RCP 8.5 - ABOVE ", lw=2, ls=ls_styles[1], color=colors[0])
    plt.fill_between(x3, (df_biomass_hard_1_mean.iloc[:, 1] - df_biomass_hard_1_std.iloc[:, 1]) ,
                    (df_biomass_hard_1_mean.iloc[:, 1] + df_biomass_hard_1_std.iloc[:, 1]) , color="lightgrey", alpha=0.3)
    plt.plot(x4, df_biomass_hard_7_mean.iloc[:, 1] , label="RCP 4.5 - ABOVE ", lw=2, ls=ls_styles[1], color=colors[1])
    plt.fill_between(x4, (df_biomass_hard_7_mean.iloc[:, 1] - df_biomass_hard_7_std.iloc[:, 1]) ,
                    (df_biomass_hard_7_mean.iloc[:, 1] + df_biomass_hard_7_std.iloc[:, 1]) , color="lightgrey", alpha=0.3)
    plt.ylabel('High quality \n (A,B/C)', labelpad = 30, fontsize = 18)
    ax.yaxis.set_label_position("right")
    plt.title("Hardwood", fontsize = 18)
    plt.subplot(2, 2, 3)
    # plt.xlim(2025,2305)
    # plt.xticks([2030, 2100, 2200, 2300])
    plt.plot(x1, df_biomass_soft_1_mean.iloc[:, 2], label="RCP 8.5 - BELOW ", lw=2, ls=ls_styles[0], color=colors[0])
    plt.fill_between(x1, (df_biomass_soft_1_mean.iloc[:, 2] - df_biomass_soft_1_std.iloc[:, 2]) ,
                    (df_biomass_soft_1_mean.iloc[:, 2] + df_biomass_soft_1_std.iloc[:, 2]) , color="lightgrey", alpha=0.3)
    plt.plot(x2, df_biomass_soft_7_mean.iloc[:, 2] , label="RCP 4.5 - BELOW ", lw=2, ls=ls_styles[0], color=colors[1])
    plt.fill_between(x2, (df_biomass_soft_7_mean.iloc[:, 2] - df_biomass_soft_7_std.iloc[:, 2]) ,
                    (df_biomass_soft_7_mean.iloc[:, 2] + df_biomass_soft_7_std.iloc[:, 2]) , color="lightgrey", alpha=0.3)
    plt.plot(x1, df_biomass_soft_1_mean.iloc[:, 3] , label="RCP 8.5 - ABOVE ", lw=2, ls=ls_styles[1], color=colors[0])
    plt.fill_between(x1, (df_biomass_soft_1_mean.iloc[:, 3] - df_biomass_soft_1_std.iloc[:, 3]) ,
                    (df_biomass_soft_1_mean.iloc[:, 3] + df_biomass_soft_1_std.iloc[:, 3]) , color="lightgrey", alpha=0.3)
    plt.plot(x2, df_biomass_soft_7_mean.iloc[:, 3] , label="RCP 4.5 - ABOVE ", lw=2, ls=ls_styles[1], color=colors[1])
    plt.fill_between(x2, (df_biomass_soft_7_mean.iloc[:, 3] - df_biomass_soft_7_std.iloc[:, 3]) ,
                    (df_biomass_soft_7_mean.iloc[:, 3] + df_biomass_soft_7_std.iloc[:, 3]) , color="lightgrey", alpha=0.3)
    plt.ylabel('Wood (m3/ha/year)')
    plt.xlabel('Year')
    ax =plt.subplot(2, 2, 4)
    # plt.xlim(2025,2305)
    # plt.xticks([2030, 2100, 2200, 2300])
    plt.plot(x3, df_biomass_hard_1_mean.iloc[:, 2], label="RCP 8.5 - BELOW ", lw=2, ls=ls_styles[0], color=colors[0])
    plt.fill_between(x3, (df_biomass_hard_1_mean.iloc[:, 2] - df_biomass_hard_1_std.iloc[:, 2]) ,
                    (df_biomass_hard_1_mean.iloc[:, 2] + df_biomass_hard_1_std.iloc[:, 2]) , color="lightgrey", alpha=0.3)
    plt.plot(x4, df_biomass_hard_7_mean.iloc[:, 2] , label="RCP 4.5 - BELOW ", lw=2, ls=ls_styles[0], color=colors[1])
    plt.fill_between(x4, (df_biomass_hard_7_mean.iloc[:, 2] - df_biomass_hard_7_std.iloc[:, 2]) ,
                    (df_biomass_hard_7_mean.iloc[:, 2] + df_biomass_hard_7_std.iloc[:, 2]) , color="lightgrey", alpha=0.3)
    plt.plot(x3, df_biomass_hard_1_mean.iloc[:, 3] , label="RCP 8.5 - ABOVE ", lw=2, ls=ls_styles[1], color=colors[0])
    plt.fill_between(x3, (df_biomass_hard_1_mean.iloc[:, 3] - df_biomass_hard_1_std.iloc[:, 3]) ,
                    (df_biomass_hard_1_mean.iloc[:, 3] + df_biomass_hard_1_std.iloc[:, 3]) , color="lightgrey", alpha=0.3)
    plt.plot(x4, df_biomass_hard_7_mean.iloc[:, 3] , label="RCP 4.5 - ABOVE ", lw=2, ls=ls_styles[1], color=colors[1])
    plt.fill_between(x4, (df_biomass_hard_7_mean.iloc[:, 3] - df_biomass_hard_7_std.iloc[:, 3]) ,
                    (df_biomass_hard_7_mean.iloc[:, 3] + df_biomass_hard_7_std.iloc[:, 3]) , color="lightgrey", alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Low quality', labelpad = 30, fontsize = 18)
    ax.yaxis.set_label_position("right")
    plt.tight_layout()
    # make manual legend for the two line styles
    legend_elements = [Line2D([0], [0], color='black', lw=2, ls=ls_styles[0], label='BELOW'),
                    Line2D([0], [0], color='black', lw=2, ls=ls_styles[1], label='ABOVE'),
                    Line2D([0], [0], color=colors[0], lw=6, ls=ls_styles[1], label='RCP 8.5'),
                    Line2D([0], [0], color=colors[1], lw=6, ls=ls_styles[1], label='RCP 4.5')
                    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05,1.3))
    #
    if show:
        plt.show()
    if save:
        plt.savefig("../figures/biomass_for_sawmills_by_category_and_altitude_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')    

def plot_percentages_of_wood_quality(df_soft, df_hard, save=True, show=False, fname = "all", percent=True):
    """
    Plots a stacked bar chart of wood quality as a percentage of total wood.

    Parameters:
    df_soft (pd.DataFrame): DataFrame with years as index and wood volume and quality as columns.
    show (bool): Whether to display the plot.
    save (bool): Whether to save the plot as a PNG file.
    fname (str): Filename prefix for saving the plot.
    percent (bool): If True, plot percentages; if False, plot absolute values.
    """

    df_biomass_soft = df_soft.groupby(['year'])[[
        'Volumen OR [m3]_for_sawmills', 'Volumen OR [m3]_not_for_sawmills']].sum().astype(float).fillna(0)
    df_biomass_hard = df_hard.groupby(['year'])[[
        'Volumen OR [m3]_for_sawmills', 'Volumen OR [m3]_not_for_sawmills']].sum().astype(float).fillna(0)
    # join the two date frames by the 'year'
    # add _soft and _hard to the corresponging columns
    df = df_biomass_soft.join(df_biomass_hard, lsuffix='_soft', rsuffix='_hard', how='outer').fillna(0)
    # rename the columns to more meaningful names
    df.columns = ['High quality - Softwood', 'Low quality - Softwood', 'High quality - Hardwood', 'Low quality - Hardwood']
    # sum the rows by gruop of 10 (i.e., aggregate the rows by 10 years)
    df = df.groupby(df.index // 10 * 10).sum()
    # change the name of the index with min and max of the grouped index
    df.index = [f"{i}-{i+9}" for i in df.index]
    # Calculate percentage of total population
    if percent ==True:
        df = df.div(df.sum(axis=1), axis=0) * 100
        fname = "percent_"+fname
    # Colors for each category
    colors = {
        "High quality - Softwood": "#ca0020",
        "Low quality - Softwood" : "#f4a582",
        "High quality - Hardwood": "#0571b0",
        "Low quality - Hardwood" : "#92c5de"
        }
    # colors = {
    #         "High quality - Softwood": "#fdae61",   # orange
    #     "Low quality - Softwood": "#d7191c",     # red-orange
    #     "High quality - Hardwood": "#abdda4",  # green
    #     "Low quality - Hardwood": "#2b83ba",      # blue
    #     "": "#1a9850"  # dark green#1a9641",  # dark green
    # }

    # Plot
    ax = df.plot(kind="bar", stacked=True, 
                         figsize=(12, 6), 
                         color=[colors[col] for col in df.columns])

    # Labels and title
    plt.title("Breakdown of assortments", fontsize=14)
    plt.ylabel("% of assortment")
    plt.xlabel("Year")
    plt.xticks(rotation=45)

    # Legend
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save:
        plt.savefig("../figures/wood_quality_"+fname+"_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')    
    if show:
        plt.show()

def plot_percentages_of_wood(s, save=True, show=False, fname = "all", percent=True, plantation_separate=True):
    """
    Plots a stacked bar chart of wood quality as a percentage of total wood.

    Parameters:
    df (pd.DataFrame): DataFrame with years as index and wood volume and quality as columns.
    show (bool): Whether to display the plot.
    save (bool): Whether to save the plot as a PNG file.
    fname (str): Filename prefix for saving the plot.
    percent (bool): If True, plot percentages; if False, plot absolute values.
    """
    df = s.copy()
    if plantation_separate == True:
        # total plantation volume per year
        plantations = (
            df.query("plantation == True")   # adjust if 'yes'/'no'
            .groupby('year')[['Volumen OR [m3]']].sum()
            .rename(columns={'Volumen OR [m3]': 'Plantation'})
        )
        df = df.query("plantation == False")
        fname = "plantation_separate_"+fname

    df_biomass_soft = df.query("is_soft == True").groupby(['year'])['Volumen OR [m3]'].sum().astype(float).fillna(0).to_frame()
    df_biomass_hard = df.query("is_hard == True").groupby(['year'])['Volumen OR [m3]'].sum().astype(float).fillna(0).to_frame()
    # join the two date frames by the 'year'
    # add _soft and _hard to the corresponging columns
    df = df_biomass_soft.join(df_biomass_hard, lsuffix='_soft', rsuffix='_hard', how='outer').fillna(0)
    # rename the columns to more meaningful names
    df.columns = ['Softwood', 'Hardwood']
    # setting order
    col_order = ['Softwood', 'Hardwood']
    if plantation_separate == True:
         df = df.join(plantations, how='left').astype(float).fillna(0)
         col_order = ['Plantation'] + col_order
    df = df[col_order]
    # sum the rows by gruop of 10 (i.e., aggregate the rows by 10 years)
    df = df.groupby(df.index // 10 * 10).sum()
    # change the name of the index with min and max of the grouped index
    df.index = [f"{i}-{i+9}" for i in df.index]
    # Calculate percentage of total population
    if percent ==True:
        df = df.div(df.sum(axis=1), axis=0) * 100
        fname = "percent_"+fname
    # Colors for each category
    colors = {
        "Softwood": "#ca0020",
        "Hardwood": "#0571b0",
        'Plantation' :  "#999999"
        }

    # Plot
    ax = df.plot(kind="bar", stacked=True, 
                         figsize=(12, 6), 
                         color=[colors[col] for col in df.columns])

    # Labels and title
    plt.title("Breakdown of assortments", fontsize=14)
    if percent:
        plt.ylabel("% of assortment")
    else:
        plt.ylabel("Volume [m3]")
    plt.xlabel("Year")
    plt.xticks(rotation=45)

    # Legend
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save:
        plt.savefig("../figures/wood_"+fname+"_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')    
    if show:
        plt.show()

def plot_biomass_by_diameter_class(s, show=False, save=True, fname = "all", percent=True, plantation_separate = True):
    """
    We divide the biomass into three diameter classes: <20cm, 20-40cm, >40cm; and bewteen softwood and hardwood
    """
    df = s.copy()
    if plantation_separate == True:
        # total plantation volume per year
        plantations = (
            df.query("plantation == True")   # adjust if 'yes'/'no'
            .groupby('year')[['Volumen OR [m3]']].sum()
            .rename(columns={'Volumen OR [m3]': 'Plantation'})
    )
        df = df.query("plantation == False")
        fname = "plantation_separate_"+fname

    df = df.groupby(['year', 'diameter_class', 'is_soft', 'is_hard'])[["Volumen OR [m3]"]].sum().astype(float).fillna(0)
    df = df.unstack(level=[1,2,3]).fillna(0)
    df.columns = [ '20-40cm - Hardwood', '20-40cm - Softwood', 
                  '<20cm - Hardwood', '<20cm - Softwood',
                    '>40cm - Hardwood',  '>40cm - Softwood']
    # changing columns orders
    col_order = ['<20cm - Softwood', '20-40cm - Softwood', '>40cm - Softwood', '<20cm - Hardwood', '20-40cm - Hardwood', '>40cm - Hardwood']
    if plantation_separate == True:
        df = df.join(plantations, how='left').astype(float).fillna(0)
        col_order =['Plantation'] + col_order
    df = df[col_order]
    # sum the rows by gruop of 10 (i.e., aggregate the rows by 10 years)
    df = df.groupby(df.index // 10 * 10).sum()
    # change the name of the index with min and max of the grouped index
    df.index = [f"{i}-{i+9}" for i in df.index]
    if percent ==True:
        df = df.div(df.sum(axis=1), axis=0) * 100
        fname = "percent_"+fname
    # Colors for each category
    colors = {'<20cm - Softwood' : "#b2182b",
            '20-40cm - Softwood': "#ef8a62",
            '>40cm - Softwood': "#fddbc7",
            '<20cm - Hardwood': "#2166ac",
            '20-40cm - Hardwood':"#67a9cf",
            '>40cm - Hardwood':  "#d1e5f0",
            'Plantation' :  "#999999"
            }
    # Plot
    ax = df.plot(kind="bar", stacked=True, 
                        figsize=(12, 6), 
                        color=[colors[col] for col in df.columns])

    # Labels and title
    plt.title("Breakdown of assortments", fontsize=14)
    if percent == True:
        plt.ylabel("% of assortment")
    else:
        plt.ylabel("m3 of assortment")
    plt.xlabel("Year")
    plt.xticks(rotation=45)

    # Legend
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save:
        plt.savefig("../figures/wood_quality_"+fname+"_"+str(case_study)+"_"+str(management)+"_by_diameter.png", dpi=300, bbox_inches='tight')    
    if show:
        plt.show()  

def process_combination(args):
    """Builds the summary table and the figures for one (case study, scenario, cohort).

    ``args`` is ``(case_study, management, folder_path, start_time, sample,
    num_cores, cohort)``; ``cohort`` may be omitted and defaults to "dead" so
    that older call sites keep working.
    """
    global case_study, management, cohort
    summary_format = "parquet"
    if len(args) == 6:
        case_study, management, folder_path, start_time, sample, num_cores = args
        cohort = "dead"
    elif len(args) == 7:
        case_study, management, folder_path, start_time, sample, num_cores, cohort = args
    else:
        case_study, management, folder_path, start_time, sample, num_cores, cohort, summary_format = args
    # --- Configuration ---
    folder_path = f"{folder_path}/{case_study}/outputs/{management}/"
    # Resolved through paths so preflight checks the same file stage 2 reads.
    stand_data_path = paths.stand_details_path(case_study)
    quality_data_path = "../data/fraction_quality.xlsx"

    soft_species = ['Tanne', "Loerche", "Fichte", "Foehre", "Ubrige Nadelholz"]
    hard_species = ["Buche", "Eiche", "Esche", "Ahorn", "Ubrige Laubolz"]

    species_quality_mapping = {
        "Tanne": "Douglasie", "Fichte": "Fichte", "Foehre": "Kiefer", "Ubrige Nadelholz": "Fichte", "Loerche": "Fichte",
        "Ahorn": "Eiche", "Buche": "Buche", "Eiche": "Eiche", "Esche": "Eiche", "Ubrige Laubolz": "Birke"
    }

    # --- Verbose ---
    show = False
    # --- Figures --- 
    save = True
    # --- Data Loading and Preprocessing ---
    # print("Loading data...")
    # df = load_data(folder_path, management)
    print("Processing case study ", management, " in ", case_study, "-", cohort, "cohort")
    print("Loading data in parallel...")
    df = load_data_parallel(folder_path, management, sample, num_cores, cohort=cohort)

    print("Preprocessing main data...")
    summaries = preprocess_data(df, management)

    print("Compacting dtypes...")
    summaries = compact_dtypes(summaries)

    print("Loading stand data...")
    stand_data = pd.read_csv(stand_data_path)

    print("Augmenting with stand data...")
    summaries = augment_with_stand_data(summaries, stand_data)

    print("Adding sawmill diameter information...")
    summaries = add_sawmill_diameter_info(summaries)

    print("Loading quality data...")
    quality_df = load_quality_data(quality_data_path)

    print("Creating quality mapping...")
    baumart2fraction = create_quality_mapping(quality_df)

    print("Splitting by softwood and hardwood...")
    summaries = split_by_soft_hard(summaries, soft_species, hard_species)

    print("Mapping species for quality assessment...")
    summaries = map_species_for_quality(summaries, species_quality_mapping)

    # --- Calculating Biomass for Sawmill Use ---
    print("Calculating biomass fractions for sawmill use...")
    if not summaries.empty and baumart2fraction:
        summaries = calculate_sawmill_split(summaries, baumart2fraction)
        # drop column baumart_for_quality
        summaries.drop(columns=["baumart_for_quality","is_for_sawmills_diameter"], inplace=True)
    else:
        print("Warning: Summaries DataFrame is empty or quality mapping is not available. Skipping sawmill biomass calculation.")

    # dead-cohort summaries keep their historical name so earlier outputs stay
    # addressable; alive-cohort summaries get their own file next to them
    suffix = "" if cohort == "dead" else f"_{cohort}"
    written = write_summary(summaries,
                            f"{paths.summary_dir()}/{case_study}_{management}{suffix}",
                            fmt=summary_format)
    print("Wrote summary table to", written,
          f"({os.path.getsize(written)/1e6:,.0f} MB)")
    summaries = summaries[summaries["simtype"] == '1']
    # figures for the alive cohort are tagged so they do not overwrite the dead ones
    fig_tag = "8_5" if cohort == "dead" else f"8_5_{cohort}"
    # by diameter
    plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=True, plantation_separate=False, fname=fig_tag + '_all_years')
    plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=False, plantation_separate=False, fname=fig_tag + '_all_years')
    if management == "WOOD" or management == "HYBRID":
        plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=True, plantation_separate=True, fname=fig_tag + '_all_years')
        plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=False, plantation_separate=True, fname=fig_tag + '_all_years')
    # 
    print("Plotting percentages of wood quality as stacked bars...")
    if management == "WOOD" or management == "HYBRID":
        plot_percentages_of_wood(summaries, save = True, show= False,fname=fig_tag + "_all_years" )
    plot_percentages_of_wood(summaries, save = True, show= False,fname=fig_tag + "_all_years", plantation_separate=False)
    print("Plotting total of wood quality as stacked bars...")
    if management == "WOOD" or management == "HYBRID":
        plot_percentages_of_wood(summaries, save = True, show= False, fname=fig_tag + "_all_years", percent=False)
    plot_percentages_of_wood(summaries, save = True, show= False, fname=fig_tag + "_all_years", percent=False, plantation_separate=False)
    # drop the rows that have "year", (i.e.,"Gruppierungsmerkmal")larger than time_cut
    time_cut = 2160
    summaries = summaries[summaries["year"]< time_cut]
    if management == "WOOD" or management == "HYBRID":
        plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=True, plantation_separate=True, fname=fig_tag)
    plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=True, plantation_separate=False, fname=fig_tag)
    if management == "WOOD" or management == "HYBRID":
        plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=False, plantation_separate=True, fname=fig_tag)
    plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=False, plantation_separate=False, fname=fig_tag)
    if management == "WOOD" or management == "HYBRID":
        plot_percentages_of_wood(summaries, save = True, show= False,fname=fig_tag )
    plot_percentages_of_wood(summaries, save = True, show= False,fname=fig_tag, plantation_separate=False)
    print("Plotting total of wood quality as stacked bars...")
    if management == "WOOD" or management == "HYBRID":
        plot_percentages_of_wood(summaries, save = True, show= False, fname=fig_tag, percent=False)
    plot_percentages_of_wood(summaries, save = True, show= False, fname=fig_tag, percent=False, plantation_separate=False)

    # plot_normalized_biomass_for_sawmill_categories_and_altitues(summaries, save = True, show= False, fname=fig_tag)
    # plot_normalized_biomass_for_sawmill_categories_and_altitues(summaries, save = True, show= False, fname=fig_tag, normalize_by_area_and_year=True)

    print("All plots generated successfully.")
    plt.close("all")
    print("Time taken:", dt.datetime.now() - start_time)


if __name__ == "__main__":
    start_time = dt.datetime.now()
    # --- Read information from key arguments ---
    case_study_input = sys.argv[1] 
    management_input = sys.argv[2] 
    folder_path = sys.argv[3] 
    num_cores = int(sys.argv[4])
    sample_size = sys.argv[5]
    cohort_input = sys.argv[6] if len(sys.argv) > 6 else "dead"
    summary_format_input = sys.argv[7] if len(sys.argv) > 7 else "parquet"
    print("Processing data for management scenario ", management_input)
    print("Cohort ", cohort_input)
    print("Summary format ", summary_format_input)
    print("Case study ", case_study_input)
    print("Number of cores to be used ", num_cores)
    print("The sample size is ", sample_size)
    # check that the argument is valid
    regions.check_case_study(case_study_input)
    regions.check_scenario(management_input)
    if cohort_input not in ("dead", "alive"):
        raise ValueError("Invalid cohort. Please provide 'dead' or 'alive'.")
    if summary_format_input not in SUMMARY_FORMATS:
        raise ValueError(f"Invalid summary format. Please provide one of {list(SUMMARY_FORMATS)}.")
    if sample_size != 'False':
        try:
            sample_size = int(sample_size)
        except ValueError:
            raise ValueError("Invalid argument for sample_size. Please provide 'False' or an integer value.")
    else:
        sample_size = False

     # Select what to run
    case_studies_to_run = regions.resolve_case_studies(case_study_input)
    scenarios_to_run = regions.resolve_scenarios(management_input)
    combinations = [(cs, ms, folder_path, start_time, sample_size, num_cores, cohort_input, summary_format_input)
                    for cs in case_studies_to_run for ms in scenarios_to_run]

    for cb in combinations:
        results = process_combination(cb)
            