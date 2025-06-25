import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
import datetime as dt
from multiprocessing import Pool


from matplotlib.lines import Line2D

def parse_filename(filename: str, scenario: str) -> tuple:
    """
    Parse a filename to extract stand and simtype (and species if needed),
    based on the management scenario.

    Parameters:
    - filename: str, e.g. "sorsim_output240_7_planted_01.csv"
    - scenario: str, one of "WOOD", "BAU", "HYBRID", "BIO"

    Returns:
    - tuple: (stand, simtype) if scenario is BIO
             (stand, simtype_planted_species) otherwise
    """
    parts = filename.replace(".csv", "").split("_")

    # Expecting something like: ['sorsim', 'output240', '7', 'planted', '01']
    stand = parts[1].replace("output", "")
    simtype = parts[2]

    if scenario == "BIO":
        return stand, simtype
    else:
        species = parts[-1] if "planted" in parts else None
        print(species)
        print(parts)
        if species is None:
            raise ValueError("Expected 'planted' and species information for non-BIO scenario.")
        return stand, simtype

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
    # np.random.seed(42)
    # sample_size = max(1, int(0.1 * len(files)))  # ensure at least 1 file
    # sampled_files = list(np.random.choice(files, size=sample_size, replace=False))

    if "assortments_summaries.csv" in files:
        files.remove("assortments_summaries.csv")

    df = pd.DataFrame()
    for file in files:
        filename = os.path.join(folder_path, file)
        if os.path.isfile(filename):
            try:
                stand, simtype = parse_filename(file, management_scenario)
                # Read the CSV file
                data = pd.read_csv(filename, sep=";")

                # Identify the row with '#Gruppierungsmerkmal' and set it as the new header
                cut_point = data[data["#ID"] == "#Gruppierungsmerkmal"]
                if not cut_point.empty:
                    cut_point_index = cut_point.index[0]
                    data = data[cut_point_index:]
                    data.columns = data.iloc[0]
                    data = data[1:]
                    data["simtype"] = simtype
                    data["stand"] = stand
                    df = pd.concat([df, data], ignore_index=True)
                else:
                    print(f"Warning: '#Gruppierungsmerkmal' row not found in {file}. Skipping.")
            except ValueError:
                print(f"Warning: Skipping file {file} due to naming convention mismatch.")
            except Exception as e:
                print(f"Warning: Error reading file {file}: {e}")
    return df

def preprocess_data(df):
    """
    Preprocesses the combined DataFrame by removing unnecessary columns,
    handling missing values, renaming columns with umlauts, and casting volume columns to float.

    Args:
        df (pandas.DataFrame): The DataFrame to preprocess.

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

    # Keep specific columns without NaN as name
    keep_columns = [x for x in summaries.columns[:7]]
    keep_columns += [x for x in summaries.columns[-3:]]
    summaries = summaries[keep_columns]

    # Rename columns with umlauts
    summaries.rename({"L�ngenklasse": "Laengenklasse", 'St�rkenklasse': "Staerkenklasse"}, axis=1, inplace=True)

    # Rename rows in 'Baumart' with umlauts and specific replacements
    summaries["Baumart"] = summaries["Baumart"].apply(lambda x: x.replace("�", "oe"))
    summaries["Baumart"] = summaries["Baumart"].apply(lambda x: x.replace("oebriges", "Ubrige"))

    # Cast volume and value columns to float
    cols_to_float = ["Volumen IR [m3]", "Volumen OR [m3]", "Wert [CHF]"]
    for col in cols_to_float:
        if col in summaries.columns:
            summaries.loc[:, col] = pd.to_numeric(summaries[col], errors='coerce') # Handle potential conversion errors
        else:
            print(f"Warning: Column '{col}' not found, skipping float conversion.")

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

    stand_to_else = stand_data[["fsID", "Above1000m", "n.patches", "area"]].copy()
    stand_to_else.index = stand_to_else["fsID"].astype(str)
    stand_to_else = stand_to_else.drop(columns=["fsID"])
    stand_to_else_dict = stand_to_else.to_dict()

    # Adding area and altitude to summaries
    summaries["Above1000m"] = summaries["stand"].astype(str).map(stand_to_else_dict["Above1000m"])
    summaries["sim_area (m2)"] = summaries["stand"].astype(str).map(stand_to_else_dict["n.patches"]) * 625
    summaries["area"] = summaries["stand"].astype(str).map(stand_to_else_dict["area"])

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

    staerken2sawmills_use = {"1a": False, "1b": False, "2a": False, "2b": False, "3a": False, "3b": False,
                             "4": True, "5": True, "6": True, "7": True, "8": True,
                             "Restholz": False}
    summaries["is_for_sawmills_diameter"] = summaries["Staerkenklasse"].apply(
        lambda x: staerken2sawmills_use[x]) 
    # We coudl use .get() in the above line to handle unknown Staerkenklasse values

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
        quality_df["Baumart"] = quality_df["Baumart"].fillna(method='ffill')
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

    df["baumart_for_quality"] = df["Baumart"].apply(lambda x: species_quality_mapping.get(x, x)) # Default to original if not found
    return df

def calculate_biomass_for_sawmills(x, baumart2fraction, biomass_column="Volumen IR [m3]"):
    """
    Calculates the biomass fraction suitable for sawmills based on diameter class and wood quality.

    Args:
        x (pandas.Series): A row of the DataFrame.
        baumart2fraction (dict): A dictionary mapping tree species to the fraction for sawmills.
        biomass_column (str, optional): The name of the column containing biomass.
                                         Defaults to "Volumen IR [m3]".

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

def calculate_biomass_not_for_sawmills(x, baumart2fraction, biomass_column="Volumen IR [m3]"):
    """
    Calculates the biomass fraction not suitable for sawmills.

    Args:
        x (pandas.Series): A row of the DataFrame.
        baumart2fraction (dict): A dictionary mapping tree species to the fraction for sawmills.
        biomass_column (str, optional): The name of the column containing biomass.
                                         Defaults to "Volumen IR [m3]".

    Returns:
        float: The biomass (from the specified column) multiplied by (1 - fraction for sawmills)
               if the diameter class is relevant for sawmills, otherwise the original biomass.
    """
    if x["is_for_sawmills_diameter"]:
        fraction = baumart2fraction.get(x["baumart_for_quality"], 0) # Default to 0 if species not found
        return x[biomass_column] * (1 - fraction)
    else:
        return x[biomass_column]

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

    soft_hard_mapping = {v: "soft" for v in soft_species}
    for v in hard_species:
        soft_hard_mapping[v] = "hard"

    summaries["is_soft"] = summaries["Baumart"].apply(lambda x: soft_hard_mapping.get(x) == "soft")
    summaries["is_hard"] = summaries["Baumart"].apply(lambda x: soft_hard_mapping.get(x) == "hard")
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
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.ylabel('Wood (m3)')
    plt.plot(df_soft_1.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum().index.astype(int),
             df_soft_1.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum().values, label="softwood")
    plt.plot(df_hard_1.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum().index.astype(int),
             df_hard_1.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum().values, label="hardwood")
    plt.xlabel('Year')
    plt.yscale('log')
    plt.xlim(2010, 2310)
    plt.title('RCP 8.5 - Total Biomass')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df_soft_7.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum().index.astype(int),
             df_soft_7.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum().values, label="softwood")
    plt.plot(df_hard_7.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum().index.astype(int),
             df_hard_7.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum().values, label="hardwood")
    plt.yscale('log')
    plt.xlim(2010, 2310)
    plt.xlabel('Year')
    plt.title('RCP 4.5 - Total Biomass')
    plt.legend()

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig("../figures/biomass_plot_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')

def plot_biomass_with_rolling_stats(summaries, show=False, save=True):
    """
    Calculates and plots the average biomass of softwood and hardwood under different RCPs
    with rolling mean and standard deviation, normalized by total area.

    Args:
        summaries (pandas.DataFrame): The processed summaries DataFrame.
        show (bool, optional): Whether to show the plot. Defaults to False.
        save (bool, optional): Whether to save the plot. Defaults to True.
    """
    # Define time window for rolling statistics
    time_window = 10

    # Filter data for RCP 8.5 and RCP 4.5
    df_soft_1 = summaries[(summaries["simtype"] == "1") & (summaries["is_soft"] == True)]
    df_hard_1 = summaries[(summaries["simtype"] == "1") & (summaries["is_hard"] == True)]
    df_soft_7 = summaries[(summaries["simtype"] == "7") & (summaries["is_soft"] == True)]
    df_hard_7 = summaries[(summaries["simtype"] == "7") & (summaries["is_hard"] == True)]

    # Calculate total area
    areas = summaries.groupby(["stand", "Above1000m"])["sim_area (m2)"].mean().groupby("Above1000m").sum().to_dict()
    total_area = sum(areas.values())

    # Group by 'Gruppierungsmerkmal' and sum 'Volumen IR [m3]'
    df_biomass_soft_1 = df_soft_1.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum()
    df_biomass_hard_1 = df_hard_1.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum()
    df_biomass_soft_7 = df_soft_7.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum()
    df_biomass_hard_7 = df_hard_7.groupby(['Gruppierungsmerkmal'])["Volumen IR [m3]"].sum()

    # Apply rolling mean and std
    df_biomass_soft_1_mean, df_biomass_soft_1_std = rolling_stats(df_biomass_soft_1, time_window=time_window)
    df_biomass_hard_1_mean, df_biomass_hard_1_std = rolling_stats(df_biomass_hard_1, time_window=time_window)
    df_biomass_soft_7_mean, df_biomass_soft_7_std = rolling_stats(df_biomass_soft_7, time_window=time_window)
    df_biomass_hard_7_mean, df_biomass_hard_7_std = rolling_stats(df_biomass_hard_7, time_window=time_window)

    if all([df_biomass_soft_1_mean is None, df_biomass_hard_1_mean is None,
            df_biomass_soft_7_mean is None, df_biomass_hard_7_mean is None]):
        print("Warning: No biomass data available for plotting with rolling statistics.")
        return

    x = df_biomass_soft_1.index.astype(int)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("RCP 8.5")
    plt.ylabel('Wood (m3 / ha)')
    if df_biomass_soft_1_mean is not None and df_biomass_soft_1_std is not None:
        plt.plot(x, df_biomass_soft_1_mean * 10**4 / total_area, label="Softwood")
        plt.fill_between(x, (df_biomass_soft_1_mean - df_biomass_soft_1_std) * 10**4 / total_area,
                         (df_biomass_soft_1_mean + df_biomass_soft_1_std) * 10**4 / total_area, color="lightgrey", alpha=0.6)
    if df_biomass_hard_1_mean is not None and df_biomass_hard_1_std is not None:
        plt.plot(x, df_biomass_hard_1_mean * 10**4 / total_area, label="Hardwood")
        plt.fill_between(x, (df_biomass_hard_1_mean - df_biomass_hard_1_std) * 10**4 / total_area,
                         (df_biomass_hard_1_mean + df_biomass_hard_1_std) * 10**4 / total_area, color="lightgrey", alpha=0.6)
    plt.xlabel('Year')
    plt.xlim(2030, 2310)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("RCP 4.5")
    if df_biomass_soft_7_mean is not None and df_biomass_soft_7_std is not None:
        plt.plot(x, df_biomass_soft_7_mean * 10**4 / total_area, label="Softwood")
        plt.fill_between(x, (df_biomass_soft_7_mean - df_biomass_soft_7_std) * 10**4 / total_area,
                         (df_biomass_soft_7_mean + df_biomass_soft_7_std) * 10**4 / total_area, color="lightgrey", alpha=0.6)
    if df_biomass_hard_7_mean is not None and df_biomass_hard_7_std is not None:
        plt.plot(x, df_biomass_hard_7_mean * 10**4 / total_area, label="Hardwood")
        plt.fill_between(x, (df_biomass_hard_7_mean - df_biomass_hard_7_std) * 10**4 / total_area,
                         (df_biomass_hard_7_mean + df_biomass_hard_7_std) * 10**4 / total_area, color="lightgrey", alpha=0.6)
    plt.xlim(2030, 2310)
    plt.xlabel('Year')
    plt.legend()

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig("../figures/biomass_plot_with_rolling_time_window_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')

def plot_biomass_for_sawmill_categories(df_soft_1, df_hard_1, df_soft_7, df_hard_7, show=False, save=True):
    """
    Generates and displays scatter plots showing the biomass suitable and not suitable
    for sawmills for softwood and hardwood under two RCP scenarios.

    Args:
        df_soft_1 (pandas.DataFrame): Softwood data for RCP 8.5 with sawmill categories.
        df_hard_1 (pandas.DataFrame): Hardwood data for RCP 8.5 with sawmill categories.
        df_soft_7 (pandas.DataFrame): Softwood data for RCP 4.5 with sawmill categories.
        df_hard_7 (pandas.DataFrame): Hardwood data for RCP 4.5 with sawmill categories.
        show (bool, optional): Whether to show the plot. Defaults to False.
        save (bool, optional): Whether to save the plot. Defaults to True.
    """
    df_biomass_soft_1 = df_soft_1.groupby(['Gruppierungsmerkmal'])[[
        'Volumen IR [m3]_for_sawmills', 'Volumen IR [m3]_not_for_sawmills']].sum().fillna(0)
    df_biomass_hard_1 = df_hard_1.groupby(['Gruppierungsmerkmal'])[[
        'Volumen IR [m3]_for_sawmills', 'Volumen IR [m3]_not_for_sawmills']].sum().fillna(0)
    df_biomass_soft_7 = df_soft_7.groupby(['Gruppierungsmerkmal'])[[
        'Volumen IR [m3]_for_sawmills', 'Volumen IR [m3]_not_for_sawmills']].sum().fillna(0)
    df_biomass_hard_7 = df_hard_7.groupby(['Gruppierungsmerkmal'])[[
        'Volumen IR [m3]_for_sawmills', 'Volumen IR [m3]_not_for_sawmills']].sum().fillna(0)

    y_soft_1 = df_biomass_soft_1.values
    y_hard_1 = df_biomass_hard_1.values
    y_soft_7 = df_biomass_soft_7.values
    y_hard_7 = df_biomass_hard_7.values
    x1 = df_biomass_soft_1.index.astype(int)
    x2 = df_biomass_soft_7.index.astype(int)
    x3 = df_biomass_hard_1.index.astype(int)
    x4 = df_biomass_hard_7.index.astype(int)

    plt.figure(figsize=(6, 6))

    # Softwood - For Sawmills
    plt.subplot(2, 2, 1)
    ls_styles = ['-', '--']
    colors = sns.color_palette("tab10", n_colors=2)
    plt.scatter(x1, y_soft_1[:, 0] * 10**3, label="RCP 8.5", alpha=0.8, lw=2, ls=ls_styles, color=colors[0])
    plt.scatter(x2, y_soft_7[:, 0] * 10**3, label="RCP 4.5", alpha=0.8, lw=2, ls=ls_styles, color=colors[1])
    plt.yscale("log")
    plt.ylabel('Wood (m3)')
    plt.title("Soft-wood", fontsize=18)

    # Hardwood - For Sawmills
    ax = plt.subplot(2, 2, 2)
    plt.scatter(x3, y_hard_1[:, 0] * 10**3, label="RCP 8.5", alpha=0.8, lw=2, ls=ls_styles, color=colors[0])
    plt.scatter(x4, y_hard_7[:, 0] * 10**3, label="RCP 4.5", alpha=0.8, lw=2, ls=ls_styles, color=colors[1])
    plt.yscale("log")
    plt.ylabel('High quality \n (A,B/C)', labelpad=30, fontsize=18)
    ax.yaxis.set_label_position("right")
    plt.title("Hard-wood", fontsize=18)

    # Softwood - Not For Sawmills
    plt.subplot(2, 2, 3)
    plt.scatter(x1, y_soft_1[:, 1] * 10**3, label="RCP 8.5", alpha=0.8, lw=2, ls=ls_styles, color=colors[0])
    plt.scatter(x2, y_soft_7[:, 1] * 10**3, label="RCP 4.5", alpha=0.8, lw=2, ls=ls_styles, color=colors[1])
    plt.yscale("log")
    plt.ylabel('Wood (m3)')
    plt.xlabel('Year')

    # Hardwood - Not For Sawmills
    ax = plt.subplot(2, 2, 4)
    plt.scatter(x3, y_hard_1[:, 1] * 10**3, label="RCP 8.5", alpha=0.8, lw=2, ls=ls_styles, color=colors[0])
    plt.scatter(x4, y_hard_7[:, 1] * 10**3, label="RCP 4.5", alpha=0.8, lw=2, ls=ls_styles, color=colors[1])
    plt.yscale("log")
    plt.xlabel('Year')
    plt.ylabel('Low quality', labelpad=30, fontsize=18)
    ax.yaxis.set_label_position("right")

    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.2))
    if show:
        plt.show()
    if save:
        plt.savefig("../figures/biomass_sawmill_categories_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')

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
    norm_factor = total_area / 10**4
    df_biomass_soft_1_norm = df_biomass_soft_1 / norm_factor
    df_biomass_hard_1_norm = df_biomass_hard_1 / norm_factor
    df_biomass_soft_7_norm = df_biomass_soft_7 / norm_factor
    df_biomass_hard_7_norm = df_biomass_hard_7 / norm_factor

    # Apply rolling mean and std
    df_biomass_soft_1_mean, df_biomass_soft_1_std = rolling_stats(df_biomass_soft_1_norm)
    df_biomass_hard_1_mean, df_biomass_hard_1_std = rolling_stats(df_biomass_hard_1_norm)
    df_biomass_soft_7_mean, df_biomass_soft_7_std = rolling_stats(df_biomass_soft_7_norm)
    df_biomass_hard_7_mean, df_biomass_hard_7_std = rolling_stats(df_biomass_hard_7_norm)

    x1 = df_biomass_soft_1.index.astype(int)
    x2 = df_biomass_soft_7.index.astype(int)
    x3 = df_biomass_hard_1.index.astype(int)
    x4 = df_biomass_hard_7.index.astype(int)

    colors = sns.color_palette("tab10", n_colors=2)
    ls_styles = '-'

    plt.figure(figsize=(8, 6))

    # Softwood - For Sawmills
    plt.subplot(2, 2, 1)
    plt.xlim(2025, 2305)
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
    plt.xlim(2025, 2305)
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
    plt.xlim(2025, 2305)
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
    plt.xlim(2025, 2305)
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
        plt.savefig("../figures/biomass_normalized_for_sawmills_by_category_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')

def plot_normalized_biomass_for_sawmill_categories_and_altitues(df_soft_1, df_hard_1, df_soft_7, df_hard_7, areas, show=False, save=True):
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

    df_biomass_soft_1 = df_soft_1.groupby(['Gruppierungsmerkmal', "Above1000m"])[["Volumen IR [m3]_for_sawmills", "Volumen IR [m3]_not_for_sawmills"]].sum()
    df_biomass_soft_1 = df_biomass_soft_1.fillna(0)
    df_biomass_soft_1 = df_biomass_soft_1.unstack()
    y_soft_1 = df_biomass_soft_1.values
    df_biomass_hard_1 = df_hard_1.groupby(['Gruppierungsmerkmal', "Above1000m" ])[["Volumen IR [m3]_for_sawmills", "Volumen IR [m3]_not_for_sawmills"]].sum()
    df_biomass_hard_1 = df_biomass_hard_1.fillna(0)
    df_biomass_hard_1 = df_biomass_hard_1.unstack()
    y_hard_1 = df_biomass_hard_1.values
    df_biomass_soft_7 = df_soft_7.groupby(['Gruppierungsmerkmal', "Above1000m"])[["Volumen IR [m3]_for_sawmills", "Volumen IR [m3]_not_for_sawmills"]].sum()
    df_biomass_soft_7 = df_biomass_soft_7.fillna(0)
    df_biomass_soft_7 = df_biomass_soft_7.unstack()
    y_soft_7 = df_biomass_soft_7.values
    df_biomass_hard_7 = df_hard_7.groupby(['Gruppierungsmerkmal', "Above1000m"])[["Volumen IR [m3]_for_sawmills", "Volumen IR [m3]_not_for_sawmills"]].sum()
    df_biomass_hard_7 = df_biomass_hard_7.fillna(0)
    df_biomass_hard_7= df_biomass_hard_7.unstack()
    y_hard_7 =df_biomass_hard_7.values

    # normalizing the volumen by total area in ha
    norm_factor = np.array([areas[0], areas[1], areas[0], areas[1]]) / 10**4
    df_biomass_soft_1 = df_biomass_soft_1 / norm_factor
    df_biomass_hard_1 = df_biomass_hard_1 / norm_factor
    df_biomass_soft_7 = df_biomass_soft_7 / norm_factor
    df_biomass_hard_7 = df_biomass_hard_7 / norm_factor
    # Apply rolling mean and std
    df_biomass_soft_1_mean, df_biomass_soft_1_std = rolling_stats(df_biomass_soft_1)
    df_biomass_hard_1_mean, df_biomass_hard_1_std = rolling_stats(df_biomass_hard_1)
    df_biomass_soft_7_mean, df_biomass_soft_7_std = rolling_stats(df_biomass_soft_7)
    df_biomass_hard_7_mean, df_biomass_hard_7_std = rolling_stats(df_biomass_hard_7)

    #
    x1 = df_biomass_soft_1.index.astype(int)
    x2 = df_biomass_soft_7.index.astype(int)
    x3 = df_biomass_hard_1.index.astype(int)
    x4 = df_biomass_hard_7.index.astype(int)
    # we use blue and orange for the "RCP 8.5" and "RCP 4.5" for consistency
    colors = sns.color_palette("tab10", n_colors=2)
    # we use two line styles for below 1000 and above 1000
    ls_styles = [":","-"]
    #making the plot
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    plt.xlim(2025,2305)
    plt.xticks([2030, 2100, 2200, 2300])
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
    plt.ylabel('Wood (m3/ha)')
    plt.title("Softwood", fontsize = 18)
    ax = plt.subplot(2, 2, 2)
    plt.xlim(2025,2305)
    plt.xticks([2030, 2100, 2200, 2300])
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
    plt.xlim(2025,2305)
    plt.xticks([2030, 2100, 2200, 2300])
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
    plt.ylabel('Wood (m3/ha)')
    plt.xlabel('Year')
    ax =plt.subplot(2, 2, 4)
    plt.xlim(2025,2305)
    plt.xticks([2030, 2100, 2200, 2300])
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

def process_combination(args):
    global case_study, management
    case_study, management, start_time = args
    # --- Configuration ---
    folder_path = f"{folder_path}/{case_study}/outputs/{management}/"
    stand_data_path = f"../data/{case_study}/stand.details.csv"
    quality_data_path = "../data/fraction_quality.xlsx"

    soft_species = ['Tanne', "Loerche", "Fichte", "Foehre", "Ubrige Nadelholz"]
    hard_species = ["Buche", "Eiche", "Esche", "Ahorn", "Ubrige Laubolz"]

    species_quality_mapping = {
        "Tanne": "Douglasie", "Fichte": "Fichte", "Foehre": "Kiefer", "Ubrige Nadelholz": "Fichte", "Loerche": "Fichte",
        "Ahorn": "Eiche", "Buche": "Buche", "Eiche": "Eiche", "Esche": "Eiche", "Ubrige Laubolz": "Birke"
    }

    # --- Verbose ---
    show = False
    save = True

    # --- Data Loading and Preprocessing ---
    print("Loading data...")
    df = load_data(folder_path, management)

    print("Preprocessing main data...")
    summaries = preprocess_data(df)

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
        summaries["Volumen IR [m3]_for_sawmills"] = summaries.apply(
            lambda x: calculate_biomass_for_sawmills(x, baumart2fraction), axis=1
        )
        summaries["Volumen IR [m3]_not_for_sawmills"] = summaries.apply(
            lambda x: calculate_biomass_not_for_sawmills(x, baumart2fraction), axis=1
        )
    else:
        print("Warning: Summaries DataFrame is empty or quality mapping is not available. Skipping sawmill biomass calculation.")

    # --- Splitting Data for Plotting ---
    df_soft_1 = summaries[(summaries["simtype"] == "1") & (summaries["is_soft"] == True)].copy()
    df_hard_1 = summaries[(summaries["simtype"] == "1") & (summaries["is_hard"] == True)].copy()
    df_soft_7 = summaries[(summaries["simtype"] == "7") & (summaries["is_soft"] == True)].copy()
    df_hard_7 = summaries[(summaries["simtype"] == "7") & (summaries["is_hard"] == True)].copy()

    # --- Plotting ---
    print("Plotting total biomass...")
    plot_biomass(df_soft_1.copy(), df_hard_1.copy(), df_soft_7.copy(), df_hard_7.copy(), show=show, save=save)

    # Calculate total area for normalization in the next plot
    if not summaries.empty:
        areas = summaries.groupby(["stand", "Above1000m"])["sim_area (m2)"].mean().groupby("Above1000m").sum().to_dict()
        total_area = sum(areas.values())

        df_biomass_soft_1_grouped = df_soft_1.groupby(['Gruppierungsmerkmal'])[[
            'Volumen IR [m3]_for_sawmills', 'Volumen IR [m3]_not_for_sawmills']].sum().fillna(0)
        df_biomass_hard_1_grouped = df_hard_1.groupby(['Gruppierungsmerkmal'])[[
            'Volumen IR [m3]_for_sawmills', 'Volumen IR [m3]_not_for_sawmills']].sum().fillna(0)
        df_biomass_soft_7_grouped = df_soft_7.groupby(['Gruppierungsmerkmal'])[[
            'Volumen IR [m3]_for_sawmills', 'Volumen IR [m3]_not_for_sawmills']].sum().fillna(0)
        df_biomass_hard_7_grouped = df_hard_7.groupby(['Gruppierungsmerkmal'])[[
            'Volumen IR [m3]_for_sawmills', 'Volumen IR [m3]_not_for_sawmills']].sum().fillna(0)

        print("Plotting normalized biomass for sawmill categories with rolling stats...")
        plot_normalized_biomass_for_sawmill_categories(
            df_biomass_soft_1_grouped.copy(),
            df_biomass_hard_1_grouped.copy(),
            df_biomass_soft_7_grouped.copy(),
            df_biomass_hard_7_grouped.copy(),
            total_area,
            show=show, save=save
        )

    print("Plotting biomass for sawmill categories...")
    plot_biomass_for_sawmill_categories(df_soft_1.copy(), df_hard_1.copy(), df_soft_7.copy(), df_hard_7.copy(), show=show, save=save)

    print("Plotting normalized biomass for sawmill categories and altitudes...")
    plot_normalized_biomass_for_sawmill_categories_and_altitues(df_soft_1.copy(), df_hard_1.copy(), df_soft_7.copy(), df_hard_7.copy(), areas, show=show, save=save)
    print("All plots generated successfully.")
    print("Time taken:", dt.datetime.now() - start_time)


if __name__ == "__main__":
    start_time = dt.datetime.now()
    # --- Read information from key arguments ---
    case_study_input = sys.argv[1] 
    management_input = sys.argv[2] 
    folder_path = sys.argv[3] 
    num_cores = int(sys.argv[4])
    print("Processing data for management scenario ", management_input)
    print("Case study ", case_study_input)
    print("Number of cores to be used ", num_cores)
    # check that the argument is valid
    valid_management_scenarios = ["BAU", "WOOD", "HYBRID", "ALL"]
    valid_case_studies = ["Entlebuch", "Vaud", "Surselva", "All"]
    if case_study_input not in valid_case_studies:
        raise ValueError(f"Invalid case study. Please provide a valid case study {valid_case_studies}.")
    if management_input not in valid_management_scenarios:
        raise ValueError(f"Invalid management scenario. Please provide a valid management scenario {valid_management_scenarios}.")
    
     # Select what to run
    case_studies_to_run = [cs for cs in valid_case_studies if cs != "All"] if case_study_input == "All" else [case_study_input]
    scenarios_to_run = [ms for ms in valid_management_scenarios if ms != "ALL"] if management_input == "ALL" else [management_input]
    combinations = [(cs, ms, start_time) for cs in case_studies_to_run for ms in scenarios_to_run]

    with Pool(processes=num_cores) as pool:
        results = pool.map(process_combination, combinations)
            