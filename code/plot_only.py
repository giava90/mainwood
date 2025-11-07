import matplotlib.pyplot as plt
import pandas as pd
import sys
import datetime as dt

import pdb

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
    df = df.groupby(df.index // 10 * 10).sum()/10
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
    if percent==True:
        plt.ylabel("% of assortment")
    else:
        plt.ylabel(r"Volume ($m^3/y$)")
        plt.ylim(0,90_000)
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
    df = df.groupby(df.index // 10 * 10).sum()/10
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
        plt.ylim(0,90_000)
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
    df = df.groupby(df.index // 10 * 10).sum()/10
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
        plt.ylim(0,90_000)
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
    global case_study, management
    case_study, management, folder_path, start_time, sample, num_cores = args
    print("Processing case study ", management, " in ", case_study )
    # --- Configuration ---
    folder_path = f"{folder_path}/{case_study}/outputs/{management}/"
    
    # --- Verbose ---
    show = False
    # --- Figures --- 
    save = True
    # --- Data Loading  ---
    print("Loading summaries...")
    summaries = pd.read_csv(f"../data/summaries_for_plots/{case_study}_{management}.csv")
    print("Keeping only simtype ==1 that means RCP8.5")
    summaries = summaries[summaries["simtype"] == 1]
    # by diameter
    print("Loading summaries...")
    plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=True, plantation_separate=False, fname='8_5_all_years')
    plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=False, plantation_separate=False, fname='8_5_all_years')
    if management == "WOOD" or management == "HYBRID":
        plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=True, plantation_separate=True, fname='8_5_all_years')
        plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=False, plantation_separate=True, fname='8_5_all_years')
    # 
    print("Plotting percentages of wood quality as stacked bars...")
    if management == "WOOD" or management == "HYBRID":
        plot_percentages_of_wood(summaries, save = True, show= False,fname ="8_5_all_years" )
    plot_percentages_of_wood(summaries, save = True, show= False,fname ="8_5_all_years", plantation_separate=False)
    print("Plotting total of wood quality as stacked bars...")
    if management == "WOOD" or management == "HYBRID":
        plot_percentages_of_wood(summaries, save = True, show= False, fname ="8_5_all_years", percent=False)
    plot_percentages_of_wood(summaries, save = True, show= False, fname ="8_5_all_years", percent=False, plantation_separate=False)
    # drop the rows that have "year", (i.e.,"Gruppierungsmerkmal")larger than time_cut
    time_cut = 2160
    summaries = summaries[summaries["year"]< time_cut]
    if management == "WOOD" or management == "HYBRID":
        plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=True, plantation_separate=True, fname='8_5')
    plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=True, plantation_separate=False, fname='8_5')
    if management == "WOOD" or management == "HYBRID":
        plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=False, plantation_separate=True, fname='8_5')
    plot_biomass_by_diameter_class(summaries, show=show, save=save, percent=False, plantation_separate=False, fname='8_5')
    if management == "WOOD" or management == "HYBRID":
        plot_percentages_of_wood(summaries, save = True, show= False,fname ="8_5" )
    plot_percentages_of_wood(summaries, save = True, show= False,fname ="8_5", plantation_separate=False)
    print("Plotting total of wood quality as stacked bars...")
    if management == "WOOD" or management == "HYBRID":
        plot_percentages_of_wood(summaries, save = True, show= False, fname ="8_5", percent=False)
    plot_percentages_of_wood(summaries, save = True, show= False, fname ="8_5", percent=False, plantation_separate=False)

    print("All plots generated successfully.")
    print("Time taken:", dt.datetime.now() - start_time)


if __name__ == "__main__":
    start_time = dt.datetime.now()
    # --- Read information from key arguments ---
    case_study_input = sys.argv[1] 
    management_input = sys.argv[2] 
    folder_path = sys.argv[3] 
    num_cores = int(sys.argv[4])
    sample_size = sys.argv[5]
    print("Processing data for management scenario ", management_input)
    print("Case study ", case_study_input)
    print("Number of cores to be used ", num_cores)
    print("The sample size is ", sample_size)
    # check that the argument is valid
    valid_management_scenarios = ["BAU", "WOOD", "BIO", "ALL"] #, "HYBRID"]
    valid_case_studies = ["Entlebuch", "Vaud", "Surselva", "Misox", "All"]
    if case_study_input not in valid_case_studies:
        raise ValueError(f"Invalid case study. Please provide a valid case study {valid_case_studies}.")
    if management_input not in valid_management_scenarios:
        raise ValueError(f"Invalid management scenario. Please provide a valid management scenario {valid_management_scenarios}.")
    if sample_size != 'False':
        try:
            sample_size = int(sample_size)
        except ValueError:
            raise ValueError("Invalid argument for sample_size. Please provide 'False' or an integer value.")
    else:
        sample_size = False

     # Select what to run
    case_studies_to_run = [cs for cs in valid_case_studies if cs != "All"] if case_study_input == "All" else [case_study_input]
    scenarios_to_run = [ms for ms in valid_management_scenarios if ms != "ALL"] if management_input == "ALL" else [management_input]
    combinations = [(cs, ms, folder_path, start_time, sample_size, num_cores) for cs in case_studies_to_run for ms in scenarios_to_run]
    print("Sample and num_cores info are ingnored")

    for cb in combinations:
        results = process_combination(cb)
    plt.close("all")
            