"""The figure functions for the SZF paper, and nothing else.

Lifted verbatim from ``SZF/wisse/code/tools.py`` -- the same code that drew the
published figures, so the output is byte-identical. Only the four functions
``make_paper_figures.py`` actually calls are here; the EWP-potential helpers and
their lookup tables stayed behind, because carrying them would mean maintaining a
second copy of code nothing in this repository runs.

Do not "improve" these bodies. Their value is that they match what was published.
The one edit made to them is an explicit ``observed=False`` on the Baumart
groupby: that is the current pandas default, so the output is unchanged, and
pinning it stops a future pandas release from altering a published figure.
If a figure needs to change, change it deliberately and say so in the commit --
the paper figures and the pipeline's own figures (``summarize_and_create_plots``)
are already drawn by different code, and this module exists so that difference is
explicit rather than accidental.

The savefig paths inside are relative and inconsistent between functions -- some
write ``figures/``, some ``../figures/``. ``make_paper_figures.py`` intercepts
every save and redirects it, so nothing here depends on the working directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def prepare_data_for_sank_plot(df_2, year_before = 2030, year_after = 2050):
    df = (
        # observed=False is today's default, passed explicitly. Baumart is a
        # Categorical (compact_dtypes makes it one and Parquet preserves it), and
        # pandas is changing this default to True. Under True, a species with no
        # volume left after filtering would vanish from the result instead of
        # appearing as a zero -- silently changing a published figure on a pandas
        # upgrade. Pinning it is not a change in behaviour: it is what happens now.
        df_2.groupby("Baumart", observed=False)
        .apply(lambda g: pd.Series({
            "volume_before": g.loc[g["year"] <= year_before, "Volumen OR [m3]"].sum(),
            "volume_after": g.loc[g["year"] >= year_after, "Volumen OR [m3]"].sum()
        }))
        .reset_index()
    )
    return df



def plot_biomass_by_diameter_class(s, show=False, save=True, fname = "all", percent=True, plantation_separate = True, y_max = None, case_study="", management="", add_legend=True):
    """
    We divide the biomass into three diameter classes: <20cm, 20-40cm, >40cm; and bewteen softwood and hardwood.

    Parameters:
    s: dataframe with the summaries
    show: whether to show the plot
    save: whether to save the plot
    fname: the name of the file to save the plot
    percent: whether to plot the percentages instead of the absolute values
    plantation_separate: whether to separate the plantation from the other categories
    y_max: the maximum value for the y-axis. Default is None, which means that the y-axis will be scaled automatically. If percent is True, the y-axis will be scaled to 100. If percent is False and y_max is not None, the y-axis will be scaled to y_max. If percent is False and y_max is None, the y-axis will be scaled automatically.
    add_legend: whether to add a legend to the plot
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
    df.index = [f"{i}\n-{i+9}" for i in df.index]
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
    # plt.title("Breakdown of assortments", fontsize=14)
    if percent == True:
        plt.ylabel("% of assortment")
    else:
        plt.ylabel("m3 of assortment")
        if y_max is not None:
            plt.ylim(0, y_max)
            # add only four ticks to the y-axis, round them to the nearest 10,000
            plt.yticks([0, round(y_max, -4)/4, y_max/2, 3*round(y_max, -4)/4, round(y_max, -4)])
        else:
            plt.ylim(0, df.sum(axis=1).max() * 1.1)
    # plt.xlabel("Year")
    plt.xticks(rotation=0)
    plt.xticks(ticks=range(0, len(df), 4), labels=df.index[::4])

    # Legend
    if add_legend:
        plt.legend(title="", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    else:
        plt.legend().set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig("../figures/wood_quality_"+fname+"_"+str(case_study)+"_"+str(management)+"_by_diameter.png", dpi=300, bbox_inches='tight')    
    if show:
        plt.show()



def plot_percentages_of_wood(s, save=True, show=False, 
                             fname = "all", percent=True, plantation_separate=True, 
                             y_max = None, case_study="", management="", add_legend=True):
    """
    Plots a stacked bar chart of wood quality as a percentage of total wood.

    Parameters:
    df (pd.DataFrame): DataFrame with years as index and wood volume and quality as columns.
    show (bool): Whether to display the plot.
    save (bool): Whether to save the plot as a PNG file.
    fname (str): Filename prefix for saving the plot.
    percent (bool): If True, plot percentages; if False, plot absolute values.
    plantation_separate (bool): If True, separate plantation from other categories; if False, include plantation in the other categories.
    y_max (float): Maximum value for the y-axis. If None, it will be set automatically. If percent is True, it will be set to 100.
    add_legend (bool): Whether to add a legend to the plot.
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
    df.index = [f"{i}\n-{i+9}" for i in df.index]
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
    

    # Labels and title
    #plt.title("Breakdown of assortments", fontsize=14)
    if percent:
        ax = df.plot(kind="bar", stacked=True, 
                                 figsize=(8, 5), 
                                 color=[colors[col] for col in df.columns])
        plt.ylabel("% of assortment")
    else:
        ax = (df/1000).plot(kind="bar", stacked=True, 
                                 figsize=(8, 5), 
                                 color=[colors[col] for col in df.columns])
        plt.ylabel(r"Volume [$10^3$ $m^3$]")
        if y_max is not None:
            y_max = y_max / 1000
            plt.ylim(0, y_max)
            # add only four ticks to the y-axis, round them to the nearest 10,000
            plt.yticks([0, int(round(y_max, -1)/4), int(y_max/2), int(3*round(y_max, -1)/4), int(round(y_max, -1))])
    # plt.xlabel("Year")
    plt.xticks(rotation=0)
    plt.xticks(ticks=range(0, len(df), 4), labels=df.index[::4])

    # Legend
    if add_legend:
        plt.legend(title="", loc='upper right', frameon=True)
    else:
        plt.legend().set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig("../figures/wood_"+fname+"_"+str(case_study)+"_"+str(management)+".png", dpi=300, bbox_inches='tight')    
    if show:
        plt.show()



def plot_change_in_species_comp(df, cs, man, year_before, year_after, fname_info = ""):
    # Normalize to 100%
    df['volume_before_pct'] = (df['volume_before'] / df['volume_before'].sum()) * 100
    df['volume_after_pct'] = (df['volume_after'] / df['volume_after'].sum()) * 100

    # --- 2. Independent Positioning ---
    buffer = 10 
    colors_map = {name: plt.cm.tab10(i) for i, name in enumerate(df['Baumart'])}
    # colors_map = {
    # # Softwood (stronger contrast greens/blues)
    # "Fichte": "#d9eef2",
    # "Tanne": "#7fb8c1",
    # "Foehre": "#3f8f9c",
    # "Loerche": "#1f5f6b",
    # "Ubrige Nadelholz": "#0b2f36",

    # # Hardwood (stronger contrast yellows/oranges)
    # "Buche": "#fff4bf",
    # "Eiche": "#ffd966",
    # "Esche": "#f4b942",
    # "Ahorn": "#d98c00",
    # "Ubrige Laubolz": "#8c5100"}

    # Left side: Sorted by Biomass
    df_left = df.sort_values('volume_before', ascending=False).copy()
    y_bio = 0
    left_coords = {}
    for _, row in df_left.iterrows():
        left_coords[row['Baumart']] = (y_bio, y_bio + row['volume_before_pct'])
        y_bio += row['volume_before_pct'] + buffer

    # Right side: Sorted by Volume
    df_right = df.sort_values('volume_after', ascending=False).copy()
    y_vol = 0
    right_coords = {}
    for _, row in df_right.iterrows():
        right_coords[row['Baumart']] = (y_vol, y_vol + row['volume_after_pct'])
        y_vol += row['volume_after_pct'] + buffer

    # --- 3. Plotting with Overlapping Ribbons ---
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')

    # Headers
    ax.text(0, -1, f'Before {year_before}', ha='center', va='bottom', fontweight='bold', color='#555555')
    ax.text(1, -1, f'After {year_after}', ha='center', va='bottom', fontweight='bold', color='#555555')

    def get_sigmoid_path(y1_start, y1_end, y2_start, y2_end):
        x = np.linspace(0, 1, 100)
        ease = 3 * x**2 - 2 * x**3
        upper = y1_start + (y2_start - y1_start) * ease
        lower = y1_end + (y2_end - y1_end) * ease
        return x, upper, lower

    # Draw ribbons in a specific order (or just use low alpha for all)
    min_band_size =1.
    buffer_space=10
    i=0
    j=0
    species_sorted_by_after = df.sort_values('volume_after', ascending=False)['Baumart']
    for k, species in enumerate(df.sort_values('volume_before', ascending=False)['Baumart']):
        color = colors_map[species]
        b_start, b_end = left_coords[species]
        if b_end - b_start > min_band_size:
            v_start, v_end = right_coords[species]
            offset_left = b_end + buffer_space
            j+=1
        else:
            i+=1
            b_start = offset_left
            b_end = b_start + min_band_size
            current_species_rank_after = species_sorted_by_after.to_list().index(species)
            l = current_species_rank_after
            # updating ending position of the k-species
            kth_species_after = species_sorted_by_after.values[k]
            v_start, v_end = right_coords[kth_species_after]
            delta = v_end - v_start
            if delta < min_band_size:
                right_coords[kth_species_after] = b_start, b_end
            else:
                pass
            #updating ending position of the ribbon
            v_start, v_end = right_coords[species]
            delta = v_end - v_start
            if delta < min_band_size:
                delta = min_band_size
                v_start = offset_left + (l - k)*(buffer_space+min_band_size)
                v_end = offset_left + (l - k)*(buffer_space+min_band_size)+delta
            else:
                pass
            offset_left = b_end + buffer_space
        x_path, upper, lower = get_sigmoid_path(b_start, b_end, v_start, v_end)
        
        # Ribbon with lower alpha to see overlaps
        ax.fill_between(x_path, upper, lower, color=color, alpha=0.4, edgecolor='none')
        
        # Side bars (higher alpha for stability)
        ax.fill_between([-0.05, 0], b_start, b_end, color=color, alpha=0.8)
        ax.fill_between([1, 1.05], v_start, v_end, color=color, alpha=0.8)

        # Labels      
        if species=="Ubrige Laubolz":
            species_label = "Ubrige Laub."
        elif species=="Ubrige Nadelholz":
            species_label = "Ubrige Nadel."
        elif species=="Loerche":
            species_label = "Laerche"
        else:
            species_label = species

        val_bef = df[df.Baumart==species]['volume_before_pct'].values[0]
        val_aft = df[df.Baumart==species]['volume_after_pct'].values[0]
        val_bef = f"{val_bef:.0f}%" if val_bef >= 1.0 else "<1%"
        val_aft = f"{val_aft:.0f}%" if val_aft >= 1.0 else "<1%"
        ax.text(-0.07, (b_start + b_end)/2 , f"{species_label} ({val_bef})", 
                ha='right', va='center')
        ax.text(1.07, (v_start + v_end)/2, f"{species_label} ({val_aft})", 
                ha='left', va='center')

    ax.set_xlim(-0.4, 1.4)
    ax.invert_yaxis()
    ax.axis('off')

    #plt.title(f"Species Proportion Re-ranking ({man})", pad=40, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"figures/change_in_species_{cs}_{man}_{fname_info}.png")
    plt.show()
