"""Regenerate the SZF paper figures from the summaries, without a notebook.

The plotting logic of ``plots_scaling_for_paper.ipynb``, extracted so it can run
on a compute node: no Jupyter, no display, explicit paths, one output folder.

Run:
    python make_paper_figures.py                       # Vaud + Entlebuch, BAU + WOOD
    python make_paper_figures.py --case-study Vaud
    python make_paper_figures.py --simtype 7           # RCP 4.5 instead of 8.5
    python make_paper_figures.py --data /cluster/scratch/giacomov/mainwood/summaries_for_plots \
                                --outdir ~/paper-figures

Needs only :mod:`plotting_tools_for_paper`, pandas and matplotlib. No Java, no
py4j, no SLURM -- it runs on a login node in seconds.

These are **not** the figures stage 2 draws. The paper's versions take fixed axes,
per-panel legends and a year window, and they live in their own module for that
reason; see :mod:`plotting_tools_for_paper`.

Three things the notebook did implicitly that had to become explicit:

* **simtype is compared as text.** The notebook wrote ``summaries["simtype"] == 1``
  with an integer. The pipeline writes that column as a string, so an integer
  comparison silently matches nothing and every figure comes out empty -- the same
  trap recorded in docs/06-known-issues.md.
* **Parquet.** Summaries are Parquet now; the notebook read ``.csv``. Either is
  accepted, Parquet first.
* **One output folder.** The plotting functions save to paths relative to the
  working directory, and are inconsistent about which. Every ``savefig`` is
  intercepted here and redirected, so the figures land together wherever
  ``--outdir`` says and nothing depends on where you launched from.
"""

import argparse
import datetime as dt
import os
import sys

import matplotlib
matplotlib.use("Agg")            # compute nodes have no display; must precede pyplot
import matplotlib.pyplot as plt
import pandas as pd

import paths
from plotting_tools_for_paper import (
    plot_biomass_by_diameter_class,
    plot_change_in_species_comp,
    plot_percentages_of_wood,
    prepare_data_for_sank_plot,
)

HERE = os.path.dirname(os.path.abspath(__file__))

#: Journal figure sizing, from the notebook.
RC_PARAMS = {
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "axes.labelweight": "normal",
}

#: Works for both Vaud and Entlebuch, per the notebook. A new region may need its
#: own value before the panels read correctly.
Y_MAX = 170_000

#: The published figures cover 2020 to 2160.
YEAR_LO, YEAR_HI = 2020, 2160

#: The species-composition change compares these two years.
SANKEY_BEFORE, SANKEY_AFTER = 2050, 2120

#: What each simtype means, for the log.
RCP = {"1": "RCP 8.5", "7": "RCP 4.5"}


def redirect_savefig(outdir):
    """Send every figure the plotting functions save into one folder.

    They write to 'figures/' in some functions and '../figures/' in others, both
    relative to the working directory. Rather than edit bodies that are kept
    verbatim to match the published figures, take the basename and put it where
    asked.

    Returns:
        list[str]: Filled in as figures are written.
    """
    os.makedirs(outdir, exist_ok=True)
    original = plt.savefig
    written = []

    def savefig(fname, *args, **kwargs):
        target = os.path.join(outdir, os.path.basename(str(fname)))
        kwargs.setdefault("dpi", 300)
        kwargs.setdefault("bbox_inches", "tight")
        original(target, *args, **kwargs)
        written.append(target)
        return target

    plt.savefig = savefig
    return written


def load_summary(data_dir, case_study, management, simtype):
    """Read one summary and apply the notebook's filters.

    Returns:
        pandas.DataFrame | None: The filtered frame, or None if there is no
        summary for this combination or nothing survives the filters.
    """
    base = os.path.join(data_dir, f"{case_study}_{management}")
    for extension, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = base + extension
        if os.path.isfile(path):
            break
    else:
        print(f"  no summary for {case_study} / {management} -- skipped")
        return None

    frame = reader(path)
    print(f"  {os.path.basename(path)}: {len(frame):,} rows")

    # As TEXT. An integer comparison here matches nothing and yields empty plots.
    before = len(frame)
    frame = frame[frame["simtype"].astype(str) == str(simtype)]
    print(f"    simtype == {simtype} ({RCP.get(str(simtype), 'unknown')}): "
          f"{len(frame):,} of {before:,} rows")
    if frame.empty:
        print(f"    nothing left -- check that simtype {simtype} exists in this summary")
        return None

    frame["year"] = frame["year"].astype(int)
    frame = frame[(frame["year"] >= YEAR_LO) & (frame["year"] < YEAR_HI)]
    print(f"    years {YEAR_LO}-{YEAR_HI}: {len(frame):,} rows")
    return None if frame.empty else frame


def plot_one(frame, case_study, management, simtype, add_legend):
    """The three per-(region, scenario) figures from notebook cells 7 and 9."""
    fname = str(simtype)
    plot_biomass_by_diameter_class(
        frame, show=False, save=True, percent=True, plantation_separate=False,
        fname=fname, y_max=Y_MAX, case_study=case_study, management=management,
        add_legend=add_legend)
    plot_percentages_of_wood(
        frame, save=True, show=False, fname=fname, plantation_separate=False,
        y_max=Y_MAX, case_study=case_study, management=management, add_legend=False)
    plot_percentages_of_wood(
        frame, save=True, show=False, fname=fname, percent=False,
        plantation_separate=False, y_max=Y_MAX, case_study=case_study,
        management=management, add_legend=add_legend)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=None,
                        help="folder holding <Region>_<scenario>.parquet or .csv "
                             "(default: MAINWOOD_SUMMARY_DIR)")
    parser.add_argument("--outdir", default=os.path.join(HERE, "..", "figures_paper"),
                        help="where every figure is written")
    parser.add_argument("--case-study", action="append", dest="case_studies",
                        help="repeatable; default Vaud and Entlebuch")
    parser.add_argument("--management", action="append", dest="managements",
                        help="repeatable; default BAU and WOOD")
    parser.add_argument("--simtype", default="1", help="1 = RCP 8.5 (default), 7 = RCP 4.5")
    parser.add_argument("--no-sankey", action="store_true",
                        help="skip the species-composition change figures")
    args = parser.parse_args(argv)

    case_studies = args.case_studies or ["Vaud", "Entlebuch"]
    managements = args.managements or ["BAU", "WOOD"]
    # Default to the configured summary folder, so this follows local.env like
    # everything else rather than carrying its own idea of where the data is.
    data_dir = os.path.abspath(args.data or paths.summary_dir())
    outdir = os.path.abspath(args.outdir)

    print(f"data    {data_dir}")
    print(f"outdir  {outdir}")
    print(f"simtype {args.simtype} ({RCP.get(str(args.simtype), 'unknown')})")
    print()

    if not os.path.isdir(data_dir):
        raise SystemExit(f"No such data folder: {data_dir}")

    plt.rcParams.update(RC_PARAMS)
    written = redirect_savefig(outdir)
    start = dt.datetime.now()

    for case_study in case_studies:
        frames = {}
        for management in managements:
            print(f"{case_study} / {management}")
            frame = load_summary(data_dir, case_study, management, args.simtype)
            if frame is None:
                continue
            frames[management] = frame
            # The published figures carry the legend on the BAU panel.
            plot_one(frame, case_study, management, args.simtype,
                     add_legend=(management == "BAU"))
            plt.close("all")

        if not args.no_sankey:
            for management, frame in frames.items():
                changed = prepare_data_for_sank_plot(
                    frame, year_before=SANKEY_BEFORE, year_after=SANKEY_AFTER)
                plot_change_in_species_comp(
                    changed, case_study, management,
                    year_before=SANKEY_BEFORE, year_after=SANKEY_AFTER,
                    fname_info=str(args.simtype))
                plt.close("all")
        print()

    if not written:
        print("No figures written. Check --data, --case-study and --simtype above.")
        return 1

    print(f"{len(written)} figure(s) written to {outdir}")
    for path in written:
        print(f"  {os.path.basename(path)}")
    print(f"Time taken: {dt.datetime.now() - start}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
