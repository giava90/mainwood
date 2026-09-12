"""Read the scaling grid's result files and plot how stage 1 scales.

Two panels, because they answer different questions and must not share an axis:

  * **wall time** -- what a run of this size actually costs you
  * **throughput** -- whether adding cores still buys anything, against a dashed
    ideal-linear reference anchored at the smallest core count

Throughput is the one that shows saturation. Wall time always falls with more
cores; throughput flattening is what tells you the extra cores stopped paying.

Usage:
    python plot_scaling.py                       # finds the newest scaling_* root
    python plot_scaling.py --root <bench_root>
    python plot_scaling.py --out scaling.png

Prints the table as well as plotting it, so the numbers are readable without the
figure and the aqua series is never identified by colour alone.
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")          # Euler compute nodes have no display
import matplotlib.pyplot as plt

import paths

#: Categorical slots 1-3 of the validated palette, in fixed order. Assigned to
#: sample sizes smallest-first and never cycled.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")

#: Ink, never a series colour -- text does not carry identity here.
INK = "#1a1a19"
INK_MUTED = "#6b7370"
GRID = "#e2e5e1"


def find_root(explicit=None):
    """Locate the benchmark root: the one given, or the most recent one."""
    if explicit:
        return explicit
    pattern = os.path.join(paths.data_root(), "scaling_*")
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not candidates:
        raise SystemExit(
            f"No benchmark root found under {paths.data_root()}. "
            "Pass --root, or run ./scaling_benchmark.sh first."
        )
    return max(candidates, key=os.path.getmtime)


def load_results(root):
    """Read every per-point result file under ``root/results``.

    Returns:
        list[dict]: Rows with numbers already converted, sorted by (samples, cores).
    """
    files = sorted(glob.glob(os.path.join(root, "results", "scaling_*.csv")))
    rows = []
    for path in files:
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    row["cores"] = int(row["cores"])
                    row["samples"] = int(row["samples"])
                    row["elapsed_s"] = float(row["elapsed_s"])
                    row["files_produced"] = int(row["files_produced"])
                    row["files_per_s"] = float(row["files_per_s"])
                    # csv gives strings; without this every point compares != 0
                    # and the whole grid reports as failed.
                    row["exit_code"] = int(row["exit_code"])
                    # Optional: absent from results written before phase timing.
                    for phase in ("convert_s", "sorsim_s"):
                        raw = row.get(phase, "")
                        row[phase] = float(raw) if raw not in ("", None) else None
                except (KeyError, ValueError) as exc:
                    print(f"Skipping malformed row in {path}: {exc}")
                    continue
                rows.append(row)
    return sorted(rows, key=lambda r: (r["samples"], r["cores"]))


def print_table(rows):
    """The table view. Required relief for the low-contrast series, and useful."""
    print(f"{'cores':>6} {'samples':>8} {'elapsed s':>10} {'phase1 s':>9} {'phase2 s':>9} "
          f"{'SorSim %':>9} {'files':>7} {'files/s':>9} {'exit':>5}")
    for row in rows:
        c, s = row.get("convert_s"), row.get("sorsim_s")
        share = f"{100*s/(c+s):>8.0f}%" if c is not None and s is not None and (c + s) else " " * 9
        print(f"{row['cores']:>6} {row['samples']:>8} {row['elapsed_s']:>10.1f} "
              f"{(f'{c:9.1f}' if c is not None else ' ' * 9)}"
              f"{(f'{s:10.1f}' if s is not None else ' ' * 10)}"
              f"{share} {row['files_produced']:>7} {row['files_per_s']:>9.3f} "
              f"{row['exit_code']:>5}")


def style(ax):
    """Recessive axes and grid: the marks carry the chart, not the furniture."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def plot(rows, out_path, title_suffix=""):
    """Draw the two panels and save a PNG."""
    by_samples = {}
    for row in rows:
        by_samples.setdefault(row["samples"], []).append(row)
    for series in by_samples.values():
        series.sort(key=lambda r: r["cores"])

    sample_sizes = sorted(by_samples)
    if len(sample_sizes) > len(SERIES_COLOURS):
        raise SystemExit(
            f"{len(sample_sizes)} sample sizes but only {len(SERIES_COLOURS)} "
            "categorical slots. Facet instead of adding hues."
        )

    has_phases = any(r.get("sorsim_s") is not None for r in rows)
    n_panels = 3 if has_phases else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.6))
    ax_time, ax_rate = axes[0], axes[1]
    ax_phase = axes[2] if has_phases else None
    fig.patch.set_facecolor("#fcfcfb")

    for index, samples in enumerate(sample_sizes):
        series = by_samples[samples]
        colour = SERIES_COLOURS[index]
        cores = [r["cores"] for r in series]

        ax_time.plot(cores, [r["elapsed_s"] for r in series], color=colour,
                     linewidth=2, marker="o", markersize=8, zorder=3,
                     label=f"{samples} files")
        ax_rate.plot(cores, [r["files_per_s"] for r in series], color=colour,
                     linewidth=2, marker="o", markersize=8, zorder=3,
                     label=f"{samples} files")

        # Direct labels: identity never rests on colour alone, and they are the
        # relief the palette validator requires for the low-contrast slot.
        for ax, key in ((ax_time, "elapsed_s"), (ax_rate, "files_per_s")):
            last = series[-1]
            ax.annotate(f"{samples}", (last["cores"], last[key]),
                        textcoords="offset points", xytext=(8, 0),
                        color=INK, fontsize=9, fontweight="normal",
                        va="center", zorder=4)

    # Ideal linear scaling, anchored at the smallest core count of the largest run.
    largest = by_samples[sample_sizes[-1]]
    if len(largest) > 1:
        base = largest[0]
        cores = [r["cores"] for r in largest]
        ideal = [base["files_per_s"] * c / base["cores"] for c in cores]
        ax_rate.plot(cores, ideal, color=INK_MUTED, linewidth=1.5,
                     linestyle="--", zorder=2, label="ideal linear")

    ax_time.set_title("Wall time", color=INK, fontsize=12, loc="left", pad=10)
    ax_time.set_xlabel("cores", color=INK_MUTED, fontsize=10)
    ax_time.set_ylabel("seconds", color=INK_MUTED, fontsize=10)
    ax_time.set_ylim(bottom=0)

    ax_rate.set_title("Throughput — where extra cores stop paying",
                      color=INK, fontsize=12, loc="left", pad=10)
    ax_rate.set_xlabel("cores", color=INK_MUTED, fontsize=10)
    ax_rate.set_ylabel("files per second", color=INK_MUTED, fontsize=10)
    ax_rate.set_ylim(bottom=0)

    all_cores = sorted({r["cores"] for r in rows})
    for ax in (ax_time, ax_rate):
        style(ax)
        ax.set_xticks(all_cores)
        ax.set_xlim(min(all_cores) - 1, max(all_cores) + max(2, max(all_cores) * 0.08))

    if ax_phase is not None:
        # Where the time actually goes. Phase 2 spawns a JVM per file and is what
        # grows with volume; a single elapsed number hides that entirely.
        labels, convert, sorsim = [], [], []
        for row in sorted(rows, key=lambda r: (r["samples"], r["cores"])):
            if row.get("convert_s") is None or row.get("sorsim_s") is None:
                continue
            labels.append(f"{row['cores']}c" + chr(10) + f"{row['samples']}f")
            convert.append(row["convert_s"])
            sorsim.append(row["sorsim_s"])

        positions = range(len(labels))
        ax_phase.bar(positions, convert, color=SERIES_COLOURS[0], label="phase 1 — tree lists",
                     zorder=3, width=0.7)
        ax_phase.bar(positions, sorsim, bottom=convert, color=SERIES_COLOURS[1],
                     label="phase 2 — SorSim", zorder=3, width=0.7,
                     edgecolor="#fcfcfb", linewidth=2)
        ax_phase.set_xticks(list(positions))
        ax_phase.set_xticklabels(labels, fontsize=8)
        ax_phase.set_title("Where the time goes", color=INK, fontsize=12, loc="left", pad=10)
        ax_phase.set_ylabel("seconds", color=INK_MUTED, fontsize=10)
        ax_phase.set_ylim(bottom=0)
        style(ax_phase)
        ax_phase.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper left")

    ax_rate.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper left")

    fig.suptitle(f"Stage 1 scaling{title_suffix}", color=INK, fontsize=14,
                 x=0.008, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="benchmark root (default: newest scaling_* found)")
    parser.add_argument("--out", help="output PNG (default: <root>/scaling.png)")
    args = parser.parse_args(argv)

    root = find_root(args.root)
    rows = load_results(root)
    if not rows:
        raise SystemExit(
            f"No result files in {os.path.join(root, 'results')}. "
            "Have the jobs finished? Check with: squeue -u $USER"
        )

    print(f"Benchmark root: {root}")
    print(f"{len(rows)} grid point(s)\n")
    print_table(rows)

    failed = [r for r in rows if r["exit_code"] != 0]
    if failed:
        print(f"\n{len(failed)} point(s) exited non-zero and are plotted anyway -- "
              "their timings are not comparable.")

    first = rows[0]
    suffix = f" — {first['case_study']} / {first['scenario']} / {first['cohort']}"
    out_path = args.out or os.path.join(root, "scaling.png")
    plot(rows, out_path, suffix)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
