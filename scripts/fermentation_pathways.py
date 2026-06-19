#!/usr/bin/env python3
# =============================================================
# Fermentation / SCFA pathway abundances from a Tax4Fun KO table
# Author: Diana Oliveira
#
# What it does (kept deliberately simple):
#   1. reads the MicrobiomeAnalyst (Tax4Fun) KO abundance table;
#   2. keeps only a short, curated list of fermentation / SCFA KEGG
#      pathways (butanoate, propanoate, sugar catabolism, ...);
#   3. sums the KO abundances into those pathways, per sample;
#   4. draws one heatmap (pathways x condition, mean %) and saves a CSV.
#
# KO -> pathway comes from KEGG. It uses the file already cached by the
# full script (data/kegg_cache/link_ko_pathway.txt); if that file is not
# there, it downloads just that one list once.
#
# Run from the project root:
#   python scripts/fermentation_pathways.py
# =============================================================

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve default paths from the project root, so it runs from anywhere.
ROOT = Path(__file__).resolve().parent.parent
KO_FILE = ROOT / "microbiomeanalyst_results/05_Tax4Fun/functionalprof_tax4fun.csv"
META_FILE = ROOT / "microbiomeanalyst_results/05_Tax4Fun/metadata.csv"
CACHE = ROOT / "data/kegg_cache/link_ko_pathway.txt"
OUTDIR = ROOT / "outputs/kegg_pathways"

# Curated fermentation / SCFA pathways: pathway number -> short label.
PATHWAYS = {
    "00010": "Glycolysis / Gluconeogenesis",
    "00030": "Pentose phosphate",
    "00040": "Pentose & glucuronate interconv.",
    "00051": "Fructose & mannose",
    "00052": "Galactose",
    "00500": "Starch & sucrose",
    "00520": "Amino & nucleotide sugar",
    "00620": "Pyruvate",
    "00640": "Propanoate (propionate)",
    "00650": "Butanoate (butyrate)",
}

# Condition order and colours for the figure.
GROUP_ORDER = ["Inoculum", "NC", "FrutaloseOFP", "FrutafitIQ",
               "MangabaPulp", "MangabaExt"]


def load_ko_to_pathway() -> pd.DataFrame:
    """Return a KO -> pathway-number table from the KEGG cache (or download)."""
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading KO->pathway list from KEGG (one time)...")
        url = "https://rest.kegg.jp/link/pathway/ko"
        urllib.request.urlretrieve(url, CACHE)

    rows = []
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        ko, _, path = line.partition("\t")
        if path.startswith("path:ko"):
            rows.append((ko.replace("ko:", ""), path.replace("path:ko", "")))
    return pd.DataFrame(rows, columns=["KO", "pathway"])


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1. KO abundances (each sample column already sums to ~1e6).
    ko = pd.read_csv(KO_FILE)
    ko = ko.rename(columns={ko.columns[0]: "KO"})
    samples = [c for c in ko.columns if c != "KO"]

    # 2. map KO -> pathway and keep only the curated set.
    link = load_ko_to_pathway()
    link = link[link["pathway"].isin(PATHWAYS)]
    merged = ko.merge(link, on="KO")

    # 3. sum KO -> pathway per sample, then express as % of total signal.
    path_abs = merged.groupby("pathway")[samples].sum()
    path_pct = path_abs / ko[samples].sum() * 100.0
    path_pct.index = [PATHWAYS[p] for p in path_pct.index]
    path_pct.to_csv(OUTDIR / "fermentation_pathways_per_sample.csv")

    # 4. mean per condition for the heatmap.
    meta = pd.read_csv(META_FILE)
    name_col = "#NAME" if "#NAME" in meta.columns else meta.columns[0]
    group = meta.set_index(meta[name_col].astype(str))["SampleType"]
    groups = [g for g in GROUP_ORDER if g in set(group)]
    means = pd.DataFrame(
        {g: path_pct[[s for s in samples if group.get(s) == g]].mean(axis=1)
         for g in groups})

    # colour = pattern across conditions (z-score per row); text = raw %.
    z = means.sub(means.mean(axis=1), axis=0)
    z = z.div(means.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(1.1 * len(groups) + 4, 0.55 * len(means) + 2))
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            ax.text(j, i, f"{means.iat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("z-score")
    ax.set_title("Fermentation / SCFA pathways - mean relative abundance (%)")
    plt.tight_layout()
    fig.savefig(OUTDIR / "fermentation_pathways_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"Done. {len(means)} pathways, {len(samples)} samples.")
    print(f"  {OUTDIR / 'fermentation_pathways_heatmap.png'}")
    print(f"  {OUTDIR / 'fermentation_pathways_per_sample.csv'}")


if __name__ == "__main__":
    main()
