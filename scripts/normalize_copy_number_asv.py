#!/usr/bin/env python3
# =============================================================
# 16S rRNA copy-number normalization pipeline
# Author: Diana Oliveira
#
# Purpose:
#   Prepare Novogene ASV count tables for MicrobiomeAnalyst after
#   correction for estimated 16S rRNA gene copy number.
#
# Main choices:
#   - Keep ASV IDs as #NAME.
#   - Use a precomputed copy-number table:
#       data/16S_copy_numbers.csv
#     with columns:
#       rank,name,mean
#   - Assign copy numbers by hierarchical matching:
#       Species > Genus > Family > Order > Class > Phylum > no correction
#   - If no match is found, CopyNumber = 1, meaning the ASV is left uncorrected.
#   - Do NOT rarefy after copy-number correction.
#   - Apply Total Sum Scaling (TSS) to obtain relative abundance.
#   - Export a scaled version multiplied by 1000 for LEfSe compatibility.
#
# Recommended use from project root:
#   python scripts/normalize_copy_number_asv.py
# =============================================================

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TAXONOMY_LEVELS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
MATCH_LEVELS = ["Species", "Genus", "Family", "Order", "Class", "Phylum"]

SILVA_PREFIXES = {
    "Kingdom": "k__",
    "Phylum": "p__",
    "Class": "c__",
    "Order": "o__",
    "Family": "f__",
    "Genus": "g__",
    "Species": "s__",
}

PREFIX_TO_LEVEL = {
    "k__": "Kingdom",
    "d__": "Kingdom",
    "p__": "Phylum",
    "c__": "Class",
    "o__": "Order",
    "f__": "Family",
    "g__": "Genus",
    "s__": "Species",
}

RANK_MAP = {
    "species": "Species",
    "genus": "Genus",
    "family": "Family",
    "order": "Order",
    "class": "Class",
    "phylum": "Phylum",
    "domain": "Kingdom",
    "kingdom": "Kingdom",
}

PLACEHOLDERS = {
    "",
    "none",
    "na",
    "nan",
    "unassigned",
    "unclassified",
    "unidentified",
    "unknown",
    "uncultured",
}


# -------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------

def log(message: str) -> None:
    print(message, flush=True)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def canonical_name(value: object) -> str:
    """
    Normalize taxon names for matching SILVA labels to copy-number table names.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.replace("_", " ")
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    if text in PLACEHOLDERS:
        return ""

    return text


# -------------------------------------------------------------
# Taxonomy parsing
# -------------------------------------------------------------

def clean_silva_value(raw_value: object, level: str) -> str:
    if pd.isna(raw_value):
        return ""

    value = str(raw_value).strip()
    value = value.replace("\t", " ")
    value = re.sub(r"\s+", " ", value)

    if canonical_name(value) == "":
        return ""

    # Copy-number table species names usually use spaces instead of underscores.
    if level == "Species":
        value = value.replace("_", " ")

    return value


def parse_silva_taxonomy(taxonomy: object) -> Dict[str, str]:
    """
    Parse a SILVA-style taxonomy string into fixed columns.
    Example:
    k__Bacteria;p__Firmicutes;c__Clostridia;o__Lachnospirales;...
    """
    parsed = {level: "" for level in TAXONOMY_LEVELS}

    if pd.isna(taxonomy):
        return parsed

    for field in str(taxonomy).split(";"):
        field = field.strip()

        for prefix, level in PREFIX_TO_LEVEL.items():
            if field.startswith(prefix):
                parsed[level] = clean_silva_value(field[len(prefix):], level)
                break

    return parsed


def add_taxonomy_columns(asv_df: pd.DataFrame, taxonomy_col: str) -> pd.DataFrame:
    parsed = asv_df[taxonomy_col].apply(parse_silva_taxonomy).apply(pd.Series)
    return pd.concat([asv_df, parsed[TAXONOMY_LEVELS]], axis=1)


def fill_unclassified_species(row: pd.Series) -> str:
    species = str(row.get("Species", "") or "").strip()
    genus = str(row.get("Genus", "") or "").strip()
    family = str(row.get("Family", "") or "").strip()

    if species:
        return species
    if genus:
        return f"unclassified {genus}"
    if family:
        return f"unclassified {family}"
    return "unclassified"


def create_taxonomy_table(asv_df: pd.DataFrame, asv_id_col: str) -> pd.DataFrame:
    """
    Create MicrobiomeAnalyst taxonomy table.
    """
    tax = pd.DataFrame()
    tax["#TAXONOMY"] = asv_df[asv_id_col].astype(str)

    for level in TAXONOMY_LEVELS:
        if level == "Species":
            tax[level] = asv_df.apply(fill_unclassified_species, axis=1)
        else:
            tax[level] = asv_df[level].replace("", np.nan).fillna("unclassified")

    return tax


# -------------------------------------------------------------
# Copy-number table loading
# -------------------------------------------------------------

def build_copy_number_lookups(copydb_file: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
    """
    Build copy-number lookups from a precomputed table.

    Expected columns:
      rank,name,mean

    Example:
      species,Bifidobacterium longum,4.00
      genus,Bifidobacterium,4.15
      family,Bifidobacteriaceae,4.20
    """
    log("\nStep 2: Loading precomputed 16S copy-number table...")

    copydb = pd.read_csv(copydb_file)
    copydb.columns = [str(c).strip() for c in copydb.columns]

    required = {"rank", "name", "mean"}
    missing = required - set(copydb.columns)

    if missing:
        raise ValueError(
            f"Copy-number table is missing required columns: {sorted(missing)}"
        )

    lookup_lists: Dict[str, Dict[str, List[float]]] = {
        level: defaultdict(list) for level in MATCH_LEVELS
    }

    all_values: List[float] = []

    for _, row in copydb.iterrows():
        rank_raw = str(row["rank"]).strip().lower()
        level = RANK_MAP.get(rank_raw)

        if level not in MATCH_LEVELS:
            continue

        key = canonical_name(row["name"])

        if not key:
            continue

        try:
            value = float(row["mean"])
        except (TypeError, ValueError):
            continue

        if not np.isfinite(value) or value <= 0:
            continue

        lookup_lists[level][key].append(value)
        all_values.append(value)

    if not all_values:
        raise ValueError("No valid copy-number values were found in the copy-number table.")

    # Usually this table is already aggregated.
    # If duplicates exist, average them.
    lookups = {
        level: {
            taxon: float(np.mean(values))
            for taxon, values in values_by_taxon.items()
        }
        for level, values_by_taxon in lookup_lists.items()
    }

    stats = {
        "global_mean_reference_only": float(np.mean(all_values)),
        "global_median_reference_only": float(np.median(all_values)),
        "n_records_used": int(len(all_values)),
    }

    for level in MATCH_LEVELS:
        log(f"  {level:<8}: {len(lookups[level])}")

    log(f"  Reference global median: {stats['global_median_reference_only']:.3f}")

    return lookups, stats


# -------------------------------------------------------------
# Copy-number assignment
# -------------------------------------------------------------

def assign_copy_number_to_row(
    row: pd.Series,
    lookups: Dict[str, Dict[str, float]],
) -> Tuple[float, str, str]:
    """
    Assign copy number using simple hierarchical matching:
    Species > Genus > Family > Order > Class > Phylum > no correction.
    """
    for level in MATCH_LEVELS:
        taxon = row.get(level, "")
        key = canonical_name(taxon)

        if key and key in lookups[level]:
            return lookups[level][key], level.lower(), str(taxon)

    # If no match is found, leave the ASV uncorrected.
    return 1.0, "no_match_assumed_1", ""


def assign_copy_numbers(
    asv_df: pd.DataFrame,
    asv_id_col: str,
    lookups: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    log("\nStep 3: Assigning 16S copy numbers to ASVs...")

    records = []

    for _, row in asv_df.iterrows():
        copy_number, match_level, matched_taxon = assign_copy_number_to_row(row, lookups)

        records.append({
            "#NAME": str(row[asv_id_col]),
            "CopyNumber": copy_number,
            "MatchLevel": match_level,
            "MatchedTaxon": matched_taxon,
            **{level: row.get(level, "") for level in TAXONOMY_LEVELS},
        })

    mapping = pd.DataFrame(records)
    counts = Counter(mapping["MatchLevel"])

    for level in [
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "no_match_assumed_1",
    ]:
        log(f"  {level:<20}: {counts[level]}")

    return mapping


# -------------------------------------------------------------
# Normalization and aggregation
# -------------------------------------------------------------

def identify_input_columns(df: pd.DataFrame) -> Tuple[str, str, List[str]]:
    """
    Identify ASV ID column, taxonomy column and sample columns.
    """
    if "#OTU_num" in df.columns:
        asv_id_col = "#OTU_num"
    elif "#NAME" in df.columns:
        asv_id_col = "#NAME"
    else:
        asv_id_col = df.columns[0]

    if "Taxonomy" in df.columns:
        taxonomy_col = "Taxonomy"
    else:
        candidates = [
            c for c in df.columns
            if c.lower() in {"taxonomy", "taxon", "tax"}
        ]

        if not candidates:
            raise ValueError(
                "Could not find taxonomy column. Expected a column named 'Taxonomy'."
            )

        taxonomy_col = candidates[0]

    excluded = {asv_id_col, taxonomy_col}
    sample_cols = []

    for col in df.columns:
        if col in excluded:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        if numeric.notna().any():
            sample_cols.append(col)

    if not sample_cols:
        raise ValueError("No numeric sample columns were detected in the ASV table.")

    return asv_id_col, taxonomy_col, sample_cols


def optional_filter_asvs(
    df: pd.DataFrame,
    sample_cols: List[str],
    min_total_count: float,
    min_prevalence: int,
) -> Tuple[pd.DataFrame, int]:
    """
    Optional ASV filtering.
    Defaults are zero, so no filtering is applied.
    """
    keep = pd.Series(True, index=df.index)

    if min_total_count > 0:
        keep &= df[sample_cols].sum(axis=1) >= min_total_count

    if min_prevalence > 0:
        keep &= (df[sample_cols] > 0).sum(axis=1) >= min_prevalence

    removed = int((~keep).sum())

    return df.loc[keep].copy(), removed


def total_sum_scale(df: pd.DataFrame, sample_cols: List[str]) -> pd.DataFrame:
    """
    Convert corrected abundances to relative abundance by sample.
    Each sample column will sum to 1.
    """
    out = df.copy()
    totals = out[sample_cols].sum(axis=0)

    zero_cols = totals[totals == 0].index.tolist()

    if zero_cols:
        raise ValueError(
            f"These samples have total abundance zero after correction: {zero_cols}"
        )

    out[sample_cols] = out[sample_cols].div(totals, axis=1)

    return out


def lineage_label(row: pd.Series, target_level: str) -> str:
    """
    Build a SILVA-like lineage label up to target_level.
    Used for aggregated taxonomic tables.
    """
    parts = []

    for level in TAXONOMY_LEVELS:
        value = str(row.get(level, "") or "").strip() or "unclassified"
        parts.append(f"{SILVA_PREFIXES[level]}{value}")

        if level == target_level:
            break

    return ";".join(parts)


def aggregate_by_taxonomic_level(
    relative_df: pd.DataFrame,
    asv_df_with_tax: pd.DataFrame,
    asv_id_col: str,
    sample_cols: List[str],
    level: str,
) -> pd.DataFrame:
    """
    Aggregate ASV relative abundances by a taxonomic level.
    """
    if level not in TAXONOMY_LEVELS:
        raise ValueError(f"Unsupported aggregation level: {level}")

    tax_labels = asv_df_with_tax[[asv_id_col] + TAXONOMY_LEVELS].copy()
    tax_labels["TaxonLabel"] = tax_labels.apply(
        lambda row: lineage_label(row, level),
        axis=1,
    )

    tax_labels = tax_labels.rename(columns={asv_id_col: "#NAME"})

    merged = relative_df.merge(
        tax_labels[["#NAME", "TaxonLabel"]],
        on="#NAME",
        how="left",
    )

    if merged["TaxonLabel"].isna().any():
        raise ValueError(
            "Some ASVs could not be matched to taxonomy labels during aggregation."
        )

    agg = merged.groupby("TaxonLabel", as_index=False)[sample_cols].sum()
    agg = agg.rename(columns={"TaxonLabel": "#NAME"})

    return agg


# -------------------------------------------------------------
# Metadata and QC
# -------------------------------------------------------------

def validate_metadata(
    metadata_file: Optional[Path],
    sample_cols: List[str],
) -> Dict[str, object]:
    qc: Dict[str, object] = {}

    if metadata_file is None:
        qc["metadata_file"] = "not provided"
        return qc

    if not metadata_file.exists():
        qc["metadata_file"] = f"not found: {metadata_file}"
        return qc

    meta = pd.read_csv(metadata_file)

    if meta.empty:
        qc["metadata_file"] = "empty"
        return qc

    meta_id_col = "#NAME" if "#NAME" in meta.columns else meta.columns[0]

    metadata_samples = meta[meta_id_col].astype(str).tolist()
    sample_set = set(map(str, sample_cols))
    metadata_set = set(metadata_samples)

    qc["metadata_file"] = str(metadata_file)
    qc["metadata_samples"] = len(metadata_samples)
    qc["metadata_duplicate_sample_ids"] = int(pd.Series(metadata_samples).duplicated().sum())
    qc["samples_missing_from_metadata"] = ";".join(sorted(sample_set - metadata_set))
    qc["metadata_rows_not_in_asv_table"] = ";".join(sorted(metadata_set - sample_set))

    return qc


def write_qc_summary(qc: Dict[str, object], output_file: Path) -> None:
    rows = [{"Metric": key, "Value": value} for key, value in qc.items()]
    pd.DataFrame(rows).to_csv(output_file, index=False)


# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    asv_file = Path(args.asv)
    copydb_file = Path(args.copydb)
    metadata_file = Path(args.metadata) if args.metadata else None
    outdir = Path(args.outdir)

    safe_mkdir(outdir)

    if not asv_file.exists():
        raise FileNotFoundError(f"ASV table not found: {asv_file}")

    if not copydb_file.exists():
        raise FileNotFoundError(f"Copy-number table not found: {copydb_file}")

    log("Step 1: Loading Novogene ASV table...")

    asv_raw = pd.read_csv(asv_file, sep="\t")
    asv_id_col, taxonomy_col, sample_cols = identify_input_columns(asv_raw)

    asv_raw[sample_cols] = (
        asv_raw[sample_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    before_unassigned = len(asv_raw)

    asv_raw = asv_raw.dropna(subset=[taxonomy_col]).copy()
    asv_raw = asv_raw[
        asv_raw[taxonomy_col].astype(str).str.lower() != "unassigned"
    ].copy()

    removed_unassigned = before_unassigned - len(asv_raw)

    asv_raw[asv_id_col] = asv_raw[asv_id_col].astype(str)

    duplicate_asv_ids = int(asv_raw[asv_id_col].duplicated().sum())

    if duplicate_asv_ids > 0:
        duplicated = (
            asv_raw.loc[asv_raw[asv_id_col].duplicated(), asv_id_col]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Duplicate ASV IDs found in input table. Examples: {duplicated}"
        )

    asv_raw, removed_by_filter = optional_filter_asvs(
        asv_raw,
        sample_cols,
        min_total_count=args.min_total_count,
        min_prevalence=args.min_prevalence,
    )

    asv_tax = add_taxonomy_columns(asv_raw, taxonomy_col)

    log(f"  ASVs loaded:              {len(asv_tax)}")
    log(f"  Samples detected:         {len(sample_cols)}")
    log(f"  Removed unassigned rows:  {removed_unassigned}")
    log(f"  Removed by filters:       {removed_by_filter}")

    for level in TAXONOMY_LEVELS:
        n_level = (asv_tax[level].astype(str) != "").sum()
        log(f"  With {level:<7}:           {n_level}")

    lookups, copydb_stats = build_copy_number_lookups(copydb_file)

    mapping = assign_copy_numbers(
        asv_df=asv_tax,
        asv_id_col=asv_id_col,
        lookups=lookups,
    )

    log("\nStep 4: Correcting ASV counts by estimated 16S copy number...")

    copy_numbers = (
        mapping
        .set_index("#NAME")
        .loc[asv_tax[asv_id_col], "CopyNumber"]
        .to_numpy()
    )

    corrected = pd.DataFrame({
        "#NAME": asv_tax[asv_id_col].astype(str)
    })

    corrected[sample_cols] = asv_tax[sample_cols].div(copy_numbers, axis=0)

    log("Step 5: Applying Total Sum Scaling (relative abundance)...")

    relative = total_sum_scale(corrected, sample_cols)

    scaled = relative.copy()
    scaled[sample_cols] = scaled[sample_cols] * float(args.scale_factor)

    taxonomy_table = create_taxonomy_table(asv_tax, asv_id_col)

    log("\nStep 6: Exporting outputs...")

    corrected_path = outdir / "CopyNumberCorrected_ASV_abundance.csv"
    relative_path = outdir / "Normalized_ASV_Table_MicrobiomeAnalyst.csv"
    scaled_path = outdir / f"Normalized_ASV_Table_MicrobiomeAnalyst_scaled{args.scale_factor:g}.csv"
    taxonomy_path = outdir / "Taxonomy_Table_MicrobiomeAnalyst.csv"
    mapping_path = outdir / "ASV_CopyNumber_Mapping.csv"
    qc_path = outdir / "QC_Normalization_Summary.csv"

    corrected.to_csv(corrected_path, index=False)
    relative.to_csv(relative_path, index=False)
    scaled.to_csv(scaled_path, index=False)
    taxonomy_table.to_csv(taxonomy_path, index=False)
    mapping.to_csv(mapping_path, index=False)

    for level in args.aggregate_levels:
        agg = aggregate_by_taxonomic_level(
            relative_df=relative,
            asv_df_with_tax=asv_tax,
            asv_id_col=asv_id_col,
            sample_cols=sample_cols,
            level=level,
        )

        agg_path = outdir / f"{level}_copy_corrected_relative_abundance.csv"
        agg.to_csv(agg_path, index=False)

        log(f"  Wrote {level:<7} aggregate table: {agg_path}")

    raw_totals = asv_tax[sample_cols].sum(axis=0)
    corrected_totals = corrected[sample_cols].sum(axis=0)
    relative_totals = relative[sample_cols].sum(axis=0)

    match_counts = Counter(mapping["MatchLevel"])

    no_match_mask = mapping["MatchLevel"] == "no_match_assumed_1"
    no_match_count = int(no_match_mask.sum())

    no_match_raw_abundance = (
        asv_tax
        .loc[no_match_mask.to_numpy(), sample_cols]
        .sum()
        .sum()
    )

    total_raw_abundance = asv_tax[sample_cols].sum().sum()

    if total_raw_abundance > 0:
        no_match_raw_abundance_pct = 100 * no_match_raw_abundance / total_raw_abundance
    else:
        no_match_raw_abundance_pct = 0

    qc: Dict[str, object] = {
        "input_asv_file": str(asv_file),
        "input_copy_number_file": str(copydb_file),
        "output_directory": str(outdir),
        "asv_id_column": asv_id_col,
        "taxonomy_column": taxonomy_col,
        "n_asvs_final": len(asv_tax),
        "n_samples": len(sample_cols),
        "duplicate_asv_ids": duplicate_asv_ids,
        "removed_unassigned_rows": removed_unassigned,
        "removed_by_filters": removed_by_filter,
        "min_total_count_filter": args.min_total_count,
        "min_prevalence_filter": args.min_prevalence,
        "fallback_strategy": "no match = copy number 1, left uncorrected",
        "copydb_global_mean_reference_only": round(copydb_stats["global_mean_reference_only"], 6),
        "copydb_global_median_reference_only": round(copydb_stats["global_median_reference_only"], 6),
        "copydb_records_used": copydb_stats["n_records_used"],
        "scale_factor_for_lefse": args.scale_factor,
        "asvs_without_copy_number_match_left_uncorrected": no_match_count,
        "raw_abundance_percent_without_copy_number_match_left_uncorrected": round(
            float(no_match_raw_abundance_pct),
            6,
        ),
        "raw_sample_total_min": round(float(raw_totals.min()), 6),
        "raw_sample_total_max": round(float(raw_totals.max()), 6),
        "corrected_sample_total_min": round(float(corrected_totals.min()), 6),
        "corrected_sample_total_max": round(float(corrected_totals.max()), 6),
        "relative_sample_total_min": round(float(relative_totals.min()), 12),
        "relative_sample_total_max": round(float(relative_totals.max()), 12),
        "taxonomy_table_duplicate_ids": int(taxonomy_table["#TAXONOMY"].duplicated().sum()),
    }

    for level in MATCH_LEVELS:
        qc[f"copydb_lookup_size_{level.lower()}"] = len(lookups[level])

    for level in [
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "no_match_assumed_1",
    ]:
        qc[f"copy_number_match_{level}"] = match_counts[level]

    qc.update(validate_metadata(metadata_file, sample_cols))

    write_qc_summary(qc, qc_path)

    log(f"  Wrote corrected ASV table:       {corrected_path}")
    log(f"  Wrote relative ASV table:        {relative_path}")
    log(f"  Wrote scaled LEfSe ASV table:    {scaled_path}")
    log(f"  Wrote taxonomy table:            {taxonomy_path}")
    log(f"  Wrote copy-number mapping table: {mapping_path}")
    log(f"  Wrote QC summary:                {qc_path}")

    log("\nDone.")
    log("Recommended MicrobiomeAnalyst settings:")
    log("  - Upload Normalized_ASV_Table_MicrobiomeAnalyst.csv as normalized data.")
    log("  - Upload Taxonomy_Table_MicrobiomeAnalyst.csv as taxonomy table.")
    log("  - Choose SILVA taxonomy labels if Novogene used SILVA.")
    log("  - In normalization: Do not rarefy, do not scale, do not transform.")
    log("  - For LEfSe, use the scaled table if the relative table causes minimum-count errors.")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Novogene ASV counts by precomputed 16S rRNA gene copy numbers "
            "and export MicrobiomeAnalyst-ready tables."
        )
    )

    parser.add_argument(
        "--asv",
        default="data/featureTable.sample.total.absolute.txt",
        help=(
            "Path to Novogene ASV absolute count table. "
            "Default: data/featureTable.sample.total.absolute.txt"
        ),
    )

    parser.add_argument(
        "--copydb",
        default="data/16S_copy_numbers.csv",
        help=(
            "Path to precomputed copy-number table with columns rank,name,mean. "
            "Default: data/16S_copy_numbers.csv"
        ),
    )

    parser.add_argument(
        "--metadata",
        default="data/metadata.csv",
        help="Optional metadata file for QC validation. Default: data/metadata.csv",
    )

    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory. Default: outputs",
    )

    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1000.0,
        help=(
            "Constant used to scale relative abundance table for LEfSe compatibility. "
            "Default: 1000"
        ),
    )

    parser.add_argument(
        "--min-total-count",
        type=float,
        default=0.0,
        help=(
            "Optional ASV filter: keep ASVs with total raw count >= this value. "
            "Default: 0, no filtering."
        ),
    )

    parser.add_argument(
        "--min-prevalence",
        type=int,
        default=0,
        help=(
            "Optional ASV filter: keep ASVs present in at least this many samples. "
            "Default: 0, no filtering."
        ),
    )

    parser.add_argument(
        "--aggregate-levels",
        nargs="*",
        default=["Phylum", "Family", "Genus", "Order"],
        choices=TAXONOMY_LEVELS,
        help=(
            "Taxonomic levels to export as aggregated relative abundance tables. "
            "Default: Phylum Family Genus Order"
        ),
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        run_pipeline(parse_args())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)