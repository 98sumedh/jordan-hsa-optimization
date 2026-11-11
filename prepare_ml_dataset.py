"""
ML Dataset Preparation Script
==============================

Purpose: Merge climate data (102 CSV files: 6 variables × 17 HSAs) with
         diarrheal disease data to create a unified dataset for machine
         learning modeling.

Strategy:
    - Prioritize keeping all 17 HSAs (national coverage)
    - Filter to valid date range: 2022-06-27 to 2024-01-29 (84 weeks)
    - Drop variables with >40% missing (NaN/null) data
    - Preserve remaining missing values for modeling-phase imputation
    - Zeros are valid values in arid regions (not treated as missing)
    - No imputation at this stage (prevents data leakage)
    - No train/val/test split (done during modeling phase)

Data Leakage Prevention:
    - Missing value imputation deferred to modeling phase
    - Feature selection based only on correlation and redundancy
    - Train/val/test split will be done during modeling
    - Imputation parameters learned from train set only

Input:
    - Climate files: {OUT_DIR}/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/*.csv
    - Diagnosis data: {OUT_DIR}/{NETWORK}_{HSA_MODE}_weekly_{DISEASE_FOCUS}_adjusted_{BOUNDARY_VERSION}.csv

Output:
    - {NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}.csv: Complete merged dataset (may contain NaNs)
    - {NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}_metadata.json: Feature descriptions and summary

Author: ML Modeling Team
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION DEFAULTS (can be overridden via CLI)
# ============================================================================

# Default directories
DEFAULT_PIPELINE_OUT_DIR = os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out"))
DEFAULT_BOUNDARY_VERSION = os.environ.get("BOUNDARY_VERSION", os.environ.get("PIPELINE_VERSION", "v7"))
DEFAULT_CLIMATE_DIR = str(Path(DEFAULT_PIPELINE_OUT_DIR) / f"DRIVE_CLIMATE_BY_HSA_DOWNLOAD_{DEFAULT_BOUNDARY_VERSION.upper()}" / "FINAL_HSA_CLIMATE")
DEFAULT_OUTPUT_DIR = str(Path(DEFAULT_PIPELINE_OUT_DIR) / "modeling")

# Default data cleaning thresholds
DEFAULT_MISSING_THRESHOLD = 0.40  # Drop variables with >40% missing data
DEFAULT_CORRELATION_THRESHOLD = 0.95  # Remove highly correlated features (r > 0.95)

# Study period dates have no hardcoded defaults; they must be passed via --start-date
# and --end-date (or START_DATE/END_DATE env vars) from the calling notebook.

from disease_focus import canonical_group, slug

# Climate file suffixes (fixed structure from GEE exports)
# Elevation is deliberately absent. It is static per HSA, so GEE exports it for
# a single week; merging that onto an 84-week panel leaves 98.8% of rows empty
# and the missing-data rule then discards the column anyway. Excluding it here
# says so outright instead of letting a threshold silently do it, and keeps the
# weekly dataset independent of elevation re-exports. The within-HSA elevation
# analysis (S4.5) reads the exports directly.
CLIMATE_SUFFIXES = [
    'precip_lags.csv',
    'tempdew_wind_lags.csv',
    'evapERA5_lags.csv',
    'soilmoistERA5_lags.csv',
    'water_balance.csv',
]

# These will be set by parse_args() in main()
CLIMATE_DIR = None
OUTPUT_DIR = None
NETWORK = None
HSA_MODE = None
DISEASE_FOCUS = None
TARGET_COL = None
DIAGNOSIS_FILE = None
MISSING_THRESHOLD = None
CORRELATION_THRESHOLD = None
START_DATE = None
END_DATE = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_facility_name(name):
    """Standardize facility names for matching: any run of spaces/underscores -> one underscore, trimmed."""
    return '_'.join(str(name).replace('_', ' ').split())


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ML dataset for modeling")
    parser.add_argument("--network", default=os.environ.get("NETWORK", "INF"),
                        help="Network type (INF, NCD, etc.)")
    parser.add_argument("--hsa-mode", default=os.environ.get("HSA_MODE", "footprint"),
                        help="HSA optimization mode")
    parser.add_argument("--disease-focus", default=os.environ.get("DISEASE_FOCUS"),
                        help="Disease focus (e.g., diarrheal, hypertension)")
    parser.add_argument("--out-dir", default=os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", DEFAULT_PIPELINE_OUT_DIR)),
                        help="Pipeline output directory containing weekly disease count files")
    # Left unset so it can be derived from --out-dir below. Giving it a concrete
    # default here pinned it to the directory resolved at import time, which
    # made --out-dir silently not move where climate is read from.
    parser.add_argument("--climate-dir", default=os.environ.get("CLIMATE_DIR"),
                        help=f"Directory containing climate CSV files (default: derived from --out-dir, e.g. {DEFAULT_CLIMATE_DIR})")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"),
                        help=f"Output directory for modeling dataset (default: derived from --out-dir, e.g. {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--missing-threshold", type=float,
                        default=float(os.environ.get("MISSING_THRESHOLD", DEFAULT_MISSING_THRESHOLD)),
                        help=f"Drop features with missing data above this fraction (default: {DEFAULT_MISSING_THRESHOLD})")
    parser.add_argument("--correlation-threshold", type=float,
                        default=float(os.environ.get("CORRELATION_THRESHOLD", DEFAULT_CORRELATION_THRESHOLD)),
                        help=f"Remove features with correlation above this (default: {DEFAULT_CORRELATION_THRESHOLD})")
    parser.add_argument("--start-date", default=os.environ.get("START_DATE"),
                        help="Start date for valid data range (YYYY-MM-DD). Required.")
    parser.add_argument("--end-date", default=os.environ.get("END_DATE"),
                        help="End date for valid data range (YYYY-MM-DD). Required.")
    parser.add_argument("--boundary-version", default=DEFAULT_BOUNDARY_VERSION,
                        help="HSA boundary version (v6, v7, v8). Drives climate-dir default and output filename.")
    return parser.parse_args()


def load_climate_file(filepath):
    """Load a climate CSV file and clean it"""
    df = pd.read_csv(filepath)

    # Drop unnecessary columns
    cols_to_drop = [col for col in df.columns if col in ['system:index', '.geo']]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Clean facility name
    if 'FacilityName' in df.columns:
        df['FacilityName'] = df['FacilityName'].apply(clean_facility_name)

    # Convert week_start to datetime
    if 'week_start' in df.columns:
        df['week_start'] = pd.to_datetime(df['week_start'])

    return df


def merge_climate_files_for_hsa(hsa_name, climate_dir, suffixes, network=None):
    """
    Merge all 6 climate file types for a single HSA

    Args:
        hsa_name: Name of the HSA/facility
        climate_dir: Directory containing climate files
        suffixes: List of file suffixes to merge

    Returns:
        Merged DataFrame with all climate variables
    """
    dfs = []

    for suffix in suffixes:
        # Construct filename
        filename = f"HSA_{hsa_name}_{suffix}"
        filepath = climate_dir / filename
        if network:
            network_filename = f"{network}_HSA_{hsa_name}_{suffix}"
            network_path = climate_dir / network_filename
            if network_path.exists():
                filename = network_filename
                filepath = network_path

        if not filepath.exists():
            print(f"  [!]  Missing: {filename}")
            continue

        # Load file
        df = load_climate_file(filepath)

        # Keep only unique columns (avoid duplicating FacilityName, week_start)
        if len(dfs) > 0:
            # Keep only new columns (excluding FacilityName and week_start)
            existing_cols = set()
            for existing_df in dfs:
                existing_cols.update(existing_df.columns)

            cols_to_keep = ['FacilityName', 'week_start']
            cols_to_keep += [col for col in df.columns if col not in existing_cols]
            df = df[cols_to_keep]

        dfs.append(df)

    # Merge all dataframes
    if len(dfs) == 0:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(
            df,
            on=['FacilityName', 'week_start'],
            how='outer'
        )

    return merged


def get_all_hsa_names(climate_dir, network=None):
    """Extract unique HSA names from climate filenames"""
    hsa_names = set()

    pattern = "HSA_*_precip_lags.csv"
    if network:
        pattern = f"{network}_HSA_*_precip_lags.csv"

    for filepath in climate_dir.glob(pattern):
        # Extract HSA name from filename
        filename = filepath.stem  # Remove .csv
        # Remove optional "{network}_" prefix, then "HSA_" prefix and suffix
        if network and filename.startswith(f"{network}_HSA_"):
            filename = filename.replace(f"{network}_HSA_", "")
        else:
            filename = filename.replace("HSA_", "")
        hsa_name = filename.replace("_precip_lags", "")
        hsa_names.add(hsa_name)

    # Fallback to non-networked names if none found
    if not hsa_names and network:
        for filepath in climate_dir.glob("HSA_*_precip_lags.csv"):
            filename = filepath.stem
            hsa_name = filename.replace("HSA_", "").replace("_precip_lags", "")
            hsa_names.add(hsa_name)

    return sorted(list(hsa_names))


def create_temporal_features(df):
    """Create temporal features from week_start"""
    df = df.copy()

    df['week_of_year'] = df['week_start'].dt.isocalendar().week
    df['month'] = df['week_start'].dt.month
    df['quarter'] = df['week_start'].dt.quarter

    # Season (Northern Hemisphere)
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })

    # Days since start of study
    min_date = df['week_start'].min()
    df['days_since_start'] = (df['week_start'] - min_date).dt.days

    # Week number (sequential, starting from 1)
    df = df.sort_values(['hsa_id', 'week_start'])
    df['week_number'] = df.groupby('hsa_id').cumcount() + 1

    return df


def create_interaction_features(df):
    """Create interaction and derived climate features"""
    df = df.copy()

    # Temperature × Precipitation interaction (if both exist)
    if 'T_mean_week_C' in df.columns and 'P_total_week' in df.columns:
        df['temp_precip_interaction'] = df['T_mean_week_C'] * df['P_total_week']

    # Extreme heat indicator
    if 'T_max_week_C' in df.columns:
        df['extreme_heat'] = (df['T_max_week_C'] > 35).astype(int)

    # Heavy rain indicator
    if 'P_total_week' in df.columns:
        df['heavy_rain'] = (df['P_total_week'] > 10).astype(int)

    # Heat-moisture stress (if heat index and wetday fraction exist)
    if 'heat_index_week_C' in df.columns and 'wetday_frac_week' in df.columns:
        df['heat_moisture_stress'] = df['heat_index_week_C'] * (1 - df['wetday_frac_week'])

    # Cumulative precipitation (last 4 weeks)
    if all(f'P_sum_lag_w-{i}' in df.columns for i in [1, 2, 3]):
        df['cumulative_precip_4weeks'] = (
            df.get('P_total_week', 0) +
            df['P_sum_lag_w-1'] +
            df['P_sum_lag_w-2'] +
            df['P_sum_lag_w-3']
        )

    return df


def feature_selection_by_importance(df, target_col=None,
                                    max_features=40):
    """
    Select top features by correlation with target

    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        max_features: Maximum number of features to keep

    Returns:
        List of selected feature names
    """
    if target_col is None:
        target_col = TARGET_COL
    # Separate features and target
    base_count_col = f"{DISEASE_FOCUS}_count"
    feature_cols = [col for col in df.columns if col not in [
        'hsa_id', 'week_start', 'week_start_iso', 'FacilityName',
        target_col, base_count_col, 'week_number', 'days_since_start',
        'week_of_year', 'month', 'quarter', 'season', '.geo', 'system:index'
    ]]

    # Compute correlations
    correlations = df[feature_cols].corrwith(df[target_col]).abs()

    # Sort by correlation
    top_features = correlations.nlargest(max_features).index.tolist()

    return top_features


def remove_highly_correlated_features(df, features, threshold=0.95):
    """
    Remove features that are highly correlated with each other

    Args:
        df: DataFrame
        features: List of feature column names
        threshold: Correlation threshold above which to remove features

    Returns:
        List of features after removing highly correlated ones
    """
    # Compute correlation matrix
    corr_matrix = df[features].corr().abs()

    # Upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = [column for column in upper_triangle.columns
               if any(upper_triangle[column] > threshold)]

    # Keep features not in to_drop list
    selected_features = [f for f in features if f not in to_drop]

    return selected_features


def split_temporal_data(df, train_end_week, val_end_week):
    """
    Split data temporally into train/validation/test sets

    Args:
        df: Complete DataFrame
        train_end_week: Last week number for training
        val_end_week: Last week number for validation

    Returns:
        train_df, val_df, test_df
    """
    train_df = df[df['week_number'] <= train_end_week].copy()
    val_df = df[(df['week_number'] > train_end_week) &
                (df['week_number'] <= val_end_week)].copy()
    test_df = df[df['week_number'] > val_end_week].copy()

    return train_df, val_df, test_df


def create_metadata(df, selected_features, output_path):
    """
    Create metadata JSON file documenting the dataset

    Args:
        df: Complete DataFrame
        selected_features: List of selected feature names
        output_path: Path to save metadata JSON
    """
    def _as_series(frame, col_name):
        col = frame[col_name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col

    week_number_col = _as_series(df, 'week_number')
    week_number_max = int(week_number_col.max())
    target_col = _as_series(df, TARGET_COL)
    metadata = {
        'dataset_info': {
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': int(len(df)),
            'total_features': int(len(selected_features)),
            'date_range': {
                'start': df['week_start'].min().strftime('%Y-%m-%d'),
                'end': df['week_start'].max().strftime('%Y-%m-%d'),
                'weeks': week_number_max
            },
            'hsas': {
                'count': int(df['hsa_id'].nunique()),
                'names': sorted(df['hsa_id'].unique().tolist())
            }
        },
        'target_variable': {
            'name': TARGET_COL,
            'description': f'Weekly adjusted count of {DISEASE_FOCUS} cases',
            'statistics': {
                'mean': float(target_col.mean()),
                'std': float(target_col.std()),
                'min': float(target_col.min()),
                'max': float(target_col.max()),
                'median': float(target_col.median()),
                'zeros': int((target_col == 0).sum().item()),
                'zero_percentage': float((target_col == 0).mean() * 100)
            }
        },
        'features': {},
        'note': 'Train/validation/test splitting should be done during modeling phase'
    }

    # Add feature metadata
    for feat in selected_features:
        if feat in df.columns:
            feat_col = _as_series(df, feat)
            if not pd.api.types.is_numeric_dtype(feat_col):
                # Skip non-numeric features like season labels.
                continue
            metadata['features'][feat] = {
                'mean': float(feat_col.mean()),
                'std': float(feat_col.std()),
                'min': float(feat_col.min()),
                'max': float(feat_col.max()),
                'missing_count': int(feat_col.isna().sum().item() if hasattr(feat_col.isna().sum(), 'item') else feat_col.isna().sum()),
                'missing_percentage': float(feat_col.isna().mean() * 100)
            }

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Metadata saved to {output_path}")

    return metadata


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def main():
    """Main execution function"""
    global NETWORK, HSA_MODE, DISEASE_FOCUS, TARGET_COL, DIAGNOSIS_FILE
    global CLIMATE_DIR, OUTPUT_DIR, MISSING_THRESHOLD, CORRELATION_THRESHOLD, START_DATE, END_DATE
    args = parse_args()
    if not args.start_date or not args.end_date:
        print("ERROR: --start-date and --end-date are required (set in notebook or via START_DATE/END_DATE env vars)",
              file=sys.stderr)
        sys.exit(1)

    # Set all globals from arguments
    NETWORK = args.network
    HSA_MODE = args.hsa_mode
    if not args.disease_focus:
        print("ERROR: --disease-focus is required (no hardcoded default); pass the "
              "group label (e.g. 'Diarrheal Diseases') or a keyword.", file=sys.stderr)
        sys.exit(1)
    # Resolve to the canonical group slug; column names are derived, never hardcoded.
    DISEASE_FOCUS = slug(canonical_group(NETWORK, args.disease_focus))
    TARGET_COL = f"{DISEASE_FOCUS}_count_adjusted"
    # Derive from --out-dir whenever the caller did not name a directory outright,
    # so an isolated run reads and writes entirely inside its own directory.
    # This previously keyed off boundary_version, so --out-dir had no effect at
    # the default version and an isolated run silently read out/.
    if not args.climate_dir:
        args.climate_dir = str(Path(args.out_dir)
                               / f"DRIVE_CLIMATE_BY_HSA_DOWNLOAD_{args.boundary_version.upper()}"
                               / "FINAL_HSA_CLIMATE")
    if not args.output_dir:
        args.output_dir = str(Path(args.out_dir) / "modeling")
    CLIMATE_DIR = Path(args.climate_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    MISSING_THRESHOLD = args.missing_threshold
    CORRELATION_THRESHOLD = args.correlation_threshold
    START_DATE = args.start_date
    END_DATE = args.end_date

    diagnosis_root = Path(args.out_dir)
    DIAGNOSIS_FILE = diagnosis_root / f"{NETWORK}_{HSA_MODE}_weekly_{DISEASE_FOCUS}_adjusted_{args.boundary_version}.csv"
    if not DIAGNOSIS_FILE.exists():
        fallback = diagnosis_root / f"{NETWORK}_{HSA_MODE}_weekly_{DISEASE_FOCUS}.csv"
        if fallback.exists():
            DIAGNOSIS_FILE = fallback
            TARGET_COL = f"{DISEASE_FOCUS}_count"
            print(f"[!] Adjusted diagnosis file missing; using unadjusted: {DIAGNOSIS_FILE}")

    print("="*80)
    print("ML DATASET PREPARATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Climate directory: {CLIMATE_DIR}")
    print(f"Network: {NETWORK}")
    print(f"HSA mode: {HSA_MODE}")
    print(f"Disease focus: {DISEASE_FOCUS}")
    print(f"Diagnosis file: {DIAGNOSIS_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

    # -------------------------------------------------------------------------
    # STEP 1: Get all HSA names
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Identifying HSAs...")
    hsa_names = get_all_hsa_names(CLIMATE_DIR, NETWORK)
    print(f"  Found {len(hsa_names)} HSAs")
    for i, name in enumerate(hsa_names, 1):
        print(f"    {i:2d}. {name}")

    # The HSA list above comes from whatever climate files sit in CLIMATE_DIR,
    # and that directory is keyed by network and boundary version but not by HSA
    # mode. Exports from a second mode, or files left behind by a superseded
    # delineation, therefore become extra HSAs in the modeling dataset without
    # any error. Reconcile against the delineation this run is supposed to use.
    geojson_path = Path(args.out_dir) / f"{NETWORK}_{HSA_MODE}_hsas_{args.boundary_version}.geojson"
    if geojson_path.exists():
        import geopandas as gpd
        anchors = gpd.read_file(geojson_path)
        name_col = "FacilityName" if "FacilityName" in anchors.columns else "HealthFacility"
        expected = {clean_facility_name(str(n)) for n in anchors[name_col]}
        found    = {clean_facility_name(n) for n in hsa_names}
        stale    = sorted(found - expected)
        missing  = sorted(expected - found)
        if stale:
            print(f"\n  ERROR: {len(stale)} anchor(s) have climate files but are not in")
            print(f"  {geojson_path.name} ({len(expected)} anchors):")
            for x in stale:
                print(f"    - {x}")
            print("  This directory mixes runs, or holds files from a superseded")
            print("  delineation. Re-export into an isolated HSA_OUT_DIR, or clear it.")
            sys.exit(1)
        if missing:
            print(f"\n  ERROR: {len(missing)} anchor(s) in {geojson_path.name} have no climate files:")
            for x in missing:
                print(f"    - {x}")
            print(f"  Re-run the weekly GEE export for {NETWORK}-{HSA_MODE}.")
            sys.exit(1)
        print(f"  Anchor check: matches {geojson_path.name} exactly ({len(expected)} anchors)")
    else:
        print(f"  WARNING: {geojson_path} not found; skipping the anchor cross-check.")

    # -------------------------------------------------------------------------
    # STEP 2: Merge climate files for each HSA
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Merging climate files for each HSA...")
    all_hsa_data = []

    for i, hsa_name in enumerate(hsa_names, 1):
        print(f"\n  [{i}/{len(hsa_names)}] Processing: {hsa_name}")

        hsa_df = merge_climate_files_for_hsa(hsa_name, CLIMATE_DIR, CLIMATE_SUFFIXES, NETWORK)

        if hsa_df is None:
            print(f"    [X] Failed to merge files for {hsa_name}")
            continue

        # Add HSA identifier (normalized)
        hsa_df['hsa_id'] = clean_facility_name(hsa_name)

        all_hsa_data.append(hsa_df)
        print(f"    [OK] Merged {len(hsa_df)} weeks, {len(hsa_df.columns)} features")

    # Concatenate all HSAs
    if len(all_hsa_data) == 0:
        print("\n[X] ERROR: No HSA data could be merged!")
        return

    climate_df = pd.concat(all_hsa_data, ignore_index=True)
    print(f"\n  [OK] Combined climate data: {len(climate_df)} records, {len(climate_df.columns)} columns")

    # -------------------------------------------------------------------------
    # STEP 3: Load diagnosis data and filter to valid date range
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Loading diagnosis data...")

    if not DIAGNOSIS_FILE.exists():
        print(f"  [X] ERROR: Diagnosis file not found: {DIAGNOSIS_FILE}")
        return

    diagnosis_df = pd.read_csv(DIAGNOSIS_FILE)
    diagnosis_df['week_start'] = pd.to_datetime(diagnosis_df['week_start'])

    # Clean HSA names for matching
    if 'hsa_id' in diagnosis_df.columns:
        diagnosis_df['hsa_id'] = diagnosis_df['hsa_id'].apply(clean_facility_name)

    print(f"  [OK] Loaded {len(diagnosis_df)} diagnosis records (all dates)")
    print(f"    Full date range: {diagnosis_df['week_start'].min()} to {diagnosis_df['week_start'].max()}")
    print(f"    Full HSA count: {diagnosis_df['hsa_id'].nunique()}")

    # Filter to valid date range (post diagnostic code change)
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)

    diagnosis_df = diagnosis_df[
        (diagnosis_df['week_start'] >= start_date) &
        (diagnosis_df['week_start'] <= end_date)
    ]

    print(f"\n  Filtered to valid date range ({START_DATE} to {END_DATE}):")
    print(f"    Records: {len(diagnosis_df)}")
    print(f"    HSAs: {diagnosis_df['hsa_id'].nunique()}")
    print(f"    Date range: {diagnosis_df['week_start'].min()} to {diagnosis_df['week_start'].max()}")

    # Check for HSA count mismatch with climate data
    diagnosis_hsas = set(diagnosis_df['hsa_id'].unique())
    climate_hsas = set(hsa_names)

    if len(diagnosis_hsas) != len(climate_hsas):
        print(f"\n  [!] WARNING: HSA count mismatch!")
        print(f"    Climate data: {len(climate_hsas)} HSAs")
        print(f"    Diagnosis data: {len(diagnosis_hsas)} HSAs")

        in_diagnosis_not_climate = diagnosis_hsas - climate_hsas
        in_climate_not_diagnosis = climate_hsas - diagnosis_hsas

        if in_diagnosis_not_climate:
            print(f"    In diagnosis but not climate: {in_diagnosis_not_climate}")
        if in_climate_not_diagnosis:
            print(f"    In climate but not diagnosis: {in_climate_not_diagnosis}")

    # -------------------------------------------------------------------------
    # STEP 4: Merge climate and diagnosis data
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Merging climate and diagnosis data...")

    # Merge on hsa_id and week_start
    merged_df = climate_df.merge(
        diagnosis_df[['hsa_id', 'week_start', TARGET_COL]],
        on=['hsa_id', 'week_start'],
        how='left'
    )

    # Fill missing target counts with 0 (weeks with no cases)
    merged_df[TARGET_COL] = merged_df[TARGET_COL].fillna(0)

    # Fallback to non-adjusted count if adjusted not present
    base_count_col = f"{DISEASE_FOCUS}_count"
    if TARGET_COL not in merged_df.columns and base_count_col in merged_df.columns:
        merged_df[TARGET_COL] = merged_df[base_count_col]

    print(f"  [OK] Merged dataset: {len(merged_df)} records")
    print(f"    Features: {len(merged_df.columns)} columns")
    print(f"    Missing diagnosis data: {merged_df[TARGET_COL].isna().sum()} records")

    # -------------------------------------------------------------------------
    # STEP 5: Data cleaning - PRIORITIZE KEEPING HSAs
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Cleaning data...")
    print(f"  Strategy: Keep all HSAs, drop variables with excessive missing data")
    print(f"  Note: Zeros are valid values in arid regions (not treated as missing)")
    print(f"  Note: Imputation will be done during modeling phase to avoid data leakage")

    # Remove rows with missing target
    initial_len = len(merged_df)
    merged_df = merged_df.dropna(subset=[TARGET_COL])
    print(f"\n  [OK] Removed {initial_len - len(merged_df)} rows with missing target")

    # Check for missing climate features (NaN/null only, not zeros!)
    climate_cols = [col for col in merged_df.columns if col not in [
        'hsa_id', 'FacilityName', 'week_start', 'week_start_iso',
        f"{DISEASE_FOCUS}_count", TARGET_COL, '.geo', 'system:index'
    ]]

    missing_summary = merged_df[climate_cols].isna().sum()
    missing_features = missing_summary[missing_summary > 0].sort_values(ascending=False)

    if len(missing_features) > 0:
        print(f"\n  [!] Features with missing (NaN) values ({len(missing_features)} features):")

        # Use global threshold for dropping variables
        threshold_count = int(len(merged_df) * MISSING_THRESHOLD)

        cols_to_drop = []
        cols_with_missing = []

        for feat, count in missing_features.items():
            pct = (count / len(merged_df)) * 100
            if count > threshold_count:
                cols_to_drop.append(feat)
                print(f"    DROP: {feat}: {count} ({pct:.1f}%) - exceeds {MISSING_THRESHOLD*100:.0f}% threshold")
            else:
                cols_with_missing.append(feat)
                print(f"    KEEP: {feat}: {count} ({pct:.1f}%) - retain for modeling phase imputation")

        # Drop variables with >40% missing
        if cols_to_drop:
            print(f"\n  Dropping {len(cols_to_drop)} variables with >{MISSING_THRESHOLD*100:.0f}% missing data:")
            for col in cols_to_drop:
                print(f"    - {col}")
            merged_df = merged_df.drop(columns=cols_to_drop)
            climate_cols = [c for c in climate_cols if c not in cols_to_drop]

        # Report remaining missing values (will be handled during modeling)
        if cols_with_missing:
            print(f"\n  Retaining {len(cols_with_missing)} variables with <{MISSING_THRESHOLD*100:.0f}% missing:")
            print(f"    These will be imputed during modeling phase (using train set only)")
            print(f"    This prevents data leakage from test set into training")

        print(f"\n  [OK] All HSAs retained: {merged_df['hsa_id'].nunique()} HSAs, {len(merged_df)} records")
        print(f"  [OK] Missing values preserved for proper modeling-phase imputation")
    else:
        print(f"  [OK] No missing climate data")

    # -------------------------------------------------------------------------
    # STEP 6: Feature engineering
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Creating engineered features...")

    # Temporal features
    merged_df = create_temporal_features(merged_df)
    print(f"  [OK] Added temporal features: week_of_year, month, season, week_number")

    # Interaction features
    merged_df = create_interaction_features(merged_df)
    print(f"  [OK] Added interaction features")

    # -------------------------------------------------------------------------
    # STEP 7: Feature selection (deferred to modeling after temporal split)
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Deferring feature selection...")

    # Get all potential features
    exclude_cols = ['hsa_id', 'FacilityName', 'week_start', 'week_start_iso',
                    f"{DISEASE_FOCUS}_count", TARGET_COL, '.geo', 'system:index']
    all_features = [col for col in merged_df.columns if col not in exclude_cols]

    print(f"  Total available features: {len(all_features)}")
    print("  [OK] Feature selection skipped to avoid leakage (do this after temporal split)")

    selected_features = all_features

    # -------------------------------------------------------------------------
    # STEP 8: Create final dataset
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Creating final dataset...")

    # Select columns for final dataset
    final_cols = ['hsa_id', 'week_start', 'week_number', 'week_of_year',
                  'month', 'season', TARGET_COL] + selected_features

    final_df = merged_df[final_cols].copy()

    # Sort by HSA and week
    final_df = final_df.sort_values(['hsa_id', 'week_start'])

    print(f"  [OK] Final dataset shape: {final_df.shape}")
    print(f"    Records: {len(final_df)}")
    print(f"    Features: {len(selected_features)}")
    print(f"    HSAs: {final_df['hsa_id'].nunique()}")
    print(f"    Weeks: {final_df['week_number'].max()}")

    # -------------------------------------------------------------------------
    # STEP 9: Save complete dataset (no splitting - done at modeling phase)
    # -------------------------------------------------------------------------
    print("\n[STEP 9] Saving complete dataset...")
    print("  Note: Train/validation/test splitting will be done during modeling phase")

    # Save full dataset
    output_path = OUTPUT_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{args.boundary_version}.csv"
    final_df.to_csv(output_path, index=False)
    print(f"  [OK] Saved: {output_path}")

    # -------------------------------------------------------------------------
    # STEP 10: Create metadata
    # -------------------------------------------------------------------------
    print("\n[STEP 10] Creating metadata...")

    metadata_path = OUTPUT_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{args.boundary_version}_metadata.json"
    metadata = create_metadata(final_df, selected_features, metadata_path)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"\nDataset Summary:")
    print(f"  Total records: {len(final_df)}")
    print(f"  HSAs: {final_df['hsa_id'].nunique()}")
    print(f"  Weeks: {final_df['week_number'].max()}")
    print(f"  Features: {len(selected_features)}")
    print(f"  Date range: {final_df['week_start'].min()} to {final_df['week_start'].max()}")

    print(f"\nTarget Variable ({TARGET_COL}):")
    print(f"  Mean: {final_df[TARGET_COL].mean():.2f}")
    print(f"  Std: {final_df[TARGET_COL].std():.2f}")
    print(f"  Min: {final_df[TARGET_COL].min():.0f}")
    print(f"  Max: {final_df[TARGET_COL].max():.0f}")
    print(f"  Median: {final_df[TARGET_COL].median():.1f}")
    print(f"  Zero counts: {(final_df[TARGET_COL] == 0).sum()} ({(final_df[TARGET_COL] == 0).mean()*100:.1f}%)")

    print(f"\nHSA Coverage:")
    hsa_list = sorted(final_df['hsa_id'].unique())
    print(f"  Retained HSAs: {len(hsa_list)}")
    for i, hsa in enumerate(hsa_list, 1):
        records = len(final_df[final_df['hsa_id'] == hsa])
        print(f"    {i:2d}. {hsa}: {records} weeks")

    # Check for remaining missing values
    feature_cols = [col for col in selected_features if col in final_df.columns]
    missing_counts = final_df[feature_cols].isna().sum()
    features_with_missing = missing_counts[missing_counts > 0]

    if len(features_with_missing) > 0:
        print(f"\nMissing Values Summary:")
        print(f"  Features with missing values: {len(features_with_missing)}")
        print(f"  Note: These will be imputed during modeling phase to prevent data leakage")
        total_missing = missing_counts.sum()
        total_cells = len(final_df) * len(feature_cols)
        pct_missing = (total_missing / total_cells) * 100
        print(f"  Total missing: {total_missing:,} / {total_cells:,} cells ({pct_missing:.2f}%)")

    print(f"\nOutput Files:")
    print(f"  {output_path}")
    print(f"  {metadata_path}")
    print("="*80)


if __name__ == "__main__":
    main()
