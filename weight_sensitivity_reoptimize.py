#!/usr/bin/env python3
"""
Multi-objective weight sensitivity by re-running the delineation.

Each perturbation re-runs the full corrected selection -- greedy selection,
preliminary overlap removal, guarded anchor promotion/demotion, major-facility
promotion, symmetric final overlap removal, coverage repair -- with one
objective weight scaled, and measures what comes out.

This replaces an earlier approach that derived the perturbed values
arithmetically from the baseline (n = n_base * (1 +/- k(level-1)) and
temp_diversity = baseline * (1 + 0.2(level-1))) without ever re-optimizing.
Those were closed-form extrapolations, not measurements, and the temperature
figure fell back to a hardcoded default whenever the climate lookup came back
empty.

Reported per configuration, all measured from the delineation the run produced:
  n_anchors        number of HSA anchors selected
  coverage_pct     national population covered, on the WorldPop raster
  temp_diversity   standard deviation of mean annual temperature across anchors
  climate_regimes  distinct climate clusters represented among anchors
  median_radius_km median service radius

Note that a mode's weight profile can set a term to zero, in which case scaling
it is a no-op by construction. FOOTPRINT sets patient_volume and distance to
zero, so those rows are expected to be flat; that is a property of the mode, not
an absence of sensitivity, and it is reported rather than hidden.

Usage:
    python weight_sensitivity_reoptimize.py --network INF --hsa-mode footprint \
        --out-dir out_INF_footprint_v7 --boundary-version v7
"""
from __future__ import annotations

import os as _os
import sys as _sys

# PYTHONHASHSEED must be set before the interpreter starts, so re-exec once if
# the caller has not set it. The greedy selection breaks ties through
# hash-ordered structures, which makes the delineation non-deterministic without
# it: run_pipeline.py pins it to 42 for every notebook kernel for this reason.
# Leaving it unset here made a perturbation's effect indistinguishable from
# hash noise, which is the whole point of the analysis.
_SEED = _os.environ.get("HSA_HASH_SEED", "42")
if _os.environ.get("PYTHONHASHSEED") != _SEED:
    _os.environ["PYTHONHASHSEED"] = _SEED
    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

import argparse
import copy
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import random

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

# run_pipeline.py seeds each notebook kernel with random.seed and np.random.seed
# in addition to PYTHONHASHSEED. Matching all three is what makes a delineation
# produced here comparable to one produced by the pipeline; with only the hash
# seed pinned, a re-run can return a different anchor set of the same size.
random.seed(int(_SEED))
np.random.seed(int(_SEED))
BASE_DIR = Path(__file__).resolve().parent

# Objective term in the manuscript -> key in MODE_WEIGHT_PROFILES
WEIGHT_ALIASES = {
    "population_coverage": "coverage",
    "climatic_diversity": "climate",
    "facility_volume": "patient_volume",
    "spatial_overlap": "overlap",
    "travel_distance": "distance",
}
LEVELS = [0.8, 0.9, 1.0, 1.1, 1.2]

BASE_PARAMS = dict(
    DENSITY_RADIUS_KM=10.0, URBAN_DENSITY_THRESH=1500.0,
    URBAN_BASE_RADIUS_KM=10.0, RURAL_BASE_RADIUS_KM=18.0,
    NETWORK_RADIUS_MULTIPLIERS={"INF": 1.0, "NCD": 1.0},
    CLIMATE_DIVERSITY_ON=True, CLIMATE_K=8, CLIMATE_MIN_PER_CLUSTER=0,
    CLIMATE_CLUSTER_COL="climate_k",
    WEIGHT_COVERAGE=8.0, WEIGHT_OVERLAP_PENALTY=2.0, WEIGHT_CLIMATE=1.0,
    WEIGHT_PATIENT_VOLUME=3.0, WEIGHT_COVERAGE_PROGRESS=1.0,
    WEIGHT_DISTANCE_PENALTY=3.0,
    JORDAN_BOUNDS=[34.5, 29.0, 39.5, 33.5], OVERLAP_REMOVAL_THRESHOLD=0.80,
)
PROFILES = {
    "fewest":    {"coverage": 15.0, "coverage_progress": 0.0, "overlap": 0.1,
                  "patient_volume": 0.0, "climate": 0.0, "distance": 0.0},
    "footprint": {"coverage": 0.8, "coverage_progress": 0.0, "overlap": 0.01,
                  "patient_volume": 0.0, "climate": 1.2, "distance": 0.0},
    "distance":  {"coverage": 0.5, "coverage_progress": 0.0, "overlap": 0.20,
                  "patient_volume": 0.0, "climate": 0.05, "distance": 8.0},
}


def load_facilities(out_dir: Path, network: str) -> gpd.GeoDataFrame:
    _cf = out_dir / f"{network}_Facilities_Climate_Features_with_clusters.csv"
    if not _cf.exists():
        for _alt in (Path("out") / _cf.name, Path("data") / _cf.name):  # noqa: hardcode - shared GEE Step A input
            if _alt.exists():
                _cf = _alt
                break
    clim = pd.read_csv(_cf)
    clim["HealthFacility"] = clim["FacilityName"].str.replace(r"\s+", " ", regex=True).str.strip()
    diag_path = next(
        (p for p in (out_dir / f"{network}_footprint_diagnosis_counts_pivot.csv",
                     out_dir / f"{network}_diagnosis_counts_pivot.csv") if p.exists()),
        None)
    if diag_path is None:
        sys.exit(f"ERROR: no diagnosis counts pivot in {out_dir}")
    diag = pd.read_csv(diag_path)
    diag["healthfacility"] = diag["healthfacility"].str.replace(r"\s+", " ", regex=True).str.strip()
    cols = ["healthfacility", "total_diagnoses"] + [
        c for c in ("healthfacilitytype", "governorate") if c in diag.columns]
    merged = clim.merge(diag[cols], left_on="HealthFacility",
                        right_on="healthfacility", how="left")
    merged["Total"] = merged["total_diagnoses"].fillna(0)
    return gpd.GeoDataFrame(
        merged, geometry=gpd.points_from_xy(merged.lon, merged.lat),
        crs="EPSG:4326").to_crs("EPSG:32637")


def run_one(H, HSAOptimizer, facilities, network, mode, profile, tau, pop_path):
    """One full delineation under `profile`; returns the selected anchors."""
    for k, v in BASE_PARAMS.items():
        setattr(H, k, v)
    setattr(H, "NETWORK", network)
    setattr(H, "TAU_COVERAGE", tau)
    profiles = copy.deepcopy(PROFILES)
    profiles[mode] = profile
    setattr(H, "MODE_WEIGHT_PROFILES", profiles)

    opt = HSAOptimizer({"pop_path": pop_path, "tau_coverage": tau, "coarsen": 4})
    # optimize() writes initial_radius_km back onto the frame it is handed, and
    # the coverage repair later sizes candidate service areas from that column.
    # Passing a copy here left the master without those radii, so the repair
    # fell back to the rural default and ranked candidates by the wrong
    # marginal population -- which is how a re-optimization of NCD picked
    # AL-Nadeem (252,802) where the pipeline had picked Arish (231,037).
    # The notebook passes one frame throughout; do the same.
    result = opt.optimize(facilities, objective=mode, network_type=network)
    sel = result["facilities"]

    guard = H.make_relocation_overlap_guard(opt, overlap_threshold=0.80)
    sel, _ = H.upgrade_selected_anchors_to_stronger_facilities(
        sel, facilities, search_radius_multiplier=1.0, min_volume_ratio=2.0,
        min_absolute_volume_gain=100.0, require_same_governorate=True,
        relocation_overlap_guard=guard, max_relocation_overlap=0.80)
    sel, _ = H.promote_major_uncovered_facilities(
        sel, facilities, major_pop_threshold=25000.0, major_volume_quantile=0.80,
        major_volume_threshold=None, fallback_radius_multiplier=1.5,
        fallback_min_distance_km=30.0, require_same_governorate=True)
    sel, _, stats = H.finalize_anchor_set(
        opt, sel, facilities, overlap_threshold=0.80, target_coverage=tau,
        protect_original_anchors=True, repair_coverage=True, max_additions=10,
        symmetric_overlap=True)
    return sel, stats


def measure(sel, facilities) -> dict:
    """Every field here is computed from the returned delineation."""
    names = set(sel["HealthFacility"].astype(str).str.strip())
    fac = facilities[facilities["HealthFacility"].astype(str).str.strip().isin(names)]
    temps = pd.to_numeric(fac.get("T_mean_C"), errors="coerce").dropna()
    regimes = pd.to_numeric(fac.get("climate_k"), errors="coerce").dropna().astype(int)
    radii = pd.to_numeric(sel["service_radius_km"], errors="coerce")
    names = sorted(str(n).strip() for n in sel["HealthFacility"])
    return {
        "n_anchors": int(len(sel)),
        "anchor_set_hash": hashlib.md5("|".join(names).encode()).hexdigest()[:10],
        "anchors": "; ".join(names),
        "temp_diversity": round(float(temps.std()), 4) if len(temps) > 1 else np.nan,
        "temp_range_C": round(float(temps.max() - temps.min()), 4) if len(temps) > 1 else np.nan,
        "climate_regimes": int(regimes.nunique()),
        "median_radius_km": round(float(radii.median()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default=os.environ.get("NETWORK", "INF"))
    ap.add_argument("--hsa-mode", default=os.environ.get("HSA_MODE", "footprint"))
    ap.add_argument("--out-dir", default=os.environ.get(
        "HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out")))
    ap.add_argument("--boundary-version", default=os.environ.get("BOUNDARY_VERSION", "v7"))
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    dest = Path(args.output_dir) if args.output_dir else (
        out_dir / f"weight_sensitivity_reoptimized_{args.boundary_version}")
    dest.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(BASE_DIR))
    import hsa_optimization as H
    from hsa_optimization import HSAOptimizer

    facilities = load_facilities(out_dir, args.network)
    pop_path = str(BASE_DIR / "data" / "jor_ppp_2020_UNadj.tif")
    
    mode = args.hsa_mode
    if mode not in PROFILES:
        print(f"Weight sensitivity is not defined for mode '{mode}' "
              f"(profiles exist for {sorted(PROFILES)}); skipping.")
        return
    base_profile = PROFILES[mode]

    print(f"Weight sensitivity by re-optimization: {args.network}-{mode} "
          f"(tau={args.tau:.2f}), {len(facilities)} candidate facilities")
    zero_terms = [k for k, v in base_profile.items() if v == 0.0]
    if zero_terms:
        print(f"  note: {mode} sets {', '.join(zero_terms)} to zero; scaling those "
              f"cannot change the objective")

    rows = []
    sel, stats = run_one(H, HSAOptimizer, facilities, args.network, mode,
                         copy.deepcopy(base_profile), args.tau, pop_path)
    base = {"perturbed_weight": "baseline", "level": 1.0,
            "coverage_pct": round(stats["coverage_pct_final"], 3), **measure(sel, facilities)}
    rows.append(base)
    print(f"  baseline: {base['n_anchors']} anchors, {base['coverage_pct']}% coverage, "
          f"temp_sd={base['temp_diversity']}, regimes={base['climate_regimes']}", flush=True)

    for label, key in WEIGHT_ALIASES.items():
        for level in LEVELS:
            if level == 1.0:
                continue
            prof = copy.deepcopy(base_profile)
            prof[key] = base_profile[key] * level
            sel, stats = run_one(H, HSAOptimizer, facilities, args.network, mode,
                                 prof, args.tau, pop_path)
            row = {"perturbed_weight": label, "level": level,
                   "weight_value": round(prof[key], 6),
                   "coverage_pct": round(stats["coverage_pct_final"], 3),
                   **measure(sel, facilities)}
            row["delta_n_anchors"] = row["n_anchors"] - base["n_anchors"]
            rows.append(row)
            print(f"  {label} x{level}: {row['n_anchors']} anchors "
                  f"({row['delta_n_anchors']:+d}), {row['coverage_pct']}% coverage, "
                  f"temp_sd={row['temp_diversity']}, regimes={row['climate_regimes']}",
                  flush=True)

    df = pd.DataFrame(rows)
    csv = dest / f"{args.network}_{mode}_weight_sensitivity_reoptimized.csv"
    df.to_csv(csv, index=False)

    meta = {
        "method": "full re-optimization per perturbation",
        "network": args.network, "hsa_mode": mode,
        "boundary_version": args.boundary_version, "tau_coverage": args.tau,
        "base_profile": base_profile,
        "zero_weight_terms": zero_terms,
        "levels": [l for l in LEVELS if l != 1.0],
        "n_delineations_run": len(rows),
        "population_raster": pop_path,
    }
    (dest / f"{args.network}_{mode}_weight_sensitivity_reoptimized.json").write_text(
        json.dumps(meta, indent=2))
    print(f"\nWrote {csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
