#!/usr/bin/env python3
"""Rebuild the S6.2 connectivity comparison from one final HSA run.

The comparison is descriptive: allocation certainty is not a measure of road
access or urbanicity.  Inputs are the final pixel-to-facility allocation,
facility-to-HSA assignment, and the already split weekly modeling tables.
"""

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def allocation_by_hsa(pixel_path, assignments_path, geojson_path, facilities_path):
    assignments = pd.read_csv(assignments_path)
    excluded = set(assignments.loc[assignments.excluded.astype(bool), "facility_id"])
    assignments = assignments.loc[~assignments.excluded.astype(bool)]
    anchor_features = json.loads(Path(geojson_path).read_text())["features"]
    anchors = {key(f["properties"]["HealthFacility"]): f["properties"] for f in anchor_features}
    facilities = pd.read_csv(facilities_path).set_index("healthfacility")
    facility_sums = {}
    for chunk in pd.read_csv(pixel_path, usecols=["facility_id", "population", "probability"], chunksize=250000):
        chunk = chunk.loc[~chunk.facility_id.isin(excluded)].copy()
        chunk["probability_population"] = chunk.probability * chunk.population
        grouped = chunk.groupby("facility_id")[["population", "probability_population"]].sum()
        for name, row in grouped.iterrows():
            pair = facility_sums.setdefault(name, [0.0, 0.0])
            pair[0] += row.population
            pair[1] += row.probability_population
    if not set(facility_sums).issubset(set(assignments.facility_id)):
        raise ValueError("Top-choice pixel facilities include a facility without a final assignment")
    sums = {}
    for r in assignments.itertuples():
        if r.facility_id not in facility_sums:
            continue  # A facility may receive fractional allocation without ever being top choice.
        pop, prob_pop = facility_sums[r.facility_id]
        if str(r.assignment_case).startswith("Case 3:"):
            candidates = [part.rsplit(":", 1)[0] for part in str(r.all_containing_hsas).split("; ")]
            fac = facilities.loc[r.facility_id]
            weights = []
            for name in candidates:
                anchor = anchors[key(name)]
                distance = max(haversine(fac.Longitude, fac.Latitude, anchor["lon"], anchor["lat"]), .01)
                weights.append((anchor["Total"] ** .75) / (distance ** 1.5))
            shares = np.array(weights) / sum(weights)
        else:
            candidates = [r.primary_hsa]
            shares = [1.0]
        for name, share in zip(candidates, shares):
            pair = sums.setdefault(key(name), [0.0, 0.0, name])
            pair[0] += pop * share
            pair[1] += prob_pop * share
    frame = pd.DataFrame(
        [(k, v[2], v[0], v[1] / v[0]) for k, v in sums.items()],
        columns=["hsa_key", "hsa_name", "top_choice_pixel_population", "mean_allocation_probability"],
    )
    if len(frame) != len(anchors):
        raise ValueError(f"Expected {len(anchors)} final HSAs, found {len(frame)}")
    if len(frame) < 4:
        raise ValueError(f"Need at least 4 HSAs to split into two groups, found {len(frame)}")
    median = frame.mean_allocation_probability.median()
    # Split at the median by rank rather than by value: a tie exactly at the
    # median would otherwise push both tied areas into the upper group and
    # leave the halves uneven. With distinct values the two are identical.
    ordered = frame.sort_values("mean_allocation_probability", kind="mergesort")
    lower = ordered.index[: len(ordered) // 2]
    frame["group"] = "higher_certainty"
    frame.loc[lower, "group"] = "lower_certainty"
    return frame.sort_values("mean_allocation_probability"), median


def _diagnosis_pivot(source: Path, network: str, mode: str) -> Path:
    """Locate the diagnosis-count pivot for a run directory.

    Counts are generated once per network, under the footprint label, and are
    shared by every optimization mode; only some runs also write a mode-specific
    copy. Resolve the same way the delineation notebook does, most specific
    first, so a non-footprint run directory still finds the network's counts.
    """
    candidates = [source / f"{network}_{mode}_diagnosis_counts_pivot.csv",
                  source / f"{network}_diagnosis_counts_pivot.csv",
                  source / f"{network}_footprint_diagnosis_counts_pivot.csv"]
    return next((c for c in candidates if c.exists()), candidates[0])


def run(args):
    source = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.network}_{args.mode}"
    model_dir = source / "modeling" / f"results_comprehensive_{args.boundary_version}"
    paths = {
        "pixels": source / f"pixel_allocations_{stem}_{args.boundary_version}.csv",
        "assignments": source / f"{stem}_facility_hsa_assignments_{args.boundary_version}.csv",
        "hsa_geojson": source / f"{stem}_hsas_{args.boundary_version}.geojson",
        "facility_locations": _diagnosis_pivot(source, args.network, args.mode),
        "feature_selection": model_dir / f"{stem}_feature_selection_info.json",
        **{split: model_dir / f"{stem}_{split}_data.csv" for split in ("train", "val", "test")},
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    groups, median = allocation_by_hsa(paths["pixels"], paths["assignments"],
                                       paths["hsa_geojson"], paths["facility_locations"])
    groups.to_csv(output / f"{stem}_connectivity_hsa_groups.csv", index=False)
    splits = {name: pd.read_csv(paths[name]) for name in ("train", "val", "test")}
    outcome = [c for c in splits["train"] if c.endswith("_count_adjusted")]
    if len(outcome) != 1:
        raise ValueError(f"Expected one adjusted disease outcome, found {outcome}")
    outcome = outcome[0]
    baseline = ["ar_lag1", "ar_lag2", "ar_lag3", "ar_lag4", "week_of_year", "month", "quarter"]
    climate = ["T_mean_week_C", "P_total_week"]
    selected = json.loads(paths["feature_selection"].read_text())["selected_features"]
    time_names = {"week_number.1", "week_of_year.1", "month.1", "quarter", "days_since_start", "hsa_encoded"}
    sensitivity_climate = [c for c in selected if c not in time_names][:20]
    if len(sensitivity_climate) != 20:
        raise ValueError("Need 20 selected weather features for the sensitivity check")
    for name, frame in splits.items():
        needed = ["hsa_id", "week_start", outcome] + baseline + climate + sensitivity_climate
        absent = set(needed) - set(frame.columns)
        if absent:
            raise ValueError(f"{name}: missing columns {sorted(absent)}")
        frame["hsa_key"] = frame.hsa_id.map(key)
        frame = frame.merge(groups[["hsa_key", "group"]], on="hsa_key", how="left", validate="many_to_one")
        if frame.group.isna().any() or frame.hsa_key.nunique() != len(groups):
            raise ValueError(f"{name}: model HSAs did not match all final allocation HSAs")
        if frame[needed].isna().any().any():
            raise ValueError(f"{name}: incomplete model data; refusing imputation")
        splits[name] = frame
    dates = {name: set(frame.week_start) for name, frame in splits.items()}
    if dates["train"] & dates["val"] or dates["train"] & dates["test"] or dates["val"] & dates["test"]:
        raise ValueError("Temporal splits share dates")
    rows = []
    for group in ("higher_certainty", "lower_certainty"):
        train = splits["train"].loc[splits["train"].group == group]
        test = splits["test"].loc[splits["test"].group == group]
        results = {}
        for model_name, features in (("AR+seasonal", baseline), ("AR+seasonal+climate", baseline + climate)):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(train[features], train[outcome])
            results[model_name] = float(r2_score(test[outcome], model.predict(test[features])))
        sensitivity_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        sensitivity_model.fit(train[baseline + sensitivity_climate], train[outcome])
        sensitivity_r2 = float(r2_score(test[outcome], sensitivity_model.predict(test[baseline + sensitivity_climate])))
        rows.append({"group": group, "n_hsas": int(train.hsa_key.nunique()),
                     "n_train": len(train), "n_test": len(test),
                     "ar_seasonal_test_r2": results["AR+seasonal"],
                     "ar_seasonal_climate_test_r2": results["AR+seasonal+climate"],
                     "climate_delta_r2": results["AR+seasonal+climate"] - results["AR+seasonal"],
                     "sensitivity_20_weather_test_r2": sensitivity_r2,
                     "sensitivity_20_weather_delta_r2": sensitivity_r2 - results["AR+seasonal"]})
    result = {
        "analysis_date": date.today().isoformat(),
        "source_run": str(source), "source_sha256": {name: sha256(p) for name, p in paths.items()},
        "grouping": "Final HSA mean probability of each pixel's top-choice facility, weighted by that pixel's population; overlap facilities carried proportionally to containing HSAs under the production gravity formula; median split. This is a top-choice proxy, not the full fractional-allocation denominator, urban/rural status, or physical connectivity.",
        "median_mean_allocation_probability": float(median),
        "model": "Ridge(alpha=1), training-only StandardScaler; existing time-disjoint comprehensive-model splits",
        "outcome": outcome, "baseline_features": baseline, "climate_features": climate,
        "sensitivity_weather_features": sensitivity_climate,
        "date_ranges": {name: [min(d), max(d)] for name, d in dates.items()},
        "rows": rows,
    }
    (output / f"{stem}_connectivity_results.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(output / f"{stem}_connectivity_results.csv", index=False)
    print(json.dumps({"output": str(output), "rows": rows}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="INF")
    parser.add_argument("--mode", default="footprint")
    parser.add_argument("--boundary-version", default="v7")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
