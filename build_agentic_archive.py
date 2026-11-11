#!/usr/bin/env python3
"""
build_agentic_archive.py -- package pipeline runs as agentic archives.

One archive per run (network x hsa_mode x boundary_version x disease). Each archive is
a zip with manifest.json at its ROOT and the run's curated outputs under out/:

    manifest.json            {"model": {...run identity + paths...}, "meta": {...}}
    SKILL_CONTEXT.md         short how-to-read note for the agent
    out/<NET>_<MODE>_hsas_<ver>.geojson
    out/modeling/...         model results (summary tables only)
    out/sensitivity/...      robustness analyses
    out/textresults/*.md     written summaries

Every path in manifest.json is relative to the archive root, so the agent's
zip_loader resolves files only through the manifest.

Usage (from the pipeline root):

    # all runs at once
    python build_agentic_archive.py out_SYNMOD*_v7 --dest agentic_archives

    # one run
    python build_agentic_archive.py out_SYNMODNCD_distance_v7 --dest agentic_archives

Run identity comes from the run folder name (out_<NETWORK>_<MODE>_<version>); the
disease comes from the run's modeling metadata (target variable), resolved through
disease_focus.py. Override with --disease if needed.

Summaries written by steps 1-2 (HSA_FINAL, Population_Allocation) live in the shared
out/ folder because those steps ran there; they are picked up from --shared-dir.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from disease_focus import canonical_group, daily_outcome_col, slug

SCHEMA_VERSION = "1.0"
ARCHIVE_ROOT = "out"                      # prefix of every data path inside the zip
ALLOWED_EXT = {".json", ".csv", ".md", ".geojson"}
MAX_FILE_KB = 1500                        # skip large raw dumps
MODES = ["footprint", "fewest", "distance", "governorate_tau_coverage",
         "governorate_fewest", "governorate"]
VERSIONS = ["v6", "v7", "v8"]
RUN_DIR_RE = re.compile(r"^out_(?P<network>[A-Za-z0-9]+)_(?P<mode>.+)_(?P<ver>v\d+)$")


class PackError(Exception):
    pass


# ----------------------------------------------------------------------- identity
def parse_run_dir(run_dir: Path) -> tuple[str, str, str]:
    m = RUN_DIR_RE.match(run_dir.name)
    if not m:
        raise PackError(f"{run_dir.name}: not a run folder (expected out_<NETWORK>_<MODE>_<version>)")
    network, mode, ver = m["network"], m["mode"], m["ver"]
    if mode not in MODES:
        raise PackError(f"{run_dir.name}: unknown HSA mode {mode!r}")
    return network, mode, ver


def detect_disease(run_dir: Path, network: str, mode: str, ver: str) -> str:
    """Disease keyword from the weekly modeling metadata's target variable."""
    meta = run_dir / "modeling" / f"{network}_{mode}_modeling_dataset_{ver}_metadata.json"
    if not meta.is_file():
        raise PackError(f"{run_dir.name}: cannot detect disease, {meta.name} not found "
                        f"(pass --disease)")
    target = json.loads(meta.read_text())["target_variable"]["name"]
    return re.sub(r"_count(_adjusted)?$", "", target)


# ----------------------------------------------------------------- file selection
def select_files(run_dir: Path, shared_dir: Path | None, network: str, mode: str,
                 ver: str) -> tuple[dict[str, Path], list[str], list[str]]:
    """Return ({archive_rel_path: source_path}, missing_expected, skipped_large)."""
    prefix = f"{network}_{mode}"
    picked: dict[str, Path] = {}
    missing: list[str] = []
    skipped: list[str] = []

    def add(src: Path, rel: str) -> None:
        if not src.is_file() or src.suffix.lower() not in ALLOWED_EXT:
            return
        if src.stat().st_size > MAX_FILE_KB * 1024:
            skipped.append(rel)
            return
        picked.setdefault(f"{ARCHIVE_ROOT}/{rel}", src)

    def add_one(rel: str, required: bool = True) -> None:
        src = run_dir / rel
        if src.is_file():
            add(src, rel)
        elif required:
            missing.append(rel)

    def add_glob(reldir: str, pattern: str, recursive: bool = False) -> None:
        d = run_dir / reldir
        hits = sorted(d.rglob(pattern) if recursive else d.glob(pattern)) if d.is_dir() else []
        if not hits:
            missing.append(f"{reldir}/{pattern}")
        for p in hits:
            add(p, p.relative_to(run_dir).as_posix())

    # boundaries
    add_one(f"{prefix}_hsas_{ver}.geojson")
    # modeling: root files, curated result tables, daily DLNM tables
    add_one(f"modeling/{prefix}_modeling_dataset_{ver}_metadata.json")
    add_one(f"modeling/{prefix}_pipeline_run_log_{ver}.json")
    comp = f"modeling/results_comprehensive_{ver}"
    for nm in ("model_summary.json", "model_comparison_pivot.csv",
               "climate_contribution_analysis.csv", "all_model_results.csv",
               "extended_climate_analysis.json", "feature_selection_info.json"):
        add_one(f"{comp}/{prefix}_{nm}")
    add_one(f"modeling/results_ml_{ver}/{prefix}_model_comparison_summary.csv")
    add_one(f"modeling/results_ml_{ver}/{prefix}_feature_importance_rankings.csv")
    add_one(f"modeling/results_improved_{ver}/{prefix}_improved_model_comparison.csv")
    add_glob(f"modeling/results_parsimonious_{ver}", f"{prefix}_*")
    add_glob(f"modeling/results_anomalies_{ver}", f"{prefix}_anomaly_*")
    add_glob(f"modeling/daily_models_{ver}", "*.csv")
    # sensitivity: every analysis_*_<ver> folder, files scoped to this network+mode
    sroot = run_dir / "sensitivity"
    sdirs = sorted(sroot.glob(f"analysis_*_{ver}")) if sroot.is_dir() else []
    if not sdirs:
        missing.append(f"sensitivity/analysis_*_{ver}/")
    for sd in sdirs:
        for p in sorted(sd.rglob(f"*{prefix}*")):
            add(p, p.relative_to(run_dir).as_posix())
    # textresults: summaries for this network/mode/version
    def md_relevant(name: str) -> bool:
        if network not in name:
            return False
        # Mode as a whole token; the longest match wins, so "fewest" is not
        # mistaken for "governorate_fewest" (or vice versa).
        found = sorted((m for m in MODES if f"_{m}_" in name), key=len)
        if found and found[-1] != mode:
            return False
        vers = [v for v in VERSIONS if v in name]
        return not vers or ver in vers
    tr = run_dir / "textresults"
    if tr.is_dir():
        for p in sorted(tr.glob("*.md")):
            if md_relevant(p.name):
                add(p, f"textresults/{p.name}")
    # steps 1-2 summaries live in the shared out/ folder (those steps ran there)
    if shared_dir is not None:
        for nm in (f"HSA_FINAL_{network}_{ver}_results.md",
                   f"Population_Allocation_Probabilistic_v2_{prefix}_{ver}_results.md"):
            src = shared_dir / "textresults" / nm
            if src.is_file() and f"{ARCHIVE_ROOT}/textresults/{nm}" not in picked:
                add(src, f"textresults/{nm}")
    return picked, missing, skipped


# -------------------------------------------------------------------- building
def skill_context(bundle_key: str, label: str, network: str, mode: str, ver: str,
                  synthetic: bool) -> str:
    lines = [
        f"# Archive: {bundle_key}",
        "",
        f"Results of the HSA climate-health pipeline for one disease ({label}) and one "
        f"delineation ({network} / {mode} / {ver}). Read `manifest.json` first; every path "
        f"in it is relative to the archive root.",
        "",
    ]
    if synthetic:
        lines += ["**SYNTHETIC DATA** (SYNMOD network): these results validate the pipeline "
                  "machinery and are NOT real epidemiology. Never present them as findings "
                  "about Jordan.", ""]
    lines += ["Contents: summary tables only (no figures, no train/val/test splits, no raw "
              f"datasets). Start with `{ARCHIVE_ROOT}/textresults/*.md`, then open the "
              "CSV/JSON tables.", ""]
    return "\n".join(lines)


def build_one(run_dir: Path, dest: Path, shared_dir: Path | None,
              disease: str | None) -> tuple[Path, list[str], list[str]]:
    network, mode, ver = parse_run_dir(run_dir)
    keyword = disease or detect_disease(run_dir, network, mode, ver)
    label = canonical_group(network, keyword)          # e.g. "Diarrheal Diseases"
    dslug = slug(label)                                 # e.g. "diarrheal_diseases"
    prefix = f"{network}_{mode}"
    bundle_key = f"{prefix}_{dslug}_{ver}"
    synthetic = network.upper().startswith("SYNMOD")

    files, missing, skipped = select_files(run_dir, shared_dir, network, mode, ver)
    geo_rel = f"{ARCHIVE_ROOT}/{prefix}_hsas_{ver}.geojson"
    must = [geo_rel,
            f"{ARCHIVE_ROOT}/modeling/results_comprehensive_{ver}/{prefix}_model_summary.json"]
    absent = [m for m in must if m not in files]
    if absent:
        raise PackError(f"{run_dir.name}: required file(s) missing: {absent}")

    n_hsas = len(json.loads(files[geo_rel].read_text()).get("features", []))
    rels = sorted(files)
    manifest = {
        "model": {
            "network": network,
            "hsa_mode": mode,
            "boundary_version": ver,
            "disease_focus_input": label,
            "disease_slug": dslug,
            "model_results": f"{ARCHIVE_ROOT}/modeling/",
            "sensitivity": f"{ARCHIVE_ROOT}/sensitivity/",
            "textresults": f"{ARCHIVE_ROOT}/textresults/",
            "hsa_boundaries": geo_rel,
        },
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "bundle_key": bundle_key,
            "synthetic": synthetic,
            "n_hsas": n_hsas,
            "outcome_col": daily_outcome_col(label),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sections_included": ["model"],
            "file_count": len(rels),
            "files": rels,
        },
    }

    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / f"{bundle_key}.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("SKILL_CONTEXT.md",
                   skill_context(bundle_key, label, network, mode, ver, synthetic))
        for rel in rels:
            z.write(files[rel], arcname=rel)
    tmp.replace(zpath)
    verify(zpath)
    return zpath, missing, skipped


def verify(zpath: Path) -> None:
    """manifest.json at the root, and every manifest path present in the zip."""
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        if "manifest.json" not in names:
            raise PackError(f"{zpath.name}: manifest.json is not at the root")
        m = json.loads(z.read("manifest.json"))
    for k, v in m["model"].items():
        if not (isinstance(v, str) and v.startswith(f"{ARCHIVE_ROOT}/")):
            continue
        ok = any(n.startswith(v) for n in names) if v.endswith("/") else v in names
        if not ok:
            raise PackError(f"{zpath.name}: manifest path model.{k}={v!r} not in the zip")
    for rel in m["meta"]["files"]:
        if rel not in names or PurePosixPath(rel).parts[0] != ARCHIVE_ROOT:
            raise PackError(f"{zpath.name}: meta.files entry {rel!r} not in the zip")


# ------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Package pipeline run folders as agentic archives.")
    ap.add_argument("run_dirs", nargs="+", help="run folders, e.g. out_SYNMOD*_v7")
    ap.add_argument("--dest", default="agentic_archives", help="where to write the zips")
    ap.add_argument("--shared-dir", default="out",
                    help="shared output folder holding step 1-2 summaries (default: out)")
    ap.add_argument("--disease", default=None,
                    help="disease keyword for every run (default: detect per run)")
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    shared = Path(args.shared_dir) if args.shared_dir else None
    if shared is not None and not shared.is_dir():
        shared = None
    run_dirs = sorted({Path(d) for d in args.run_dirs if Path(d).is_dir()})
    if not run_dirs:
        print("ERROR: no run folders given", file=sys.stderr)
        return 1

    built, failed = [], []
    for rd in run_dirs:
        try:
            zpath, missing, skipped = build_one(rd, dest, shared, args.disease)
        except (PackError, KeyError, ValueError, FileNotFoundError) as e:
            failed.append(f"{rd.name}: {e}")
            print(f"[FAIL] {rd.name}: {e}")
            continue
        n = len(json.loads(zipfile.ZipFile(zpath).read("manifest.json"))["meta"]["files"])
        print(f"[ OK ] {rd.name:46s} -> {zpath.name}  ({n} files, "
              f"{zpath.stat().st_size / 1024:.0f} KB)")
        for m in missing:
            print(f"         note: not found: {m}")
        for s in skipped:
            print(f"         note: skipped (>{MAX_FILE_KB} KB): {s}")
        built.append(zpath)

    print(f"\n{len(built)} archive(s) written to {dest}/" +
          (f", {len(failed)} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())