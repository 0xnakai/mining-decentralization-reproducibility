#!/usr/bin/env python3
"""Generate the all-scenario supplementary table from saved per-trial endpoints.

No sampling or inference is run. The proposed-method identifier is
proposed_hybrid in experiments/results/primary/synthetic_intervals.csv.gz.
All five block counts are exported separately; the publication table fixes a=512.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd

SCENARIOS = [
    ("stationary", "Stationary reference"),
    ("history_concentration_2048", "Historical concentration"),
    ("history_rank_swap_1024", "Historical rank swap"),
    ("anchor_concentration_30pp_midpoint", "Within-window concentration"),
    ("anchor_rank_swap_midpoint", "Within-window rank swap"),
    ("anchor_unseen18_midpoint", "Moderate entrant"),
    ("anchor_unseen70_midpoint", "Large entrant"),
    ("anchor_unseen80_quarter", "Dominant entrant"),
    ("stationary_nc_boundary", "Strict-majority boundary"),
    ("anchor_gradual_concentration", "Gradual concentration"),
]
BLOCK_COUNTS = [128, 256, 512, 1024, 2048]
COLUMNS = ["method", "scenario", "scenario_index", "anchor_length", "repetition", "seed",
           "inference_block_count", "nc_true", "hhi_true", "nc_lower", "nc_upper",
           "nc_size", "nc_covered", "nc_empty", "nc_singleton", "nc_upper_infinite",
           "hhi_lower", "hhi_upper", "hhi_width", "hhi_covered", "joint_covered"]

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build(synthetic: Path, output: Path, reference_table: Path | None = None) -> dict:
    source = synthetic / "synthetic_intervals.csv.gz" if synthetic.is_dir() else synthetic
    require(source.is_file(), f"Missing per-trial source: {source}")
    records = pd.read_csv(source, usecols=COLUMNS)
    records = records.loc[records.method.eq("proposed_hybrid")].copy()
    keys = ["scenario_index", "anchor_length", "repetition"]
    require(len(records) == 30000, "Expected 30,000 proposed-method records")
    require(not records.duplicated(keys).any(), "Duplicate scenario/block-count/repetition key")
    require(records[keys + ["seed", "nc_true", "hhi_true", "nc_lower", "nc_upper",
                             "hhi_lower", "hhi_upper"]].notna().all().all(), "Missing endpoint or key")
    require(sorted(records.scenario_index.unique()) == list(range(10)), "Scenario IDs differ")
    require(sorted(records.anchor_length.unique()) == BLOCK_COUNTS, "Block counts differ")
    np.testing.assert_array_equal(records.inference_block_count, records.anchor_length)
    nc_size = np.maximum(0, records.nc_upper - records.nc_lower + 1)
    nc_cover = ((records.nc_lower <= records.nc_true) & (records.nc_true <= records.nc_upper)).astype(int)
    hhi_cover = ((records.hhi_lower <= records.hhi_true) & (records.hhi_true <= records.hhi_upper)).astype(int)
    joint = (nc_cover.astype(bool) & hhi_cover.astype(bool)).astype(int)
    widths = records.hhi_upper - records.hhi_lower
    for old, new in [("nc_size", nc_size), ("nc_covered", nc_cover),
                     ("nc_empty", (records.nc_lower > records.nc_upper).astype(int)),
                     ("nc_singleton", (records.nc_lower == records.nc_upper).astype(int)),
                     ("nc_upper_infinite", np.isposinf(records.nc_upper).astype(int)),
                     ("hhi_covered", hhi_cover), ("joint_covered", joint)]:
        np.testing.assert_array_equal(records[old], new, err_msg=old)
    np.testing.assert_allclose(records.hhi_width, widths, rtol=2e-12, atol=2e-12)
    require((widths >= 0).all(), "Negative final HHI width")
    records["computed_nc_size"] = nc_size
    records["computed_nc_covered"] = nc_cover
    records["computed_hhi_covered"] = hhi_cover
    records["computed_joint_covered"] = joint
    records["computed_hhi_width"] = widths
    summary = []
    for (j, a), g in records.groupby(["scenario_index", "anchor_length"], sort=True):
        j, a = int(j), int(a)
        require(len(g) == 600 and sorted(g.repetition) == list(range(600)), "Incomplete repetitions")
        require(g.scenario.eq(SCENARIOS[j][0]).all(), f"Scenario identifier mismatch for S{j}")
        require(g.nc_true.nunique() == 1 and g.hhi_true.nunique() == 1, "Nonconstant target in a condition")
        np.testing.assert_array_equal(g.seed, 20261018 + 10000019*a + 1000003*j + g.repetition)
        summary.append(dict(
            scenario_id=f"S{j}", scenario_name=SCENARIOS[j][1],
            scenario_code=SCENARIOS[j][0], scenario_index=j, target_window_block_count=a,
            method_id="proposed_hybrid", repetitions=600,
            true_nc=int(g.nc_true.iloc[0]), true_hhi=float(g.hhi_true.iloc[0]),
            nc_covered=int(g.computed_nc_covered.sum()), hhi_covered=int(g.computed_hhi_covered.sum()),
            joint_covered=int(g.computed_joint_covered.sum()),
            nc_coverage=float(g.computed_nc_covered.mean()), hhi_coverage=float(g.computed_hhi_covered.mean()),
            joint_coverage=float(g.computed_joint_covered.mean()),
            finite_nc_upper_count=int(np.isfinite(g.nc_upper).sum()),
            nc_empty_count=int(g.nc_empty.sum()), nc_singleton_count=int(g.nc_singleton.sum()),
            mean_nc_set_size=float(g.computed_nc_size.mean()),
            mean_final_hhi_width=float(g.computed_hhi_width.mean()),
        ))
    summary = pd.DataFrame(summary)
    selected = summary[summary.target_window_block_count.eq(512)].copy()
    require(len(selected) == 10, "Expected 10 table rows")
    # These facts support the table's explanatory paragraph, not hidden filtering.
    require(selected.finite_nc_upper_count.eq(600).all(), "The table includes infinite NC endpoints")
    require(selected.hhi_covered.eq(600).all(), "The table includes HHI failures")
    require(selected.nc_empty_count.eq(0).all(), "The table includes empty NC sets")
    subset_checks = []
    if reference_table is not None:
        text = reference_table.read_text(encoding="utf-8")
        for line in text.splitlines():
            if not re.match(r"S[0378] &", line):
                continue
            cells = [x.strip().removesuffix(r"\\").strip() for x in line.split("&")]
            row = selected[selected.scenario_id.eq(cells[0])].iloc[0]
            actual = [str(int(row.true_nc)), f"{row.true_hhi:.4f}",
                      f"{int(row.joint_covered)}/600", f"{row.mean_nc_set_size:.3f}",
                      f"{row.mean_final_hhi_width:.4f}"]
            require(cells[2:] == actual, f"Main-paper Table 5 differs at {cells[0]}: {cells[2:]} vs {actual}")
            subset_checks.append(cells[0])
        require(sorted(subset_checks) == ["S0", "S3", "S7", "S8"], "Expected all four main-paper rows")
    generated = output / "generated"; numeric = output / "numeric_data"
    generated.mkdir(parents=True, exist_ok=True); numeric.mkdir(parents=True, exist_ok=True)
    summary.to_csv(numeric/"all_scenarios_all_block_counts.csv", index=False, float_format="%.17g")
    selected.to_csv(numeric/"all_scenarios_512.csv", index=False, float_format="%.17g")
    table = [r"\begin{table*}[!t]", r"\centering", r"\small",
        r"\caption{Proposed-method outputs for all ten scenarios at $a=512$, with 600 trials per row. Joint coverage counts trials containing both true NC and HHI. Mean NC-set size counts included integers, whereas mean HHI width is the difference between the endpoints of the final interval $I_{\mathrm{HHI}}$. Both means use all 600 trials. Scenario definitions are in Table~\ref{tab:compact-all-scenarios}.}",
        r"\label{tab:all-scenario-results}", r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{.25\textwidth}rrrrr@{}}",
        r"\toprule",
        r"ID & Scenario & \shortstack{True\\NC} & \shortstack{True\\HHI} & \shortstack{Joint\\coverage} & \shortstack{Mean\\NC-set size} & \shortstack{Mean final\\HHI width} \\",
        r"\midrule"]
    for _, row in selected.iterrows():
        table.append(f"{row.scenario_id} & {row.scenario_name} & {int(row.true_nc)} & {row.true_hhi:.4f} & {int(row.joint_covered)}/600 & {row.mean_nc_set_size:.3f} & {row.mean_final_hhi_width:.4f} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    table_file = generated / "all_scenario_results.tex"
    table_file.write_text("\n".join(table), encoding="utf-8")
    report = dict(status="PASS", source_path_suffix="experiments/results/primary/synthetic_intervals.csv.gz",
        source_sha256=digest(source), source_method_id="proposed_hybrid", total_proposed_records=len(records),
        total_conditions=len(summary), displayed_block_count=512, displayed_rows=len(selected),
        trials_per_displayed_row=600, displayed_trials=6000,
        coverage_computed_from_endpoints=True, nc_size_computed_as_integer_cardinality=True,
        hhi_width_computed_from_final_endpoints=True,
        all_512_hhi_covered=True, all_512_nc_upper_finite=True, all_512_nc_nonempty=True,
        matches_main_table5_rows=subset_checks,
        total_nc_failures=int((1-nc_cover).sum()), total_hhi_failures=int((1-hhi_cover).sum()),
        displayed_nc_failures=int((600-selected.nc_covered).sum()),
        simulation_or_inference_rerun=False, averaging_across_block_counts=False,
        float_tolerance=dict(rtol=2e-12, atol=2e-12),
        outputs_sha256={str(p.relative_to(output)):digest(p) for p in [table_file, numeric/"all_scenarios_512.csv", numeric/"all_scenarios_all_block_counts.csv"]})
    (output/"all_scenario_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", required=True, type=Path, help="Primary result directory or per-trial CSV.GZ")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-table", type=Path, help="generated/compact_scenarios.tex for the four-row equality check")
    args = parser.parse_args()
    build(args.synthetic, args.output, args.reference_table)
