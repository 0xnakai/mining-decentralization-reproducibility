# File inventory

The package contains the processed inputs, code, reference values, and documentation required by the reproduction commands. Execution reports are generated in the chosen output directory when those commands run.

| File | Purpose |
| --- | --- |
| `.gitattributes` | Documentation, citation, license, or repository configuration |
| `.gitignore` | Documentation, citation, license, or repository configuration |
| `CITATION.cff` | Documentation, citation, license, or repository configuration |
| `LICENSE-CODE` | Documentation, citation, license, or repository configuration |
| `LICENSE-DATA.md` | Documentation, citation, license, or repository configuration |
| `LICENSE_SCOPE.md` | Documentation, citation, license, or repository configuration |
| `README.md` | Documentation, citation, license, or repository configuration |
| `README_ja.md` | Documentation, citation, license, or repository configuration |
| `THIRD_PARTY_NOTICES.md` | Documentation, citation, license, or repository configuration |
| `documentation/DATA_DICTIONARY.md` | Documentation, citation, license, or repository configuration |
| `documentation/REPRODUCIBILITY_BOUNDARY.md` | Documentation, citation, license, or repository configuration |
| `documentation/artifact_map.csv` | Documentation, citation, license, or repository configuration |
| `documentation/method_ids.csv` | Documentation, citation, license, or repository configuration |
| `empirical/code/analyze.py` | Numerical calculation or execution entry point |
| `empirical/code/analyze_primary.py` | Numerical calculation or execution entry point |
| `empirical/code/attribution_primary.py` | Numerical calculation or execution entry point |
| `empirical/code/attribution_sensitivity.py` | Numerical calculation or execution entry point |
| `empirical/data/btc_blocks.npz` | Processed block arrays |
| `empirical/data/btc_input_manifest.json` | Data schema, counts, and input hashes |
| `empirical/data/btc_labels.json` | Code-to-label dictionary |
| `empirical/data/ethpow_blocks.npz` | Processed block arrays |
| `empirical/data/ethpow_input_manifest.json` | Data schema, counts, and input hashes |
| `empirical/data/ethpow_labels.json` | Code-to-label dictionary |
| `empirical/data/input_manifest.json` | Data schema, counts, and input hashes |
| `empirical/data/ltc_blocks.npz` | Processed block arrays |
| `empirical/data/ltc_input_manifest.json` | Data schema, counts, and input hashes |
| `empirical/data/ltc_labels.json` | Code-to-label dictionary |
| `empirical/design.json` | Fixed empirical evaluation settings |
| `empirical/provenance/LICENSE` | Relevant third-party reference source, source identifiers, or license |
| `empirical/provenance/README.md` | Relevant third-party reference source, source identifiers, or license |
| `empirical/provenance/mapping_reference.py` | Relevant third-party reference source, source identifiers, or license |
| `empirical/results_primary/attribution_sensitivity_intervals.csv` | Reference numerical values read by reproduction or comparison |
| `empirical/results_primary/dataset_summary.csv` | Reference numerical values read by reproduction or comparison |
| `empirical/results_primary/matched_ablation_intervals.csv` | Reference numerical values read by reproduction or comparison |
| `empirical/results_primary/monthly_intervals.csv` | Reference numerical values read by reproduction or comparison |
| `empirical/results_primary/monthly_label_counts.csv` | Reference numerical values read by reproduction or comparison |
| `empirical/results_primary/reference_metadata.json` | Method identifiers, hashes, and sensitivity comparison constants |
| `environment/validated_environment.json` | Validated experimental environment or dependency versions |
| `experiments/code/grid_audit.py` | Numerical calculation or execution entry point |
| `experiments/code/inference.py` | Numerical calculation or execution entry point |
| `experiments/code/io_utils.py` | Numerical calculation or execution entry point |
| `experiments/code/primary_inference.py` | Numerical calculation or execution entry point |
| `experiments/code/run_primary.py` | Numerical calculation or execution entry point |
| `experiments/data/reference_weights.csv` | 25 synthetic reference weights, totaling 4,219 |
| `experiments/provenance/published_comparator/LICENSE` | Relevant third-party reference source, source identifiers, or license |
| `experiments/provenance/published_comparator/README.md` | Relevant third-party reference source, source identifiers, or license |
| `experiments/provenance/published_comparator/main.py` | Relevant third-party reference source, source identifiers, or license |
| `experiments/provenance/published_comparator/manifest.json` | Relevant third-party reference source, source identifiers, or license |
| `experiments/results/audit240_counts.npz` | Saved sample counts for replay and seeded-regeneration comparison |
| `experiments/results/audit240_design.json` | Experiment settings, probabilities, and seeds |
| `experiments/results/audit240_distributions.csv` | Reference numerical values read by reproduction or comparison |
| `experiments/results/audit240_results.csv` | Reference numerical values read by reproduction or comparison |
| `experiments/results/boundary_exact.csv` | Reference numerical values read by reproduction or comparison |
| `experiments/results/primary/hhi_transform_intervals.csv.gz` | Reference numerical values read by reproduction or comparison |
| `experiments/results/primary/reference_metadata.json` | Input and code hashes for reference comparison |
| `experiments/results/primary/synthetic_intervals.csv.gz` | Reference numerical values read by reproduction or comparison |
| `experiments/results/primary/synthetic_summary.csv` | Reference numerical values read by reproduction or comparison |
| `experiments/results/published_boundary_exact.csv` | Reference numerical values read by reproduction or comparison |
| `experiments/results/synthetic_design.json` | Experiment settings, probabilities, and seeds |
| `experiments/results/synthetic_intervals.csv.gz` | Reference numerical values read by reproduction or comparison |
| `experiments/results/synthetic_segment_counts.npz` | Saved sample counts for replay and seeded-regeneration comparison |
| `experiments/results/synthetic_summary.csv` | Reference numerical values read by reproduction or comparison |
| `experiments/tests/verify_artifact.py` | Executable numerical verification, integrity check, or figure/table generation |
| `experiments/tests/verify_primary.py` | Executable numerical verification, integrity check, or figure/table generation |
| `publication_templates/expected_plot_values.json` | 64 expected numerical plot values |
| `publication_templates/figures/method_overview.pdf` | Current paper figure/table template used by the generator |
| `publication_templates/figures/method_overview.svg` | Current paper figure/table template used by the generator |
| `publication_templates/generated/all_scenario_results.tex` | Current paper figure/table template used by the generator |
| `publication_templates/generated/compact_all_scenarios.tex` | Current paper figure/table template used by the generator |
| `publication_templates/generated/compact_empirical_results.tex` | Current paper figure/table template used by the generator |
| `publication_templates/generated/compact_empirical_sources.tex` | Current paper figure/table template used by the generator |
| `publication_templates/generated/compact_grid_audit.tex` | Current paper figure/table template used by the generator |
| `publication_templates/generated/compact_scenarios.tex` | Current paper figure/table template used by the generator |
| `publication_templates/generated/comparison_design.tex` | Current paper figure/table template used by the generator |
| `publication_templates/generated/exact.tex` | Current paper figure/table template used by the generator |
| `publication_templates/generated/scenario_overview.tex` | Current paper figure/table template used by the generator |
| `reproduce.py` | Numerical calculation or execution entry point |
| `requirements.txt` | Validated experimental environment or dependency versions |
| `tools/build_all_scenario_table.py` | Executable numerical verification, integrity check, or figure/table generation |
| `tools/check_integrity.py` | Executable numerical verification, integrity check, or figure/table generation |
| `tools/render_publication.py` | Executable numerical verification, integrity check, or figure/table generation |
| `tools/verify_publication.py` | Executable numerical verification, integrity check, or figure/table generation |
| `CONTENTS.md` | This inventory |
| `SHA256SUMS.txt` | Checksums of the distributed files |
