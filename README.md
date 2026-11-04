# Mining decentralization: data and code

Computational companion to **What Can Finite Proof-of-Work Block Histories Certify About Mining Decentralization?**, by Taishi Nakai, Akira Sakurai, and Kazuyuki Shudo.

## Run the calculations

The experiments and full seeded-regeneration validation were performed on **Linux x86_64 with Python 3.13.5**. Versions are recorded in `environment/validated_environment.json`; install the dependencies from `requirements.txt`.

From the extracted repository directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py replay --output runs/replayed
```

`replay` recomputes the numerical results from the supplied processed blockchain data and saved simulation samples. After dependency installation, the commands run offline without BigQuery credentials or a TeX installation.

| Command | Operation |
| --- | --- |
| `python reproduce.py verify --output runs/verified` | Check supplied file hashes and reference results; regenerate the paper's numerical summaries, result figures, and tables. |
| `python reproduce.py replay --output runs/replayed` | Additionally recompute the current confidence intervals and attribution sensitivity results from the supplied inputs. |
| `python reproduce.py full --output runs/full` | Additionally regenerate the synthetic and stationary-audit samples from their seeds, then recompute and compare the results. |

Use a new or empty output directory for each run. A successful run writes `RUN_STATUS.json` with `status: PASS`. Recomputed intervals and intermediate data are stored in that output directory; publication figures, tables, and CSV summaries are in its `publication_assets/` subdirectory. Execution logs are generated under its `logs/` subdirectory. The validated Linux `full` run took approximately 9 minutes; runtime depends on the environment.

## Files

| Location | Contents |
| --- | --- |
| `empirical/data/` | Processed Bitcoin, Litecoin, and pre-Merge Ethereum block arrays; label dictionaries and input manifests. Total: 12,914,458 block records. |
| `empirical/code/`, `empirical/design.json` | Empirical inference, fixed reporting settings, and composite-label sensitivity calculation. |
| `empirical/results_primary/` | Reference numerical outputs needed to compare recomputed results. |
| `experiments/data/`, `experiments/code/` | Reference weights, synthetic models, inference algorithms, and stationary-audit generation. |
| `experiments/results/` | Saved samples, probability/seed specifications, and reference results used by the commands above. |
| `experiments/tests/`, `tools/` | Executable numerical checks and figure/table generation. |
| `publication_templates/` | Figure/table templates and the 64 reference plot values used by the generator. |
| `empirical/provenance/`, `experiments/provenance/` | Relevant third-party reference source, source information, and licenses. |
| `documentation/` | Data schema, method identifiers, figure/table mapping, and reproduction scope. |

See `CONTENTS.md` for the complete file inventory. The `reference_metadata.json` files contain input/code hashes and reference comparison constants, not execution histories. Recompute reporting heights, historical counts, candidate sets, bands, and other derived outputs with `replay`.

## Scope

The empirical analysis starts from **frozen producer-label assignments**. The code reproduces the reported numerical calculations conditional on those labels. The complete raw BigQuery extraction history and every final attribution rule are not supplied, so this package does not claim to reconstruct the entire collection and labeling process. See `documentation/REPRODUCIBILITY_BOUNDARY.md` and `documentation/DATA_DICTIONARY.md`.

## License and citation

Author-created code is licensed under MIT (`LICENSE-CODE`). Authors' contributions to data and documentation are licensed under CC BY 4.0 (`LICENSE-DATA.md`). Existing third-party conditions remain applicable; see `LICENSE_SCOPE.md` and `THIRD_PARTY_NOTICES.md`.

Citation metadata is in `CITATION.cff`. Cite [release v1.0.0](https://github.com/0xnakai/mining-decentralization-reproducibility/releases/tag/v1.0.0) and its associated commit. `SHA256SUMS.txt` provides file checksums. No DOI is assigned.
