# Reproduction scope

The inputs support numerical reproduction of the associated paper conditional on the supplied frozen producer-label partition.

- Three chains: 447,049 Bitcoin, 1,762,701 Litecoin, and 10,704,708 pre-Merge Ethereum block records, including histories outside the reporting periods.
- Synthetic experiment: 25 reference weights totaling 4,219; 10 scenarios, 5 target-window sizes, 600 repetitions per condition; saved probability segments, PCG64 seeds, and 30,000 paired samples.
- Stationary audit: 240 distributions at two sample sizes, with 2,000 draws per condition (960,000 draws).
- Empirical calculation: 108 monthly reporting heights, 648 primary reports, and six fixed composite-label resolutions. `replay` generates heights, counts, candidate memberships, bands, intervals, and sensitivity outputs.

`verify` checks supplied files and reaggregates reference outputs. `replay` recomputes intervals from supplied samples and processed block arrays. `full` also regenerates seeded simulation samples. The experimental environment is recorded in `environment/validated_environment.json`.

The complete raw BigQuery acquisition archives, extraction-job logs, final augmentation and unknown-cluster rules, and all manual attribution evidence are not supplied. The retained mapping reference and source identifiers provide provenance, not a complete reconstruction of final assignments. A recorded hash does not substitute for a missing raw payload.

Real-blockchain coverage against unknown latent mining shares is not empirically established. Computational checks do not replace the mathematical coverage proofs.
