# Producer-label provenance

The analysis uses the final saved producer-label strings in `empirical/data/*_labels.json`. The source manifests retain the raw labeled-input hashes and source-row indices. Original raw JSONL archives and complete final augmentation/unknown-cluster rules are not included.

The earlier collection/mapping project is [Blockchain-Technology-Lab/consensus-decentralization](https://github.com/Blockchain-Technology-Lab/consensus-decentralization). Its recorded source revision is `4712eeae8e38fb331a0bb269e92c6aca6966d7a6`; the recorded last mapping-information revision is `b4f9483b275f54fa837ec3024c921a9ab40978cb`. These identifiers do not identify every final attribution step.

`mapping_reference.py` is the preserved default-mapping source from that earlier snapshot. It is included as provenance, not as an executable reconstruction of the final data. Its original MIT license is in `LICENSE`. In particular, the source can join multiple matching pool names using a slash; such a label does not establish common ownership.

The sensitivity calculation defines all six whole-label resolutions of `Bitdeer/NiceHash` and `Bitdeer/CKPool/NiceHash` in `empirical/code/attribution_sensitivity.py`. It preserves every other label and every block. The two labels are treated as explicit saved-label ambiguities, without inferring organizational control.

The recorded collection code referred to the public BigQuery Bitcoin, Litecoin, and Ethereum blockchain tables. Extraction job identifiers, complete execution logs, and final manual attribution evidence are not supplied. Numerical reproduction starts at the processed arrays.
