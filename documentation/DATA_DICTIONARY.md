# Processed blockchain data dictionary

The data are in `empirical/data/`. Each chain has a compressed NumPy `.npz` file, a JSON label dictionary, and a manifest. These are processed analytical inputs, not original BigQuery exports. Stored histories can extend beyond the evaluation periods; evaluation windows are specified by `empirical/design.json`.

| Chain | Stored records | Block heights (inclusive) | Timestamp span (UTC) | Distinct saved labels |
| --- | ---: | --- | --- | ---: |
| Bitcoin (`btc`) | 447,049 | 501,961–949,009 | 2018-01-01T00:08:28+00:00 to 2026-05-11T23:21:49+00:00 | 597 |
| Litecoin (`ltc`) | 1,762,701 | 1,342,486–3,105,186 | 2018-01-01T00:04:36+00:00 to 2026-05-10T23:59:56+00:00 | 2,522 |
| pre-Merge Ethereum (`ethpow`) | 10,704,708 | 4,832,686–15,537,393 | 2018-01-01T00:00:02+00:00 to 2022-09-15T06:42:42+00:00 | 1,720 |

## Array fields

All five arrays have one entry per block and are ordered by increasing block height. Load them with `numpy.load(path, allow_pickle=False)`.

| Field | NumPy dtype | Meaning |
| --- | --- | --- |
| `height` | int32 | Block height. |
| `timestamp_utc_s` | int64 | Unix timestamp in seconds, UTC. Timestamps need not increase at every consecutive height. |
| `entity_code` | int32 | Zero-based index into `labels` in the corresponding chain's `*_labels.json`; not a globally shared entity ID. |
| `source_line` | int32 | Original one-based row number in the saved labeled JSONL source, before sorting by height. |
| `status_code` | int16 | Zero-based index into `status_labels` in the corresponding dictionary. |

Label strings are preserved exactly and sorted by Python Unicode lexicographic order. Unknown-cluster and composite labels retain their original meanings; they must not be silently merged. Original raw JSONL payloads are not bundled. The manifest records their hashes and the hashes of the compressed files, dictionaries, and individual array byte representations.

```python
from pathlib import Path
import json
import numpy as np

root = Path('empirical/data')
dictionary = json.loads((root / 'btc_labels.json').read_text())
with np.load(root / 'btc_blocks.npz', allow_pickle=False) as data:
    first_height = int(data['height'][0])
    first_label = dictionary['labels'][int(data['entity_code'][0])]
    print(first_height, first_label)
```

The empirical calculations condition on this frozen partition. The retained mapping reference and source information are described in `empirical/provenance/README.md`; they do not reconstruct every final attribution decision. See `REPRODUCIBILITY_BOUNDARY.md`.

## Synthetic inputs and outputs

`experiments/data/reference_weights.csv` contains 25 positive weights totaling 4,219. Simulation designs, partition probabilities, saved PCG64 seeds, and segment counts are in `experiments/results/`; the current method's saved intervals are in its `primary/` subdirectory. They cover 10 scenarios, 5 target-window block counts, and 600 repetitions per condition (30,000 paired observations). `audit240_counts.npz` stores 960,000 count draws for 240 stationary distributions and two sample sizes. Numeric result CSVs use the method identifiers explained in `method_ids.csv`.
