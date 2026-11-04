# Pinned source for the published NC comparison

`main.py` is an unmodified copy of the public source in
`Blockchain-Technology-Lab/nc-statistical-confidence`, retrieved directly by its
Git blob identifier on 2026-10-19. `LICENSE` is the corresponding unmodified
upstream MIT license. Both byte sequences have been verified against their Git
blob identifiers, and both occur in the pinned repository commit below.

- Repository: https://github.com/Blockchain-Technology-Lab/nc-statistical-confidence
- Commit: `c2ba171844a52463dc8f4c2a837821b43e869c3e`
- Tree: `d8e67fa7cf59eb55ff3ca9b819b49ad947937670`
- Source: https://github.com/Blockchain-Technology-Lab/nc-statistical-confidence/blob/c2ba171844a52463dc8f4c2a837821b43e869c3e/main.py
- License: https://github.com/Blockchain-Technology-Lab/nc-statistical-confidence/blob/c2ba171844a52463dc8f4c2a837821b43e869c3e/LICENSE

The source file contains `compute_nakamoto_coefficient` (lines 31--44),
`compute_nakamoto_coefficients` (lines 47--56), and `find_nc_range` (lines 59--115).
The complete file is retained so that the point-estimate computation,
floating-point summation, binomial-tail alternatives, and endpoint adjustments
can be inspected in their original context. It is provenance for the comparator,
not a replacement for the artifact's vectorized implementation or its independent
scalar checks. The source's historical data-acquisition and plotting workflow is
not claimed to have been reproduced by saving this file.

The source Git blob is the same identifier already recorded for the comparison
audited in the manuscript. `manifest.json` records the full SHA-256 and Git blob
identifiers. `SHA256SUMS.txt` verifies the two upstream files locally.
