#!/usr/bin/env python3
"""Verify the curated companion against its own SHA256SUMS.txt (offline)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(report: Path | None = None) -> dict:
    errors = []
    count = 0
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = (ROOT / relative.lstrip('*')).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append(relative)
        else:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                errors.append(relative)
        count += 1
    result = dict(status='FAIL' if errors else 'PASS', expected_files=count,
                  matched_files=count-len(errors), errors=errors)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if errors:
        raise RuntimeError('Companion file integrity check failed.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    verify(parser.parse_args().report)
