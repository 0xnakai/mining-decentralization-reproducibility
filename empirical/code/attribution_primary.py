#!/usr/bin/env python3
"""Replay the six existing composite-label resolutions with MT/sharp limits."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import attribution_sensitivity as audit
import primary_inference as primary

ROOT=Path(__file__).resolve().parents[1]

def run(data,output,design):
    audit.interval_from_counts=primary.interval_from_counts
    audit.run(data,output,design,'hybrid',output/'archived_full_tables')
    for name in ('attribution_sensitivity_design.json','attribution_sensitivity_checks.json'):
        p=output/name;d=json.loads(p.read_text())
        d.update(primary_scalar='Mattner-Tasto',primary_hhi_conversion='sharp B_k',
                 primary_code_sha256=audit.sha(Path(primary.__file__)),adapter_sha256=audit.sha(Path(__file__)))
        audit.dump(p,d)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'results_primary')
    p.add_argument('--design',type=Path,default=ROOT/'design.json');a=p.parse_args();run(a.data,a.output,a.design)
