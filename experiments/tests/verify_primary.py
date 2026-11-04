#!/usr/bin/env python3
"""Verify the reconstructed MT/sharp result set and optional fresh replay.

This check addresses version confusion and numerical provenance: it binds every
primary output to the frozen scenario/trial metadata, checks all saved hashes,
and compares every interval when a separately reproduced directory is supplied.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compare(left,right,keys):
    a=pd.read_csv(left).sort_values(keys).reset_index(drop=True)
    b=pd.read_csv(right).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=2e-12,atol=2e-12)
    return len(a)

def run(synthetic,empirical,compare_synthetic=None,compare_empirical=None,report=None):
    checks={}
    s=json.loads((synthetic/'reference_metadata.json').read_text())
    e=json.loads((empirical/'reference_metadata.json').read_text())
    frozen=ROOT/'experiments/results'
    for name,digest in s['inputs_sha256'].items():assert sha(frozen/name)==digest,name
    for name,digest in s['code_sha256'].items():assert sha(ROOT/'experiments/code'/name)==digest,name
    assert sha(ROOT/'experiments/code/primary_inference.py')==e['primary_code_sha256']
    assert sha(ROOT/'empirical/code/analyze_primary.py')==e['adapter_sha256']
    frame=pd.read_csv(synthetic/'synthetic_intervals.csv.gz')
    original=pd.read_csv(frozen/'synthetic_intervals.csv.gz')
    keys=['anchor_length','scenario_index','repetition']
    a=frame[frame.method=='proposed_hybrid'].sort_values(keys).reset_index(drop=True)
    b=original[original.method=='proposed_multiscale'].sort_values(keys).reset_index(drop=True)
    for col in keys+['scenario','seed','nc_true','hhi_true','inference_block_count']:
        pd.testing.assert_series_equal(a[col],b[col],check_names=False,check_dtype=False,check_exact=False,rtol=2e-12,atol=2e-12)
    assert len(a)==30000 and len(frame)==360000 and frame.method.nunique()==12
    assert a.groupby(['anchor_length','scenario_index']).size().eq(600).all()
    assert int((a.nc_covered==0).sum())==6 and int((a.hhi_covered==0).sum())==0
    assert int(a.joint_covered.sum())==29994
    assert a.nc_empty.sum()==0 and a.hhi_fallback.sum()==0
    summary=pd.read_csv(synthetic/'synthetic_summary.csv')
    g=a.groupby(['anchor_length','scenario'])
    expected=g.agg(nc_size_mean=('nc_size','mean'),hhi_width_mean=('hhi_width','mean'),
                   joint_covered_rate=('joint_covered','mean')).sort_index()
    actual=summary[summary.method=='proposed_hybrid'].set_index(['anchor_length','scenario'])[list(expected.columns)].sort_index()
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=2e-12,atol=2e-12)
    em=pd.read_csv(empirical/'monthly_intervals.csv')
    assert len(em)==1944 and em.groupby(['chain','month','anchor_length']).size().eq(3).all()
    assert em.nc_empty.sum()==em.hhi_fallback.sum()==0 and np.isfinite(em.nc_upper).all()
    counts=pd.read_csv(empirical/'monthly_label_counts.csv')
    c=counts[(counts.chain=='btc')&(counts.month=='2026-04')].block_count.to_numpy()
    weights=pd.read_csv(ROOT/'experiments/data/reference_weights.csv').weight.to_numpy()
    np.testing.assert_array_equal(np.sort(c),np.sort(weights));assert c.sum()==4219
    sensitivity=e['attribution_reference']
    assert sensitivity['paired_comparisons']==1296 and sensitivity['nc_set_changed_pairs']==0
    assert sensitivity['primary_code_sha256']==sha(ROOT/'experiments/code/primary_inference.py')
    checks.update(status='PASS',saved_synthetic_samples_verified=30000,synthetic_method_outputs=360000,
       primary_condition_aggregates_checked=50,primary_empirical_outputs=648,
       empirical_history_variant_outputs=1944,reference_weights_recounted=True,
       source_and_primary_code_hashes_match=True,primary_nc_failures=6,primary_hhi_failures=0,
       primary_joint_covered=29994,attribution_pairs=1296)
    if compare_synthetic:
        checks['fresh_synthetic_rows_compared']=compare(synthetic/'synthetic_intervals.csv.gz',
          compare_synthetic/'synthetic_intervals.csv.gz',keys+['method'])
        checks['fresh_hhi_transform_rows_compared']=compare(synthetic/'hhi_transform_intervals.csv.gz',
          compare_synthetic/'hhi_transform_intervals.csv.gz',keys+['transform'])
    if compare_empirical:
        checks['fresh_empirical_rows_compared']=compare(empirical/'monthly_intervals.csv',
          compare_empirical/'monthly_intervals.csv',['chain','month','anchor_length','method'])
        checks['fresh_empirical_ablation_rows_compared']=compare(empirical/'matched_ablation_intervals.csv',
          compare_empirical/'matched_ablation_intervals.csv',['chain','month','anchor_length','method'])
        checks['fresh_attribution_rows_compared']=compare(empirical/'attribution_sensitivity_intervals.csv',
          compare_empirical/'attribution_sensitivity_intervals.csv',['month','anchor_length','variant'])
    if report:
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',type=Path,default=ROOT/'experiments/results/primary')
    p.add_argument('--empirical',type=Path,default=ROOT/'empirical/results_primary')
    p.add_argument('--compare-synthetic',type=Path);p.add_argument('--compare-empirical',type=Path)
    p.add_argument('--report',type=Path)
    a=p.parse_args();run(a.synthetic,a.empirical,a.compare_synthetic,a.compare_empirical,a.report)
