#!/usr/bin/env python3
"""Compute empirical MT/sharp intervals from frozen block records.

Use analyze.py for data selection, producer labels, and historical counts,
and primary_inference.py for confidence intervals. Generate the primary
results and matched ablation comparisons without modifying the input data.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import analyze
import primary_inference as primary
from io_utils import csv, sha

ROOT=Path(__file__).resolve().parents[1]

def run(data,out,design):
    analyze.interval_from_counts=primary.interval_from_counts
    analyze.run(data,out,design)
    rows=pd.read_csv(out/'monthly_intervals.csv');summary=pd.read_csv(out/'monthly_summary.csv')
    records=[]
    ac=pd.read_csv(out/'anchor_counts.csv');hc=pd.read_csv(out/'history_counts.csv')
    anchors={k:g for k,g in ac.groupby(['chain','month','anchor_length'])}
    histories={k:g for k,g in hc.groupby(['chain','month','anchor_length'])}
    for _,row in rows[rows.method=='proposed_multiscale'].iterrows():
        key=(row.chain,row.month,int(row.anchor_length));a=anchors[key];h=histories[key]
        size=int(max(a.entity_code.max(),h.entity_code.max()))+1
        count=np.zeros(size,dtype=int);count[a.entity_code.to_numpy()]=a['count'].to_numpy()
        hist={}
        for r,g in h.groupby('history_length'):
            c=np.zeros(size,dtype=int);c[g.entity_code.to_numpy()]=g['count'].to_numpy();hist[int(r)]=c
        family=analyze.candidates(hist,'multiscale')
        for name,kwargs in [('mt_hybrid',{}),('kl_hybrid',{'tail_method':'kl'}),
                            ('mt_history',{'lower_method':'history'}),
                            ('mt_hybrid_elementary',{'hhi_transform':'legacy'})]:
            res=primary.interval_from_counts(count,family,**kwargs)
            res.pop('band_L');res.pop('band_U')
            if name=='mt_hybrid':
                for col in ('nc_lower','nc_upper','hhi_lower','hhi_upper'):
                    np.testing.assert_allclose(res[col],row[col],rtol=2e-12,atol=2e-12)
            records.append({'chain':row.chain,'month':row.month,'anchor_length':int(row.anchor_length),'method':name,**res})
    matched=pd.DataFrame(records);csv(matched,out/'matched_ablation_intervals.csv')
    def get(method):return matched[matched.method==method].set_index(['chain','month','anchor_length']).sort_index()
    mt,kl,hist,elem=[get(m) for m in ('mt_hybrid','kl_hybrid','mt_history','mt_hybrid_elementary')]
    strict=(mt.nc_lower>=kl.nc_lower)&(mt.nc_upper<=kl.nc_upper)&((mt.nc_lower>kl.nc_lower)|(mt.nc_upper<kl.nc_upper))
    widening=(mt.nc_lower<kl.nc_lower)|(mt.nc_upper>kl.nc_upper)
    focus=[]
    for ch in ('btc','ltc','ethpow'):
        for a in (512,4096):
            r=summary[(summary.chain==ch)&(summary.anchor_length==a)&(summary.method=='proposed_multiscale')].iloc[0]
            focus.append({k:(int(r[k]) if k=='singleton_months' else float(r[k])) for k in
                ('duration_days_median','singleton_months','hhi_width_median','hhi_width_q95')}|{'chain':ch,'anchor_length':a})
    checks=json.loads((out/'analysis_checks.json').read_text())
    checks.update(primary_scalar='Mattner-Tasto',primary_hhi_conversion='sharp B_k',
       matched_ablation_outputs=len(matched),mt_vs_kl_strictly_smaller_nc=int(strict.sum()),
       mt_vs_kl_wider_nc=int(widening.sum()),mt_hhi_mean_width=float(mt.hhi_width.mean()),
       kl_hhi_mean_width=float(kl.hhi_width.mean()),elementary_hhi_mean_width=float(elem.hhi_width.mean()),
       sharp_hhi_narrower_than_elementary=int((mt.hhi_upper<elem.hhi_upper-1e-12).sum()),
       hybrid_history_same_nc=int(((mt.nc_lower==hist.nc_lower)&(mt.nc_upper==hist.nc_upper)).sum()),
       focus=focus,primary_code_sha256=sha(Path(primary.__file__)),adapter_sha256=sha(Path(__file__)))
    assert checks['anchor_conditions']==648 and checks['method_outputs']==1944
    assert checks['nc_empty_outputs']==checks['hhi_fallback_outputs']==checks['nc_unbounded_outputs']==0
    assert checks['mt_vs_kl_strictly_smaller_nc']==115 and checks['mt_vs_kl_wider_nc']==0
    assert checks['sharp_hhi_narrower_than_elementary']==64 and checks['hybrid_history_same_nc']==648
    targets={('btc',512):.1479,('btc',4096):.0656,('ltc',512):.1413,('ltc',4096):.0611,('ethpow',512):.1472,('ethpow',4096):.0656}
    for r in focus:assert round(r['hhi_width_median'],4)==targets[(r['chain'],r['anchor_length'])]
    checks['reported_values_match']=True
    analyze.dump(out/'analysis_checks.json',checks)
    env=json.loads((out/'environment.json').read_text())
    env.update(primary_inference_code_sha256=sha(Path(primary.__file__)),analyze_primary_code_sha256=sha(Path(__file__)))
    analyze.dump(out/'environment.json',env)
    print(json.dumps({'status':'PASS','primary_conditions':648,'history_variant_outputs':1944,
      'reported_values_match':True,'mt_vs_kl_strictly_smaller_nc':int(strict.sum())},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'results_primary')
    p.add_argument('--design',type=Path,default=ROOT/'design.json');args=p.parse_args();run(args.data,args.output,args.design)
