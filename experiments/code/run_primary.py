#!/usr/bin/env python3
"""Compute MT/sharp intervals and matched comparisons from saved samples.

Read the supplied segment counts, seeds, and reference outcomes without
modification. Recompute confidence intervals and comparison summaries without
generating new random samples.
"""
from __future__ import annotations
import argparse, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binom
import inference as old
import primary_inference as primary
from io_utils import csv, sha

ROOT=Path(__file__).resolve().parents[1]
METHODS=('proposed_short128','proposed_long16384','proposed_multiscale',
         'proposed_direct','proposed_hybrid','mt_history_full_budget',
         'kl_hybrid','non_iid_tv')
COPIED=('stationary_CP_anchor','stationary_CP_8192',
        'Milad_published_alpha005','Milad_alpha00125_diagnostic')

def relation(left,right):
    lsub=(left.nc_lower>=right.nc_lower)&(left.nc_upper<=right.nc_upper)
    rsub=(left.nc_lower<=right.nc_lower)&(left.nc_upper>=right.nc_upper)
    return {'strict_subset':int((lsub&~rsub).sum()),
            'strict_superset':int((rsub&~lsub).sum()),
            'same':int((lsub&rsub).sum()),'incomparable':int((~lsub&~rsub).sum())}

def boundary_checks():
    alpha=old.DC*6/(math.pi**2)/4
    result={}
    for method in ('mt','kl'):
        lo,_=primary.bounds(100,alpha,method)
        threshold=int(np.flatnonzero(lo>.5)[0])
        result[method]={'minimum_count':threshold,
          'wrong_singleton_probability':float(2*binom.sf(threshold-1,100,.5))}
    assert result['mt']['minimum_count']==64 and result['kl']['minimum_count']==67
    count=0
    for n in (*old.ANCHORS,4096,8192):
        for k in range(1,old.KMAX+1):
            for m in range(1,len(old.RS)+1):
                alpha=old.DC*6/(math.pi**2*k*k)/(2*m)
                mt=primary.bounds(n,alpha,'mt');cp=primary.bounds(n,alpha,'cp')
                np.testing.assert_array_equal(mt[0]>.5,cp[0]>.5)
                np.testing.assert_array_equal(mt[1]<=.5,cp[1]<=.5)
                count+=1
    result['majority_MT_CP_scalar_arrays_checked']=count
    result['packing_extremes']={'zero':float(primary.packing(0.)),
                                'one':float(primary.packing(1.))}
    assert result['packing_extremes']=={'zero':0.,'one':1.}
    return result

def run(baseline:Path,out:Path):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    inputs={name:sha(baseline/name) for name in
            ('synthetic_intervals.csv.gz','synthetic_segment_counts.npz','synthetic_design.json')}
    frozen=pd.read_csv(baseline/'synthetic_intervals.csv.gz')
    reference=frozen[frozen.method=='proposed_multiscale'].set_index(
        ['anchor_length','scenario_index','repetition'])
    assert len(reference)==30000
    rows=[];hhi_rows=[];bands={};replayed=0;cp_matches=0
    cp=frozen[frozen.method=='stationary_CP_anchor'].set_index(
        ['anchor_length','scenario_index','repetition'])
    with np.load(baseline/'synthetic_segment_counts.npz',allow_pickle=False) as z:
        for a in old.ANCHORS:
            for j,scenario in enumerate(old.NAMES):
                tag=f'a{a}_s{j}'
                samples=z[tag+'_counts'];edges=z[tag+'_edges'];seeds=z[tag+'_seeds']
                counts=samples[:,(edges[:-1]>=0)&(edges[1:]<=a)].sum(axis=1)
                histories={r:samples[:,(edges[:-1]>=-r)&(edges[1:]<=0)].sum(axis=1) for r in old.RS}
                assert np.all(counts.sum(axis=1)==a)
                for r,h in histories.items():assert np.all(h.sum(axis=1)==r)
                saved_bands={m:{'L':[],'U':[]} for m in METHODS if m!='non_iid_tv'}
                for rep,count in enumerate(counts):
                    hist={r:h[rep] for r,h in histories.items()}
                    families={m:old.candidates(hist,m) for m in ('multiscale','short128','long16384')}
                    family=families['multiscale']
                    results={
                      'proposed_short128':primary.interval_from_counts(count,families['short128'],lower_method='history'),
                      'proposed_long16384':primary.interval_from_counts(count,families['long16384'],lower_method='history'),
                      'proposed_multiscale':primary.interval_from_counts(count,family,lower_method='history'),
                      'proposed_direct':primary.interval_from_counts(count,None,lower_method='direct'),
                      'proposed_hybrid':primary.interval_from_counts(count,family,lower_method='hybrid'),
                      'mt_history_full_budget':primary.interval_from_counts(count,family,lower_method='history',normalize_history_budget=True),
                      'kl_hybrid':primary.interval_from_counts(count,family,'kl',lower_method='hybrid'),
                      'non_iid_tv':primary.tv_interval_from_counts(count)}
                    hybrid=results['proposed_hybrid'];history=results['proposed_multiscale']
                    np.testing.assert_array_equal(hybrid['band_U'],history['band_U'])
                    assert np.all(hybrid['band_L']>=history['band_L'])
                    assert hybrid['nc_lower']==history['nc_lower'] and hybrid['nc_upper']<=history['nc_upper']
                    cr=cp.loc[(a,j,rep)]
                    assert history['nc_lower']==cr.nc_lower and history['nc_upper']==cr.nc_upper
                    cp_matches+=1
                    upper=hybrid['band_U'];k=np.arange(1,old.KMAX+1)
                    for transform,curve_upper in (
                      ('sharp',hybrid['hhi_curve_upper']),
                      ('elementary',float(np.min(upper**2+upper*(1-upper)/k))),
                      ('kvalseth',float(np.min(np.maximum(upper**2,upper/k))))):
                        lower=max(hybrid['hhi_curve_lower'],hybrid['hhi_pair_lower'])
                        high=min(curve_upper,hybrid['hhi_pair_upper']);fallback=lower>high
                        if fallback:lower,high=0.,1.
                        hhi_rows.append(dict(anchor_length=a,scenario_index=j,scenario=scenario,
                          repetition=rep,transform=transform,curve_upper=curve_upper,
                          hhi_lower=lower,hhi_upper=high,hhi_width=high-lower,fallback=int(fallback)))
                    base=reference.loc[(a,j,rep)].to_dict()
                    for method,res in results.items():
                        L,U=res.pop('band_L'),res.pop('band_U')
                        if method!='non_iid_tv':
                            saved_bands[method]['L'].append(L);saved_bands[method]['U'].append(U)
                        row={**base,'anchor_length':a,'scenario_index':j,'repetition':rep,
                             'seed':int(seeds[rep]),'method':method,**res}
                        rows.append(old.finish_row(row))
                    replayed+=1
                for method,bu in saved_bands.items():
                    for side,arr in bu.items():bands[f'{tag}_{method}_{side}']=np.asarray(arr)
            print(f'a={a}: {replayed} saved history-anchor samples replayed',flush=True)
    frame=pd.DataFrame(rows)
    combined=pd.concat([frame,frozen[frozen.method.isin(COPIED)]],ignore_index=True)
    combined=combined.sort_values(['anchor_length','scenario_index','repetition','method']).reset_index(drop=True)
    assert len(frame)==240000 and len(combined)==360000
    assert not combined.duplicated(['anchor_length','scenario_index','repetition','method']).any()
    csv(combined,out/'synthetic_intervals.csv.gz')
    summary=old.summarize(combined);csv(summary,out/'synthetic_summary.csv')
    hf=pd.DataFrame(hhi_rows);csv(hf,out/'hhi_transform_intervals.csv.gz')
    csv(hf.groupby(['anchor_length','scenario_index','scenario','transform'],sort=True).agg(
        trials=('repetition','size'),curve_upper_mean=('curve_upper','mean'),
        hhi_width_mean=('hhi_width','mean'),fallbacks=('fallback','sum')).reset_index(),out/'hhi_transform_summary.csv')
    temporary=out/f'bands.{os.getpid()}.tmp.npz'
    np.savez_compressed(temporary,**bands);temporary.replace(out/'bands.npz')
    return finish(baseline,out,start)

def finish(baseline,out,start=None):
    """Validate saved replay outputs; also usable after a render-only interruption."""
    if start is None:start=time.monotonic()
    inputs={name:sha(baseline/name) for name in
            ('synthetic_intervals.csv.gz','synthetic_segment_counts.npz','synthetic_design.json')}
    combined=pd.read_csv(out/'synthetic_intervals.csv.gz')
    summary=pd.read_csv(out/'synthetic_summary.csv')
    hf=pd.read_csv(out/'hhi_transform_intervals.csv.gz')
    def get(method):
        return combined[combined.method==method].set_index(['anchor_length','scenario_index','repetition']).sort_index()
    primary_frame=get('proposed_hybrid');hist=get('proposed_multiscale');kl=get('kl_hybrid');tv=get('non_iid_tv')
    cp=get('stationary_CP_anchor')
    cp_matches=int(((hist.nc_lower==cp.nc_lower)&(hist.nc_upper==cp.nc_upper)).sum())
    replayed=len(primary_frame)
    assert cp_matches==replayed==30000
    primary_summary=summary[summary.method=='proposed_hybrid']
    failures=primary_frame[primary_frame.joint_covered==0].reset_index()
    csv(failures,out/'primary_failures.csv')
    s0=primary_frame.loc[(512,0)];s8=primary_frame.loc[(512,8)];s7=primary_frame.loc[(512,7)]
    h7=hist.loc[(512,7)]
    elementary=hf[hf['transform']=='elementary'].reset_index(drop=True)
    sharp=hf[hf['transform']=='sharp'].reset_index(drop=True)
    kvalseth=hf[hf['transform']=='kvalseth'].reset_index(drop=True)
    report={'status':'PASS','saved_sample_count':replayed,'paired_method_outputs':len(combined),
      'method_count':int(combined.method.nunique()),'primary_method':'proposed_hybrid',
      'primary_scalar':'Mattner-Tasto','primary_hhi_conversion':'sharp B_k',
      'primary_nc_failures':int(30000-primary_frame.nc_covered.sum()),
      'primary_hhi_failures':int(30000-primary_frame.hhi_covered.sum()),
      'primary_joint_covered':int(primary_frame.joint_covered.sum()),
      'minimum_condition_joint_coverage':float(primary_summary.joint_covered_rate.min()),
      'failure_breakdown':[{'scenario':'S'+str(j),'anchor_length':int(a),'failures':len(g)}
          for (a,j),g in failures.groupby(['anchor_length','scenario_index'])],
      'primary_hhi_mean_width':float(primary_frame.hhi_width.mean()),
      'mt_hybrid_vs_kl_hybrid':relation(primary_frame,kl),
      'mt_hybrid_vs_history':relation(primary_frame,hist),
      'mt_hybrid_vs_history_full_budget':relation(primary_frame,get('mt_history_full_budget')),
      'mt_hybrid_vs_tv':relation(primary_frame,tv),
      'tv_finite_nc':int(np.isfinite(tv.nc_upper).sum()),'tv_singleton_nc':int(tv.nc_singleton.sum()),
      'tv_hhi_mean_width':float(tv.hhi_width.mean()),
      'CP_anchor_equal_MT_history_nc_samples':cp_matches,
      'sharp_vs_elementary_curve_upper_improved':int((sharp.curve_upper<elementary.curve_upper-1e-12).sum()),
      'sharp_vs_elementary_hhi_improved':int((sharp.hhi_width<elementary.hhi_width-1e-12).sum()),
      'sharp_vs_kvalseth_hhi_improved':int((sharp.hhi_width<kvalseth.hhi_width-1e-12).sum()),
      'focus':{'S0_a512_nc_size_mean':float(s0.nc_size.mean()),'S0_a512_hhi_width_mean':float(s0.hhi_width.mean()),
        'S0_a512_curve_width_mean':float(s0.hhi_curve_width.mean()),'S0_a512_mcd_width_mean':float(s0.hhi_mcd_width.mean()),
        'S0_a512_Milad_covered':int(get('Milad_published_alpha005').loc[(512,0)].nc_covered.sum()),
        'S8_a512_joint_covered':int(s8.joint_covered.sum()),
        'S8_a512_Milad_covered':int(get('Milad_published_alpha005').loc[(512,8)].nc_covered.sum()),
        'S7_a512_history_infinite':int(np.isinf(h7.nc_upper).sum()),
        'S7_a512_hybrid_finite':int(np.isfinite(s7.nc_upper).sum()),
        'S7_a512_hybrid_singletons':int(s7.nc_singleton.sum())},
      'boundary_checks':boundary_checks(),
      'inputs_sha256':inputs,'code_sha256':{'inference.py':sha(Path(old.__file__)),
        'primary_inference.py':sha(Path(primary.__file__)),'run_primary.py':sha(Path(__file__))},
      'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                     'scipy':scipy.__version__,'platform':platform.platform()},
      'notes':['Sample counts, seeds, true targets and published-comparator outputs are supplied as frozen inputs.',
        'MT, sharp-HHI and matched ablations are computed deterministically from the saved samples.',
        'Input files are read without modification; no new random samples are generated.',
        'Primary empirical frequencies do not prove the uniform coverage theorem.'],
      'elapsed_seconds':round(time.monotonic()-start,3)}
    assert report['primary_nc_failures']==6 and report['primary_hhi_failures']==0
    assert report['primary_joint_covered']==29994
    assert report['mt_hybrid_vs_kl_hybrid']=={'strict_subset':5098,'strict_superset':0,'same':24902,'incomparable':0}
    assert report['mt_hybrid_vs_history_full_budget']=={'strict_subset':6854,'strict_superset':169,'same':22977,'incomparable':0}
    assert report['mt_hybrid_vs_tv']=={'strict_subset':29035,'strict_superset':187,'same':772,'incomparable':6}
    assert report['tv_finite_nc']==25096 and report['tv_singleton_nc']==0
    assert report['sharp_vs_elementary_hhi_improved']==2844 and report['sharp_vs_kvalseth_hhi_improved']==1954
    assert report['focus']['S7_a512_hybrid_singletons']==26
    for name,digest in inputs.items():assert sha(baseline/name)==digest
    old.write_json(out/'verification.json',report)
    print(json.dumps({k:report[k] for k in ('status','saved_sample_count','paired_method_outputs',
         'primary_nc_failures','primary_hhi_failures','primary_joint_covered','elapsed_seconds')},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-results',type=Path,default=ROOT/'results')
    p.add_argument('--output',type=Path,default=ROOT/'results'/'primary')
    p.add_argument('--verify-only',action='store_true',help='Validate already reconstructed primary outputs.')
    args=p.parse_args()
    if args.verify_only:finish(args.baseline_results,args.output)
    else:run(args.baseline_results,args.output)
