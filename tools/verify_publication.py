#!/usr/bin/env python3
"""Check the paper's claims against per-trial records; export traceable plot/table data.

No external data or network access. Nothing in empirical/ or experiments/
is modified. Discrete claims are exact; floating comparisons use 2e-12.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT
sys.path.insert(0, str(UP / 'experiments/code'))
import inference as original


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def verify(synthetic: Path, empirical: Path, baseline: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(synthetic/'synthetic_intervals.csv.gz')
    keys = ['anchor_length','scenario_index','repetition']
    def method(name):
        return rows[rows.method.eq(name)].sort_values(keys).reset_index(drop=True)
    proposed = method('proposed_hybrid')
    history = method('proposed_multiscale')
    kl = method('kl_hybrid')
    milad = method('Milad_published_alpha005')
    for frame in (proposed, history, kl, milad):
        assert len(frame) == 30000
        assert not frame.duplicated(keys).any()
        pd.testing.assert_frame_equal(frame[keys], proposed[keys], check_exact=True)
        np.testing.assert_array_equal(frame.seed, proposed.seed)
        np.testing.assert_array_equal(frame.nc_true, proposed.nc_true)
        np.testing.assert_allclose(frame.hhi_true, proposed.hhi_true, rtol=2e-12, atol=2e-12)
    assert proposed.groupby(keys[:2]).size().eq(600).all()
    assert sorted(proposed.anchor_length.unique()) == [128,256,512,1024,2048]
    assert sorted(proposed.scenario_index.unique()) == list(range(10))
    for frame in (proposed,history,kl,milad):
        np.testing.assert_array_equal(frame.nc_covered, ((frame.nc_lower<=frame.nc_true)&(frame.nc_true<=frame.nc_upper)).astype(int))
        np.testing.assert_array_equal(frame.nc_size, np.maximum(0,frame.nc_upper-frame.nc_lower+1))
        np.testing.assert_array_equal(frame.nc_upper_infinite, np.isinf(frame.nc_upper).astype(int))
        np.testing.assert_array_equal(frame.nc_empty, (frame.nc_lower>frame.nc_upper).astype(int))
        np.testing.assert_array_equal(frame.nc_singleton, (frame.nc_lower==frame.nc_upper).astype(int))
    for frame in (proposed,history,kl):
        np.testing.assert_array_equal(frame.hhi_covered, ((frame.hhi_lower<=frame.hhi_true)&(frame.hhi_true<=frame.hhi_upper)).astype(int))
        np.testing.assert_array_equal(frame.joint_covered, (frame.nc_covered.eq(1)&frame.hhi_covered.eq(1)).astype(int))
        np.testing.assert_allclose(frame.hhi_width, frame.hhi_upper-frame.hhi_lower, rtol=2e-12, atol=2e-12)
    assert milad.hhi_lower.isna().all() and milad.hhi_upper.isna().all()
    assert ((proposed.nc_lower==history.nc_lower)&(proposed.nc_upper<=history.nc_upper)).all()
    # Same historical and upper allocations; the direct allocation is merely disabled.
    ah = original.topk_error_allocation('history'); ab = original.topk_error_allocation('hybrid')
    for key in ['history_total','upper']:
        np.testing.assert_array_equal(ah[key], ab[key])
    assert np.all(ah['direct']==0)
    facts = {}
    def fact(key, actual, expected, location):
        if isinstance(expected, float):
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12, err_msg=key)
        else:
            assert actual == expected, (key,actual,expected)
        facts[key] = dict(value=actual,expected=expected,location=location,status='PASS')
    fact('synthetic_samples',len(proposed),30000,'VI-A / VII-A')
    fact('synthetic_conditions',len(proposed.groupby(keys[:2])),50,'VI-A / VII-A')
    fact('NC_failures',int((proposed.nc_covered==0).sum()),6,'VII-A')
    fact('HHI_failures',int((proposed.hhi_covered==0).sum()),0,'VII-A')
    fact('joint_covered',int(proposed.joint_covered.sum()),29994,'VII-A')
    fact('minimum_condition_joint_coverage',float(proposed.groupby(keys[:2]).joint_covered.mean().min()),598/600,'VII-A')
    fails=proposed[proposed.nc_covered.eq(0)]
    assert fails.scenario_index.eq(8).all()
    assert fails.groupby('anchor_length').size().to_dict()=={256:2,512:1,1024:2,2048:1}
    fails.to_csv(out/'synthetic_failures.csv',index=False)
    def focus(frame,a,j): return frame[frame.anchor_length.eq(a)&frame.scenario_index.eq(j)]
    for j,success in [(0,493),(8,318)]:
        fact(f'Milad_S{j}_512_covered',int(focus(milad,512,j).nc_covered.sum()),success,'VII-A / Figure 2(a)')
    fact('proposed_S0_512_covered',int(focus(proposed,512,0).nc_covered.sum()),600,'VII-A')
    fact('proposed_S8_512_covered',int(focus(proposed,512,8).nc_covered.sum()),599,'VII-A')
    fact('S7_without_direct_infinite_upper',int(focus(history,512,7).nc_upper_infinite.sum()),600,'VII-B')
    fact('S7_with_direct_finite_upper',int(np.isfinite(focus(proposed,512,7).nc_upper).sum()),600,'VII-B')
    fact('S7_with_direct_singletons',int(focus(proposed,512,7).nc_singleton.sum()),26,'VII-B')
    fact('S7_true_NC',int(focus(proposed,512,7).nc_true.iloc[0]),1,'VII-B')
    lsub=(proposed.nc_lower>=kl.nc_lower)&(proposed.nc_upper<=kl.nc_upper)
    strict=lsub&((proposed.nc_lower>kl.nc_lower)|(proposed.nc_upper<kl.nc_upper))
    fact('MT_KL_strict_subset',int(strict.sum()),5098,'Supplement S2.3')
    fact('MT_KL_wider',int((~lsub).sum()),0,'Supplement S2.3')
    for a,j,pm,km in [(2048,0,67,20),(512,3,207,36)]:
        fact(f'MT_singleton_S{j}_{a}',int(focus(proposed,a,j).nc_singleton.sum()),pm,'Supplement S2.3')
        fact(f'KL_singleton_S{j}_{a}',int(focus(kl,a,j).nc_singleton.sum()),km,'Supplement S2.3')
    fact('S0_512_IC_mean_width',float(focus(proposed,512,0).hhi_curve_width.mean()),0.19767078106918548,'Supplement S2.4')
    fact('S0_512_final_HHI_mean_width',float(focus(proposed,512,0).hhi_width.mean()),0.15267653737273312,'Supplement S2.4 / VII-A')
    # Explicit checks for the new result-led captions and Table 3 findings.
    fact('joint_coverage_percent',100*float(proposed.joint_covered.mean()),99.98,'Table 3(A)')
    for j in [0,3,7,8]:
        g=focus(proposed,512,j)
        fact(f'S{j}_512_joint_at_least_9983_percent',bool(g.joint_covered.mean()>=.9983),True,'Table 5 caption')
        fact(f'S{j}_512_mean_HHI_width_below_019',bool(g.hhi_width.mean()<.19),True,'Table 5 caption')
        fact(f'S{j}_512_mean_NC_size_between_166_and_216',bool(1.66<g.nc_size.mean()<2.16),True,'Table 5 caption')
        w=proposed[proposed.scenario_index.eq(j)].groupby('anchor_length').hhi_width.mean().sort_index()
        fact(f'S{j}_mean_HHI_width_strictly_decreases',bool(np.all(np.diff(w.to_numpy())<0)),True,'Table 3(A) / Figure 2 caption')
    summary = original.summarize(rows[rows.method.isin(['proposed_hybrid','proposed_multiscale','kl_hybrid','Milad_published_alpha005'])])
    summary.to_csv(out/'synthetic_summary.csv',index=False,float_format='%.17g')
    pd.DataFrame([dict(scenario='S'+str(j),anchor_length=512,true_nc=int(focus(proposed,512,j).nc_true.iloc[0]),
       true_hhi=float(focus(proposed,512,j).hhi_true.iloc[0]),joint_covered=int(focus(proposed,512,j).joint_covered.sum()),
       repetitions=600,nc_size_mean=float(focus(proposed,512,j).nc_size.mean()),
       hhi_width_mean=float(focus(proposed,512,j).hhi_width.mean())) for j in [0,3,7,8]]).to_csv(out/'table5_data.csv',index=False,float_format='%.17g')
    # Reaggregate the published empirical reports, conditional on the supplied labels.
    emall=pd.read_csv(empirical/'monthly_intervals.csv')
    em=emall[emall.method.eq('proposed_multiscale')].copy()
    assert json.loads((UP/'empirical/results_primary/reference_metadata.json').read_text())['primary_scalar']=='Mattner-Tasto'
    fact('primary_empirical_reports',len(em),648,'VI-C / VII-C')
    fact('empirical_empty_NC',int(em.nc_empty.sum()),0,'VII-C')
    fact('empirical_infinite_NC_upper',int(np.isinf(em.nc_upper).sum()),0,'VII-C')
    fact('empirical_HHI_fallbacks',int(em.hhi_fallback.sum()),0,'VII-C')
    es=[]
    for (chain,a),g in em.groupby(['chain','anchor_length'],sort=True):
        assert len(g)==36
        es.append(dict(chain=chain,anchor_length=int(a),months=len(g),
          duration_days_median=float(g.duration_days.median()),
          nc_lower_median=float(g.nc_lower.median()),nc_upper_median=float(g.nc_upper.median()),
          singleton_months=int(g.nc_singleton.sum()) if 'nc_singleton' in g else int((g.nc_lower==g.nc_upper).sum()),
          hhi_width_median=float(g.hhi_width.median()),hhi_width_q95=float(g.hhi_width.quantile(.95))))
    es=pd.DataFrame(es);es.to_csv(out/'empirical_summary.csv',index=False,float_format='%.17g')
    expected_singletons={'btc':(2,24),'ltc':(11,26),'ethpow':(1,24)}
    table5=[]
    for ch in ['btc','ltc','ethpow']:
        for idx,a in enumerate([512,4096]):
            r=es[(es.chain==ch)&(es.anchor_length==a)].iloc[0]
            fact(f'{ch}_{a}_singletons',int(r.singleton_months),expected_singletons[ch][idx],'VII-C / Table 6')
            if a==512: assert em[em.chain.eq(ch)&em.anchor_length.eq(a)].nc_lower.ge(2).all()
            table5.append(r.to_dict())
    pd.DataFrame(table5).to_csv(out/'table6_data.csv',index=False,float_format='%.17g')
    for ch in ['btc','ltc','ethpow']:
        g=es[es.chain.eq(ch)].set_index('anchor_length').sort_index()
        fact(f'{ch}_HHI_width_more_than_halved_512_to_4096',bool(g.loc[4096,'hhi_width_median']<.5*g.loc[512,'hhi_width_median']),True,'Table 3(C) / Table 6 captions')
        fact(f'{ch}_singleton_NC_sets_increase_512_to_4096',bool(g.loc[4096,'singleton_months']>g.loc[512,'singleton_months']),True,'Table 3(C) / Table 6 captions')
        fact(f'{ch}_median_HHI_width_strictly_decreases',bool(np.all(np.diff(g.hhi_width_median.to_numpy())<0)),True,'Figure 3 caption')
        fact(f'{ch}_median_averaging_duration_strictly_increases',bool(np.all(np.diff(g.duration_days_median.to_numpy())>0)),True,'Figure 3 caption')
    # Exact values are calculated by the upstream full-enumeration implementation.
    exact=pd.read_csv(baseline/'published_boundary_exact.csv')
    for m,n,nc,cv,diag in [(2,100,2,7.9589,91.14),(4,100,3,.1003,75.68),(2,144,2,6.6375,None)]:
        r=exact[(exact.entities==m)&(exact.n==n)]
        x=100*float(r[r.implementation=='published'].coverage.iloc[0])
        assert round(x,4)==cv
        if diag is not None:
            y=100*float(r[r.implementation=='inclusive_endpoint_diagnostic'].coverage.iloc[0])
            assert round(y,2)==diag
        fact(f'exact_{m}_{n}_coverage_percent_rounded',round(x,4),cv,'III-D / Table 2')
    exact.to_csv(out/'table2_data.csv',index=False,float_format='%.17g')
    # Reaggregate the 480 primary stationary audit cells (each with 2,000 samples).
    grid=pd.read_csv(baseline/'audit240_results.csv');grid=grid[grid.method.eq('published')]
    assert len(grid)==480 and grid.repetitions.eq(2000).all()
    gr=[]
    for n,g in grid.groupby('n'):
        gr.append(dict(n=int(n),mean_coverage=float(g.coverage.mean()),median_coverage=float(g.coverage.median()),
          raw_count=int(g.raw_below_095.sum()),fwer_count=int(g.undercoverage_supported_fwer.sum())))
    grid_summary=pd.DataFrame(gr);grid_summary.to_csv(out/'tableS2_data.csv',index=False,float_format='%.17g')
    for n,mean,median,raw,adj in [(144,79.92,97.67,107,100),(1000,88.28,100.,61,59)]:
        r=grid_summary[grid_summary.n.eq(n)].iloc[0]
        assert f'{100*r.mean_coverage:.2f}'==f'{mean:.2f}' and f'{100*r.median_coverage:.2f}'==f'{median:.2f}'
        assert int(r.raw_count)==raw and int(r.fwer_count)==adj
    ds=pd.read_csv(empirical/'dataset_summary.csv');ds.to_csv(out/'tableS4_data.csv',index=False)
    # All 64 values in the empirical/synthetic publication plots, including NC error bars.
    plot=[]
    for name in ['proposed_hybrid','Milad_published_alpha005']:
        for j in [0,3,7,8]:
            g=focus(method(name),512,j);k=int(g.nc_covered.sum());lo,hi=original.cp_interval(k,600)
            plot.append(dict(figure='synthetic',panel='NC coverage',scenario='S'+str(j),method=name,
                             anchor_length=512,value=k/600,lower=lo,upper=hi))
    for j in [0,3,7,8]:
        for a in [128,256,512,1024,2048]:
            plot.append(dict(figure='synthetic',panel='HHI width',scenario='S'+str(j),anchor_length=a,
                             value=float(focus(proposed,a,j).hhi_width.mean())))
    for ch in ['btc','ltc','ethpow']:
        for col in ['duration_days_median','hhi_width_median']:
            for _,r in es[es.chain.eq(ch)].sort_values('anchor_length').iterrows():
                plot.append(dict(figure='empirical',panel=col,chain=ch,anchor_length=int(r.anchor_length),value=float(r[col])))
    oldplot=json.loads((ROOT/'publication_templates/expected_plot_values.json').read_text())
    def sk(r): return tuple(str(r.get(k,'')) for k in ['figure','panel','scenario','method','chain','anchor_length'])
    assert len(plot)==len(oldplot)==64
    for a,b in zip(sorted(plot,key=sk),sorted(oldplot,key=sk)):
        assert sk(a)==sk(b)
        for k in ['value','lower','upper']:
            if k in a: np.testing.assert_allclose(a[k],b[k],rtol=2e-12,atol=2e-12)
    pd.DataFrame(plot).to_csv(out/'figure_data.csv',index=False,float_format='%.17g')
    write_json(out/'figure_data.json',plot)
    pd.DataFrame([dict(claim=k,**v) for k,v in facts.items()]).to_csv(out/'verified_claims.csv',index=False)
    record=dict(status='PASS',companion_version='1.0.0',synthetic_independent_samples=30000,synthetic_primary_conditions=50,
      synthetic_stored_method_rows=len(rows),primary_empirical_reports=648,primary_stationary_grid_cells=480,
      primary_stationary_random_samples=960000,plot_values_checked=64,claim_checks=len(facts),
      failure_counts_by_window={str(int(k)):int(v) for k,v in fails.groupby('anchor_length').size().items()},
      matched_history_containment=True,source_identifiers_unchanged=True,HHI_not_attributed_to_Milad=True,
      comparison_tolerance=dict(rtol=2e-12,atol=2e-12),claims=facts,
      boundary='These are computational checks on saved or regenerated records, not independent verification of upstream entity attribution or a proof of uniform coverage.')
    write_json(out/'PUBLICATION_VALIDATION.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='claims'},indent=2))
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',type=Path,default=UP/'experiments/results/primary')
    p.add_argument('--empirical',type=Path,default=UP/'empirical/results_primary')
    p.add_argument('--baseline',type=Path,default=UP/'experiments/results')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.synthetic.resolve(),a.empirical.resolve(),a.baseline.resolve(),a.output.resolve())
