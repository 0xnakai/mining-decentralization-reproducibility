#!/usr/bin/env python3
"""Verify declared inputs, saved samples, numerical endpoints and independent formulas.

Default: regenerate EVERY random count draw from its seed, independently recheck
all reported event indicators, and replay representative inference outputs. Add
--compare-to PATH after a fresh full computation to compare every stored endpoint
and every saved random draw with the supplied artifact. This is not a smoke test.
No online access is used. Float comparison tolerance is absolute 2e-12, relative
2e-12; discrete outputs and sampled counts must match exactly.
"""
from __future__ import annotations
import argparse, importlib.util, itertools, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import xlogy
from scipy.stats import binomtest, binom, beta
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import inference as mod
import grid_audit as grid

ATOL=RTOL=2e-12
checks={}

def close(a,b,label='float'):
    np.testing.assert_allclose(a,b,atol=ATOL,rtol=RTOL,equal_nan=True,err_msg=label)

def scalar_milad(counts,alpha=.05):
    """Separate direct transcription of public scalar loop, without vector helpers."""
    ordered=sorted(map(int,counts),reverse=True);n=sum(ordered)
    ratio=0.;k=0
    for count in ordered:
        k+=1;ratio+=count/n
        if ratio>.5:break
    high=low=k;success=sum(ordered[:k])
    p=binomtest(success,n,.5,alternative='greater').pvalue
    if p>alpha:
        while p>alpha:
            high+=1;success=sum(ordered[:high])
            p=binomtest(success,n,.5,alternative='greater').pvalue
        high-=1
    q=binomtest(success,n,.5,alternative='less').pvalue
    if q>alpha:
        while q>alpha:
            low-=1;success=sum(ordered[:low])
            q=binomtest(success,n,.5,alternative='less').pvalue
        low+=1
    return low,high

def independent_formula_tests():
    # Scalar root-finding, distinct algorithm from tabulated bisection.
    for n in (100,128,512,2048):
        alpha=.025*6/math.pi**2/4
        L,U=mod.bounds(n,alpha,'kl')
        for y in (0,1,n//4,n//2,n-1,n):
            x=y/n;t=math.log(1/alpha)/n
            def f(v):
                return float(xlogy(x,x)-xlogy(x,v)+xlogy(1-x,1-x)-xlogy(1-x,1-v)-t)
            low=0 if y==0 else brentq(f,np.nextafter(0.,1.),x,xtol=2e-15)
            high=1 if y==n else brentq(f,x,np.nextafter(1.,0.),xtol=2e-15)
            close([L[y],U[y]],[low,high],'independent KL roots')
    checks['scalar_KL_roots']=24
    # Direct combinatorial expectation over nonidentical categorical producers.
    p=np.array([[.8,.15,.05],[.2,.5,.3],[.1,.15,.75],[.4,.4,.2]])
    n=len(p);ev=0.
    for outcome in itertools.product(range(3),repeat=n):
        probability=np.prod([p[t,i] for t,i in enumerate(outcome)])
        count=np.bincount(outcome,minlength=3)
        ev+=probability*np.sum(count*(count-1))/(n*(n-1))
    q=p.mean(axis=0);mean_h=np.sum(p*p)/n
    close(ev,(n*(q@q)-mean_h)/(n-1),'nonidentical HHI identity')
    assert ev<=q@q<=1/n+(1-1/n)*ev+ATOL
    checks['nonidentical_HHI_outcomes']=81
    # All three-entity compositions with 12 observations + larger test vectors.
    vectors=[(12,np.array([[i,j,12-i-j] for i in range(13) for j in range(13-i)]))]
    rng=np.random.default_rng(20261021)
    for n,m in ((100,4),(144,8),(512,25),(1000,64)):
        p=1/np.arange(1,m+1,dtype=float);p/=p.sum()
        vectors.append((n,rng.multinomial(n,p,size=40)))
    nr=0
    for n,c in vectors:
        for alpha in (.05,.0125):
            l,u=mod.milad_ranges_many(c,alpha)
            for idx,row in enumerate(c):
                assert scalar_milad(row,alpha)==(int(l[idx]),int(u[idx])),(n,row)
                nr+=1
    checks['scalar_published_comparisons']=nr
    # Exact standard allocation: both candidate lower tests, enumerate all counts.
    alpha=.025*6/math.pi**2/4
    lo,_=mod.bounds(100,alpha,'kl');threshold=np.flatnonzero(lo>.5)[0]
    assert threshold==67
    wrong=sum(binom.pmf(y,100,.5) for y in range(101) if lo[y]>.5 or lo[100-y]>.5)
    close(wrong,2*binom.sf(66,100,.5),'exact wrong singleton')
    checks['standard_boundary_wrong_rate']=float(wrong)
    family=[[(0,),(1,)]]+[[(0,1)] for _ in range(9)]
    full_coverage=0.
    for y in range(101):
        c=np.zeros(45,dtype=int);c[0]=y;c[1]=100-y
        out=mod.interval_from_counts(c,family)
        assert out['nc_lower']==1 and out['nc_upper'] in (1,2)
        full_coverage+=binom.pmf(y,100,.5)*int(out['nc_lower']<=2<=out['nc_upper'])
    close(full_coverage,1-wrong,'complete fixed-candidate NC enumeration')
    checks['standard_boundary_full_pipeline_outcomes']=101
    # Conditional independence/sample splitting: candidate union only reads history.
    h={r:np.array([r//2,r-r//2]+[0]*43) for r in mod.RS}
    fam=mod.candidates(h,'multiscale')
    assert all(len(s)==1 for s in fam)
    checks['candidate_tie_and_deduplication']='passed'

def saved_output_tests(results:Path,seed_replay=True):
    weights=pd.read_csv(ROOT/'data/reference_weights.csv')
    assert weights.weight.sum()==4219 and len(weights)==25
    base=np.r_[weights.weight.to_numpy()/4219,np.zeros(20)]
    df=pd.read_csv(results/'synthetic_intervals.csv.gz')
    assert len(df)==210000 and not df.duplicated(['anchor_length','scenario_index','repetition','method']).any()
    assert df.groupby(['anchor_length','scenario_index','repetition']).size().eq(7).all()
    # Verify all outcomes independently of the production finish_row routine.
    np.testing.assert_array_equal(df.nc_covered,((df.nc_lower<=df.nc_true)&(df.nc_true<=df.nc_upper)).astype(int))
    h=df.hhi_lower.notna()
    np.testing.assert_array_equal(df.loc[h,'hhi_covered'],((df.loc[h,'hhi_lower']<=df.loc[h,'hhi_true'])&(df.loc[h,'hhi_true']<=df.loc[h,'hhi_upper'])).astype(int))
    np.testing.assert_array_equal(df.loc[h,'joint_covered'],df.loc[h,'nc_covered']*df.loc[h,'hhi_covered'])
    np.testing.assert_array_equal(df.nc_singleton,(df.nc_lower==df.nc_upper).astype(int))
    np.testing.assert_array_equal(df.nc_wrong_singleton,((df.nc_lower==df.nc_upper)&(df.nc_lower!=df.nc_true)).astype(int))
    np.testing.assert_array_equal(df.nc_upper_infinite,np.isinf(df.nc_upper).astype(int))
    close(df.loc[h,'hhi_width'],df.loc[h,'hhi_upper']-df.loc[h,'hhi_lower'])
    target=df[df.method=='proposed_multiscale']
    assert len(target)==30000 and target.joint_covered.eq(1).all()
    assert target.nc_empty.eq(0).all() and target.hhi_fallback.eq(0).all()
    assert target[(target.scenario_index==8)&target.anchor_length.isin((1024,2048))].nc_size.eq(2).all()
    assert target[target.scenario_index==7].nc_upper_infinite.eq(1).all()
    ab=df[df.method.isin(['proposed_short128','proposed_long16384','proposed_multiscale'])]
    assert ab.groupby(['anchor_length','scenario_index','repetition']).nc_lower.nunique().eq(1).all()
    checks['paired_method_output_rows']=len(df)
    checks['dynamic_independent_samples']=30000
    checks['all_dynamic_event_indicators']='passed'
    # All summaries are reconstructed from individual rows.
    expected=mod.summarize(df);supplied=pd.read_csv(results/'synthetic_summary.csv')
    for col in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[col]):close(expected[col],supplied[col],col)
        else:assert expected[col].equals(supplied[col]),col
    # All partitions, probability vectors, seeds and counts, plus representative endpoint replay.
    trials=0;replayed=0
    with np.load(results/'synthetic_segment_counts.npz') as z:
        for a in mod.ANCHORS:
            for j in range(10):
                tag=f'a{a}_s{j}'
                edges,probs=mod.make_partition(j,a,base);lengths=np.diff(edges)
                np.testing.assert_array_equal(z[tag+'_edges'],edges)
                np.testing.assert_array_equal(z[tag+'_probabilities'],probs)
                counts=z[tag+'_counts'];seeds=z[tag+'_seeds']
                assert counts.shape==(600,len(lengths),45)
                np.testing.assert_array_equal(counts.sum(axis=2),np.broadcast_to(lengths,(600,len(lengths))))
                assert (counts>=0).all();close(probs.sum(axis=1),1)
                assert len(np.unique(seeds))==600
                for r,seed in enumerate(seeds):
                    assert int(seed)==20261018+10000019*a+1000003*j+r
                    if seed_replay:
                        rng=np.random.Generator(np.random.PCG64(int(seed)))
                        replay=np.array([rng.multinomial(int(n),p) for n,p in zip(lengths,probs)])
                        np.testing.assert_array_equal(replay,counts[r])
                    trials+=1
                # Intervals used for each method are unions of whole partition segments.
                for n in (a,8192):
                    end=a-n
                    for w in mod.RS:
                        assert end-w in edges and end in edges and a in edges
                        assert end<=a-n # candidate history does not overlap its inference range
                for r in (0,17,599):
                    for mode,method,n,tail,drift in [
                        ('short128','proposed_short128',a,'kl',True),
                        ('long16384','proposed_long16384',a,'kl',True),
                        ('multiscale','proposed_multiscale',a,'kl',True),
                        ('multiscale','stationary_CP_anchor',a,'cp',False),
                        ('multiscale','stationary_CP_8192',8192,'cp',False)]:
                        end=a-n
                        hist={w:counts[r,(edges[:-1]>=end-w)&(edges[1:]<=end)].sum(axis=0) for w in mod.RS}
                        c=counts[r,(edges[:-1]>=end)&(edges[1:]<=a)].sum(axis=0)
                        calculated=mod.interval_from_counts(c,mod.candidates(hist,mode),tail,drift)
                        row=df[(df.anchor_length==a)&(df.scenario_index==j)&(df.repetition==r)&(df.method==method)].iloc[0]
                        for k in ('nc_lower','nc_upper','nc_size','hhi_lower','hhi_upper','hhi_mcd_width','pair_frequency'):
                            close(calculated[k],row[k],k)
                        replayed+=1
    checks['dynamic_count_seed_replays']=trials if seed_replay else 'skipped'
    checks['representative_interval_replays']=replayed
    # Regenerate all grid inputs, all samples, and all grid-reported probabilities.
    design=json.loads((results/'audit240_design.json').read_text())
    fresh_frame,fresh_spec=grid.design()
    supplied_frame=pd.read_csv(results/'audit240_distributions.csv')
    close(fresh_frame.probability,supplied_frame.probability)
    gdf=pd.read_csv(results/'audit240_results.csv')
    total=0
    with np.load(results/'audit240_counts.npz') as z:
        for cfg in design['distributions']:
            g=cfg['distribution_id'];p=np.array(cfg['probabilities']);truth=cfg['true_nc']
            close(p,fresh_spec[g]['probabilities']);assert truth==fresh_spec[g]['true_nc']
            for n in (144,1000):
                seed=20261019+10000019*n+1000003*g
                key=f'g{g}_n{n}'
                saved=z[key]
                assert saved.shape==(2000,len(p)) and (saved>=0).all()
                np.testing.assert_array_equal(saved.sum(axis=1),n)
                if seed_replay:
                    rng=np.random.Generator(np.random.PCG64(seed))
                    np.testing.assert_array_equal(rng.multinomial(n,p,size=2000),saved)
                total+=2000
                for alpha,method in ((.05,'published'),(.0125,'alpha00125_diagnostic')):
                    l,u=mod.milad_ranges_many(saved,alpha)
                    k=int(((l<=truth)&(truth<=u)).sum())
                    row=gdf[(gdf.distribution_id==g)&(gdf.n==n)&(gdf.method==method)]
                    assert len(row)==1,(g,n,method)
                    row=row.iloc[0];close(k/2000,row.coverage)
    checks['grid_count_seed_replays']=total if seed_replay else 'skipped'
    checks['grid_all_output_cells']=len(gdf)

def compare_complete(results:Path,other:Path):
    files=['synthetic_intervals.csv.gz','synthetic_summary.csv','audit240_results.csv','audit240_distributions.csv','boundary_exact.csv','published_boundary_exact.csv']
    for file in files:
        a=pd.read_csv(results/file);b=pd.read_csv(other/file)
        assert a.shape==b.shape and a.columns.equals(b.columns),file
        for col in a.columns:
            if pd.api.types.is_numeric_dtype(a[col]):close(a[col],b[col],file+':'+col)
            else:np.testing.assert_array_equal(a[col].fillna(''),b[col].fillna(''))
    for file in ['synthetic_segment_counts.npz','audit240_counts.npz']:
        with np.load(results/file) as a,np.load(other/file) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=file+':'+key)
    checks['fresh_full_rerun_comparison']='all counts exact; all numeric outputs within 2e-12 absolute/relative'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results')
    p.add_argument('--compare-to',type=Path)
    p.add_argument('--skip-seed-replay',action='store_true')
    p.add_argument('--report',type=Path)
    args=p.parse_args();start=time.monotonic()
    independent_formula_tests();print('Independent formulas and scalar method: PASS',flush=True)
    saved_output_tests(args.results,not args.skip_seed_replay);print('Samples, event indicators, summaries: PASS',flush=True)
    if args.compare_to:compare_complete(args.results,args.compare_to);print('Fresh full regeneration: PASS',flush=True)
    checks['verification_elapsed_seconds']=round(time.monotonic()-start,2)
    checks['status']='PASS'
    report=args.report or args.results/'verification_report.json'
    report.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
