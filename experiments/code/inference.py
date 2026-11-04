#!/usr/bin/env python3
"""Finite-history mining decentralization experiments.

All probabilities, seeds, estimators and target definitions are fixed in this file.
No external blockchain data or online service is needed to reproduce the results.
Random samples are sufficient multinomial counts on disjoint constant-probability
segments. Every history and inference window is a union of these segments.
"""
from __future__ import annotations
import argparse, functools, hashlib, json, math, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.special import xlogy, betaincinv, gammaln
from scipy.stats import binom, beta

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results'
INPUTS = ROOT / 'data'
ANCHORS = (128,256,512,1024,2048)
RS = (128,256,512,1024,2048,4096,8192,16384)
DC = DH = .025
GAMMA = .0125
KMAX = 10
BASE_SEED = 20261018
NAMES = ['stationary','history_concentration_2048','history_rank_swap_1024',
         'anchor_concentration_30pp_midpoint','anchor_rank_swap_midpoint',
         'anchor_unseen18_midpoint','anchor_unseen70_midpoint',
         'anchor_unseen80_quarter','stationary_nc_boundary',
         'anchor_gradual_concentration']

def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def cp_interval(successes: int, trials: int, alpha: float=.05) -> tuple[float,float]:
    return (0.0 if successes==0 else float(beta.ppf(alpha/2,successes,trials-successes+1)),
            1.0 if successes==trials else float(beta.ppf(1-alpha/2,successes+1,trials-successes)))

@functools.lru_cache(maxsize=None)
def bounds(n: int, alpha: float, method: str) -> tuple[np.ndarray,np.ndarray]:
    """All possible one-sided KL/Chernoff or exact-binomial CP endpoints."""
    y=np.arange(n+1); x=y/n
    if method=='cp':
        lower=np.zeros(n+1); upper=np.ones(n+1)
        lower[1:]=betaincinv(y[1:],n-y[1:]+1,alpha)
        upper[:-1]=1-betaincinv(n-y[:-1],y[:-1]+1,alpha)
        return lower,upper
    if method!='kl': raise ValueError(method)
    target=np.log(1/alpha)/n
    def divergence(v: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore',invalid='ignore'):
            return xlogy(x,x)-xlogy(x,v)+xlogy(1-x,1-x)-xlogy(1-x,1-v)
    lo=np.zeros(n+1); hi=x.copy()
    for _ in range(56):
        mid=(lo+hi)/2; outside=divergence(mid)>target
        lo=np.where(outside,mid,lo); hi=np.where(outside,hi,mid)
    lower=(lo+hi)/2; lower[0]=0
    lo=x.copy(); hi=np.ones(n+1)
    for _ in range(56):
        mid=(lo+hi)/2; outside=divergence(mid)>target
        hi=np.where(outside,mid,hi); lo=np.where(outside,lo,mid)
    upper=(lo+hi)/2; upper[-1]=1
    return lower,upper

def candidates(histories: dict[int,np.ndarray], mode: str) -> list[list[tuple[int,...]]]:
    windows=(128,) if mode=='short128' else (16384,) if mode=='long16384' else RS
    out=[]
    for k in range(1,KMAX+1):
        family=set()
        for r in windows:
            counts=histories[r]; ids=np.flatnonzero(counts>0)
            order=ids[np.lexsort((ids,-counts[ids]))]
            family.add(tuple(sorted(map(int,order[:k]))))
        out.append(sorted(family))
    return out

def topk_error_allocation(lower_method: str='history', kmax: int=KMAX,
                          delta_c: float=DC) -> dict[str,np.ndarray|float]:
    """Prespecified per-k budgets for the three components of our construction.

    History and direct-only each spend delta_c * W_K; the hybrid spends delta_c.
    The hybrid retains every historical-lower and common-upper allocation and
    assigns the otherwise unallocated finite-k tail mass to its direct lower.
    It does not renormalize or weaken the historical allocations.
    """
    if lower_method not in ('history','direct','hybrid'):
        raise ValueError('lower_method must be history, direct, or hybrid.')
    if kmax<1 or not 0<delta_c<1:
        raise ValueError('Positive kmax and an error probability in (0,1) required.')
    weights=np.array([6/(math.pi**2*k*k) for k in range(1,kmax+1)])
    weight_sum=float(weights.sum());half=delta_c*weights/2
    residual=delta_c*(1-weight_sum)
    history=half.copy() if lower_method!='direct' else np.zeros(kmax)
    direct=(half.copy() if lower_method=='direct' else
            residual*weights/weight_sum if lower_method=='hybrid' else np.zeros(kmax))
    return {'weights':weights,'weight_sum':weight_sum,'history_total':history,
            'upper':half,'direct':direct,'residual':residual,
            'spent':float(np.sum(history+half+direct))}

def direct_topk_lower(counts: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Selection-corrected anchor-only lower bounds, before monotone envelopes.

    Conditional independence suffices, without identical probabilities or an
    upper bound on latent support size. The sqrt(k/n) term controls the expected
    empirical top-k excess; bounded differences controls its upper deviation.
    An alpha of zero disables that coordinate and returns the trivial bound 0.
    """
    counts=np.asarray(counts);alpha=np.asarray(alpha,dtype=float)
    n=int(counts.sum());kmax=len(alpha)
    if n<1 or np.any(counts<0) or np.any((alpha<0)|(alpha>=1)):
        raise ValueError('Nonnegative counts and per-k alpha in [0,1) required.')
    top=np.cumsum(np.sort(np.pad(counts,(0,max(0,kmax-len(counts)))))[::-1])[:kmax]
    answer=np.zeros(kmax);active=alpha>0;k=np.arange(1,kmax+1)
    answer[active]=np.maximum(0.,top[active]/n-np.sqrt(k[active]/n)
                             -np.sqrt(np.log(1/alpha[active])/(2*n)))
    return answer

def interval_from_counts(counts: np.ndarray, family: list[list[tuple[int,...]]]|None,
                         tail_method: str='kl', drift: bool=True, *,
                         lower_method: str='history') -> dict:
    """NC/HHI bounds using history, direct, or hybrid top-k lower bounds.

    The default is the historical component, preserving saved baseline outputs.
    Direct-only may be called with family=None; no historical data are needed.
    HHI uses the chosen top-k curve and the same pair interval. Its [0,1] empty-
    intersection fallback is retained; only the NC set has unconditional
    samplewise containment for hybrid versus history.
    """
    n=int(counts.sum())
    if n<2: raise ValueError('At least two inference blocks are required.')
    allocation=topk_error_allocation(lower_method)
    if lower_method!='direct' and (family is None or len(family)!=KMAX or any(not x for x in family)):
        raise ValueError('One nonempty historical candidate family per k is required.')
    top=np.cumsum(np.sort(np.pad(counts,(0,max(0,KMAX-len(counts)))))[::-1])[:KMAX]
    L=[]; U=[]
    for k in range(1,KMAX+1):
        w=6/(math.pi**2*k*k)
        if lower_method!='direct':
            sets=family[k-1]
            lo,_=bounds(n,DC*w/(2*len(sets)),tail_method)
            L.append(max(float(lo[int(counts[list(s)].sum())]) for s in sets))
        else:
            L.append(0.)
        _,up= bounds(n,DC*w/2,tail_method)
        U.append(float(up[int(top[k-1])]))
    if lower_method!='history':
        L=np.maximum(L,direct_topk_lower(counts,allocation['direct']))
    L=np.maximum.accumulate(L); U=np.minimum.accumulate(U[::-1])[::-1]
    loNC=1+max([0]+[k for k in range(1,KMAX+1) if U[k-1]<=.5])
    hiNC=next((k for k in range(1,KMAX+1) if L[k-1]>.5),math.inf)
    kk=np.arange(1,KMAX+1)
    curve_lo=float(np.max(L*L/kk)); curve_hi=float(np.min(U*U+U*(1-U)/kk))
    V=float(np.sum(counts*(counts-1))/(n*(n-1)))
    r=math.sqrt(math.log(1/GAMMA)/(2*(n//2)))
    pair_lo=max(0.,V-r)
    pair_hi=min(1.,1/n+(1-1/n)*(V+r)) if drift else min(1.,V+r)
    h_lo=max(curve_lo,pair_lo); h_hi=min(curve_hi,pair_hi)
    r_bd=math.sqrt(2*math.log(1/GAMMA)/n)
    bd_lo=max(curve_lo,0.,V-r_bd)
    bd_hi=min(curve_hi,1.,1/n+(1-1/n)*(V+r_bd)) if drift else min(curve_hi,1.,V+r_bd)
    if bd_lo>bd_hi: bd_lo,bd_hi=0.,1.
    fallback=h_lo>h_hi
    if fallback: h_lo,h_hi=0.,1.
    empty=loNC>hiNC
    return {'nc_lower':loNC,'nc_upper':hiNC,
            'nc_size':max(0.,hiNC-loNC+1),'nc_empty':int(empty),
            'hhi_lower':h_lo,'hhi_upper':h_hi,'hhi_width':h_hi-h_lo,
            'hhi_pair_lower':pair_lo,'hhi_pair_upper':pair_hi,
            'hhi_curve_lower':curve_lo,'hhi_curve_upper':curve_hi,
            'hhi_fallback':int(fallback),'pair_frequency':V,
            'hhi_mcd_lower':bd_lo,'hhi_mcd_upper':bd_hi,
            'hhi_mcd_width':bd_hi-bd_lo,'hhi_pair_width':pair_hi-pair_lo,
            'hhi_curve_width':max(0.,curve_hi-curve_lo),
            'candidates_k1':len(family[0]) if lower_method!='direct' else 0,
            'candidates_k10':len(family[-1]) if lower_method!='direct' else 0,
            'band_L':L,'band_U':U}

@functools.lru_cache(maxsize=None)
def binomial_tails(n: int):
    y=np.arange(n+1)
    return binom.sf(y-1,n,.5),binom.cdf(y,n,.5)

def milad_ranges_many(counts: np.ndarray, alpha: float=.05,
                      include_significant_endpoint: bool=False) -> tuple[np.ndarray,np.ndarray]:
    """Vectorization of the published algorithm, including its empirical-NC arithmetic.
    Validated separately against public main.py. Uses no HHI extension.
    """
    counts=np.atleast_2d(counts); n=int(counts[0].sum())
    if np.any(counts.sum(axis=1)!=n): raise ValueError('Equal sample sizes required.')
    if 2.**(-n)>alpha: raise ValueError('Original upper loop may not terminate.')
    top=np.cumsum(np.sort(counts,axis=1)[:,::-1],axis=1)
    sf,cdf=binomial_tails(n)
    # The public compute_nakamoto_coefficient sums floating block shares in rank order.
    # Preserve that behavior; for the new simulations n is a power of two, so shares are exact.
    ordered=np.sort(counts,axis=1)[:,::-1]
    plug=np.argmax(np.cumsum(ordered/n,axis=1)>.5,axis=1)+1
    first_sig=np.argmax(sf[top]<=alpha,axis=1)+1
    hi=first_sig if include_significant_endpoint else np.maximum(plug,first_sig-1)
    # q at the final upper-test count is > alpha at these sample sizes;
    # the original downward search then finds the last lower-tail rejection.
    lo=1+np.sum(cdf[top]<=alpha,axis=1)
    return lo.astype(int),hi.astype(int)

def true_nc(p: np.ndarray) -> int:
    cum=np.cumsum(np.sort(p)[::-1])
    # Analytically specified equalities at 0.5 remain equal after floating-point sums.
    cum[np.isclose(cum,.5,rtol=0,atol=2e-14)]=.5
    return int(np.flatnonzero(cum>.5)[0]+1)

def exact_audit() -> None:
    rows=[]
    for m,alpha,label in [(1,DC*6/math.pi**2/2,'standard_fixed_one_candidate'),
                          (2,DC*6/math.pi**2/4,'standard_fixed_two_candidates'),
                          (2,.025,'diagnostic_alternative_not_standard')]:
        lower,_=bounds(100,alpha,'kl')
        threshold=int(np.flatnonzero(lower>.5)[0])
        wrong=float(m*binom.sf(threshold-1,100,.5))
        rows.append(dict(setting=label,n=100,singleton_candidate_count=m,
                         one_sided_alpha=alpha,minimum_count=threshold,
                         wrong_singleton_rate=wrong,coverage=1-wrong,
                         computation='exact_binomial_tail'))
    pd.DataFrame(rows).to_csv(DATA/'boundary_exact.csv',index=False)
    published=[]
    for n in (100,144):
        c=np.column_stack([np.arange(n+1),n-np.arange(n+1)])
        pmf=binom.pmf(c[:,0],n,.5)
        for diag in (False,True):
            L,U=milad_ranges_many(c,include_significant_endpoint=diag)
            cover=(L<=2)&(2<=U)
            published.append(dict(entities=2,n=n,true_nc=2,
                implementation='inclusive_endpoint_diagnostic' if diag else 'published',
                coverage=float(pmf@cover),wrong_singleton_rate=float(pmf@((L==U)&(L!=2)))))
    n=100
    compositions=np.array([(a,b,c,n-a-b-c) for a in range(n+1)
                           for b in range(n-a+1) for c in range(n-a-b+1)],dtype=np.int16)
    pmf=np.exp(gammaln(n+1)-np.sum(gammaln(compositions+1),axis=1)-n*np.log(4.))
    for diag in (False,True):
        L,U=milad_ranges_many(compositions,include_significant_endpoint=diag)
        published.append(dict(entities=4,n=n,true_nc=3,
            implementation='inclusive_endpoint_diagnostic' if diag else 'published',
            coverage=float(pmf@((L<=3)&(3<=U))),
            wrong_singleton_rate=float(pmf@((L==U)&(L!=3)))))
    pd.DataFrame(published).to_csv(DATA/'published_boundary_exact.csv',index=False)
    write_json(DATA/'exact_audit_checks.json',{'four_entity_configurations':len(compositions),
          'sum_multinomial_probability':float(pmf.sum()),
          'upper_test_false_positive_2_equal_n100':float(2*binom.sf(58,100,.5))})

def scenario_segments(j: int,a: int,p0: np.ndarray) -> list[tuple[int,int,np.ndarray]]:
    begin=a-8192-16384; p1=p0.copy(); split=None
    if j in (1,3):
        delta=.10 if j==1 else .30
        p1[0]=p0[0]+delta; p1[1:]=p0[1:]*(1-p1[0])/(1-p0[0])
        split=-2048 if j==1 else a//2
    elif j in (2,4):
        for x,y in zip(range(4),range(8,12)): p1[x],p1[y]=p0[y],p0[x]
        split=-1024 if j==2 else a//2
    elif j in (5,6,7):
        eta={5:.18,6:.70,7:.80}[j]
        p1=(1-eta)*p0; p1[25]=eta
        split=a//4 if j==7 else a//2
    elif j==8:
        p1[:2]=p0[:2]*(.5/p0[:2].sum());p1[2:]=p0[2:]*(.5/p0[2:].sum())
        return [(begin,a,p1)]
    elif j==9:
        p1[0]=p0[0]+.30
        p1[1:]=p0[1:]*(1-p1[0])/(1-p0[0])
        return [(begin,0,p0)]+[(a*t//8,a*(t+1)//8,
                    (1-(t+1)/8)*p0+((t+1)/8)*p1) for t in range(8)]
    if split is None:return [(begin,a,p0)]
    return [(begin,split,p0),(split,a,p1)]

def make_partition(j:int,a:int,p0:np.ndarray):
    stages=scenario_segments(j,a,p0)
    edges=set([a,a-8192,a-a])
    for n in (a,8192):
        for r in RS:edges.add(a-n-r)
    for l,r,_ in stages:edges.update((l,r))
    edges=np.array(sorted(edges),dtype=int)
    probs=[]
    for l,r in zip(edges[:-1],edges[1:]):
        hits=[p for left,right,p in stages if left<=l and r<=right]
        if len(hits)!=1:raise AssertionError('Partition does not refine probability stages.')
        probs.append(hits[0])
    probs=np.array(probs)
    assert np.allclose(probs.sum(axis=1),1)
    return edges,probs

def summarize(df:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (a,scenario,method),g in df.groupby(['anchor_length','scenario','method'],sort=True):
        n=len(g);row=dict(anchor_length=a,scenario=scenario,method=method,repetitions=n,
           nc_true=int(g.nc_true.iloc[0]),hhi_true=float(g.hhi_true.iloc[0]),
           nc_size_mean=float(g.nc_size.mean()),nc_size_median=float(g.nc_size.median()),
           nc_upper_mean=float(g.nc_upper.mean()),
           hhi_width_mean=float(g.hhi_width.mean()) if g.hhi_width.notna().any() else np.nan,hhi_width_median=float(g.hhi_width.median()) if g.hhi_width.notna().any() else np.nan,
           hhi_width_p90=float(g.hhi_width.quantile(.90)) if g.hhi_width.notna().any() else np.nan,
           hhi_width_p95=float(g.hhi_width.quantile(.95)) if g.hhi_width.notna().any() else np.nan)
        for outcome in ('nc_covered','hhi_covered','joint_covered','nc_singleton',
                        'nc_wrong_singleton','nc_upper_infinite','nc_at_least_2','nc_at_least_3',
                        'hhi_below_020','hhi_above_020','hhi_decided_020',
                        'nc_native_covered','hhi_native_covered'):
            if g[outcome].isna().all():
                row[outcome+'_rate']=np.nan;row[outcome+'_cp95_lower']=np.nan;row[outcome+'_cp95_upper']=np.nan
            else:
                k=int(g[outcome].sum()); lo,hi=cp_interval(k,n)
                row[outcome+'_rate']=k/n;row[outcome+'_cp95_lower']=lo;row[outcome+'_cp95_upper']=hi
        rows.append(row)
    return pd.DataFrame(rows)

def run_synthetic(repetitions:int=600) -> None:
    base=pd.read_csv(INPUTS/'reference_weights.csv')
    p0=np.r_[base.weight.to_numpy(dtype=float)/int(base.weight.sum()),np.zeros(20)]
    rows=[];datasets={};config=[]
    digest=hashlib.sha256(Path(__file__).read_bytes()+(INPUTS/'reference_weights.csv').read_bytes()).hexdigest()[:16]
    checkpoints=DATA/('_checkpoints_'+digest);checkpoints.mkdir(exist_ok=True,parents=True)
    for a in ANCHORS:
        for j,scenario in enumerate(NAMES):
            tag=f'a{a}_s{j}'
            cpbase=checkpoints/f'{tag}_r{repetitions}'
            if cpbase.with_suffix('.json').exists():
                saved=pd.read_csv(cpbase.with_suffix('.csv.gz'))
                rows.extend(saved.to_dict('records'))
                with np.load(cpbase.with_suffix('.npz')) as z:datasets.update({k:z[k] for k in z.files})
                config.append(json.loads(cpbase.with_suffix('.json').read_text()))
                print(f'{scenario} a={a}: resumed completed checkpoint',flush=True)
                continue
            row_start=len(rows)
            edges,probs=make_partition(j,a,p0); lengths=np.diff(edges)
            q=(lengths[edges[:-1]>=0,None]*probs[edges[:-1]>=0]).sum(axis=0)/a
            qlong=(lengths[edges[:-1]>=a-8192,None]*probs[edges[:-1]>=a-8192]).sum(axis=0)/8192
            nc=true_nc(q);hh=float(q@q)
            seeds=np.array([BASE_SEED+10000019*a+1000003*j+r for r in range(repetitions)],dtype=np.uint64)
            samples=np.empty((repetitions,len(lengths),len(p0)),dtype=np.int32)
            for r,seed in enumerate(seeds):
                rng=np.random.Generator(np.random.PCG64(int(seed)))
                for s,(length,p) in enumerate(zip(lengths,probs)):
                    samples[r,s]=rng.multinomial(int(length),p)
            tag=f'a{a}_s{j}'
            datasets[tag+'_counts']=samples;datasets[tag+'_edges']=edges
            datasets[tag+'_probabilities']=probs;datasets[tag+'_seeds']=seeds
            configs=dict(anchor_length=a,scenario_index=j,scenario=scenario,
               nc_truth=nc,hhi_truth=hh,anchor_average=q.tolist(),long_window_average=qlong.tolist(),
               edges=edges.tolist(),probabilities=probs.tolist(),repetitions=repetitions,
               seed_formula='20261018 + 10000019*a + 1000003*j + repetition_index',
               note='Fully specified synthetic benchmark; no empirical network-distribution claim.')
            config.append(configs)
            def aggregate(left:int,right:int):
                ix=(edges[:-1]>=left)&(edges[1:]<=right)
                assert int(lengths[ix].sum())==right-left
                return samples[:,ix].sum(axis=1)
            counts_a=aggregate(0,a); counts_w=aggregate(a-8192,a)
            history_a={w:aggregate(-w,0) for w in RS}
            history_w={w:aggregate(a-8192-w,a-8192) for w in RS}
            milad_out={alpha:milad_ranges_many(counts_a,alpha) for alpha in (.05,.0125)}
            for rep in range(repetitions):
                ha={w:c[rep] for w,c in history_a.items()};hw={w:c[rep] for w,c in history_w.items()}
                for method,mode,tail,drift,n in [
                    ('proposed_short128','short128','kl',True,a),
                    ('proposed_long16384','long16384','kl',True,a),
                    ('proposed_multiscale','multiscale','kl',True,a),
                    ('stationary_CP_anchor','multiscale','cp',False,a),
                    ('stationary_CP_8192','multiscale','cp',False,8192)]:
                    fam=candidates(ha if n==a else hw,mode)
                    result=interval_from_counts(counts_a[rep] if n==a else counts_w[rep],fam,tail,drift)
                    result.pop('band_L');result.pop('band_U')
                    target=q if n==a else qlong
                    row=dict(anchor_length=a,scenario_index=j,scenario=scenario,repetition=rep,
                        seed=int(seeds[rep]),method=method,inference_block_count=n,nc_true=nc,hhi_true=hh,
                        native_nc_true=true_nc(target),native_hhi_true=float(target@target),
                        nc_guarantee_stationary=.975,joint_guarantee_stationary=.95,
                        guarantee_within_anchor_drift=int(drift),**result)
                    rows.append(finish_row(row))
                for alpha,method in [(.05,'Milad_published_alpha005'),(.0125,'Milad_alpha00125_diagnostic')]:
                    lo,hi=milad_out[alpha]
                    row=dict(anchor_length=a,scenario_index=j,scenario=scenario,repetition=rep,
                        seed=int(seeds[rep]),method=method,inference_block_count=a,nc_true=nc,hhi_true=hh,
                        native_nc_true=nc,native_hhi_true=hh,nc_lower=int(lo[rep]),nc_upper=int(hi[rep]),
                        nc_size=int(hi[rep]-lo[rep]+1),nc_empty=int(lo[rep]>hi[rep]),
                        hhi_lower=np.nan,hhi_upper=np.nan,hhi_width=np.nan,
                        nc_guarantee_stationary=np.nan,joint_guarantee_stationary=np.nan,
                        guarantee_within_anchor_drift=0)
                    rows.append(finish_row(row))
            pd.DataFrame(rows[row_start:]).to_csv(cpbase.with_suffix('.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
            np.savez_compressed(cpbase.with_suffix('.npz'),**{k:v for k,v in datasets.items() if k.startswith(tag+'_')})
            write_json(cpbase.with_suffix('.json'),configs)
            print(f'{scenario} a={a}: {repetitions} independent samples completed and saved',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(DATA/'synthetic_intervals.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary=summarize(frame);summary.to_csv(DATA/'synthetic_summary.csv',index=False)
    np.savez_compressed(DATA/'synthetic_segment_counts.npz',**datasets)
    write_json(DATA/'synthetic_design.json',config)
    write_json(DATA/'simulation_environment.json',dict(python=sys.version,numpy=np.__version__,
        scipy=scipy.__version__,pandas=pd.__version__,platform=platform.platform(),
        independent_history_anchor_samples=len(ANCHORS)*len(NAMES)*repetitions,
        paired_method_rows=len(frame),methods=7,random_generator='numpy.random.PCG64',
        inference_tail='KL/Chernoff or one-sided binomial CP, as identified in each method',
        maximum_k=KMAX,renormalize_weights=False,delta_C=DC,delta_H=DH,
        gamma_minus=GAMMA,gamma_plus=GAMMA,
        proposed_selector='Fixed 128, fixed 16384, or deduplicated union of all 8 history windows',
        hhi_decision_threshold=.20,hhi_threshold_status='Prespecified for the simulations; illustrative, not a security cutoff',
        within_month_or_reporting_time_simultaneous_guarantee=False))

def finish_row(row:dict)->dict:
    L,U=row['nc_lower'],row['nc_upper'];Hlo,Hhi=row['hhi_lower'],row['hhi_upper']
    nc=row['nc_true'];hh=row['hhi_true']
    hasH=not np.isnan(Hlo); nonempty=L<=U
    row.update(nc_covered=int(L<=nc<=U),
       hhi_covered=int(Hlo<=hh<=Hhi) if hasH else np.nan,
       joint_covered=int(L<=nc<=U and Hlo<=hh<=Hhi) if hasH else np.nan,
       nc_singleton=int(L==U),nc_wrong_singleton=int(L==U and L!=nc),
       nc_upper_infinite=int(np.isinf(U)),
       nc_at_least_2=int(nonempty and L>=2),nc_at_least_3=int(nonempty and L>=3),
       hhi_below_020=int(Hhi<.2) if hasH else np.nan,
       hhi_above_020=int(Hlo>.2) if hasH else np.nan,
       hhi_decided_020=int(Hhi<.2 or Hlo>.2) if hasH else np.nan,
       nc_native_covered=int(L<=row['native_nc_true']<=U),
       hhi_native_covered=int(Hlo<=row['native_hhi_true']<=Hhi) if hasH else np.nan)
    return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repetitions',type=int,default=600)
    parser.add_argument('--exact-only',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    global DATA
    if args.output_dir is not None: DATA=args.output_dir.resolve()
    DATA.mkdir(exist_ok=True,parents=True);exact_audit()
    if not args.exact_only:run_synthetic(args.repetitions)

if __name__=='__main__':main()
