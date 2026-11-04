#!/usr/bin/env python3
"""Deterministic 240-distribution stationary audit of the published NC procedure.

Support sizes are listed in SUPPORTS. For each size, the exponents are j/10,
j=0,...,19, and p_i is proportional to i**(-exponent), i=1,...,m. The exact
uniform threshold is treated analytically; nonuniform targets are evaluated at
80 decimal digits before conversion to the stored floating-point probabilities.
Each grid/sample-size cell uses a separate PCG64 seed and saves all count draws.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import mpmath as mp
from scipy.stats import beta
from inference import milad_ranges_many,cp_interval,write_json
ROOT=Path(__file__).resolve().parents[1]
SUPPORTS=(2,3,4,5,6,8,10,12,16,24,32,64)
SIZES=(144,1000)
SEED=20261019
REPS=2000

def design():
    mp.mp.dps=80;rows=[];spec=[]
    for index_m,m in enumerate(SUPPORTS):
        for j in range(20):
            g=20*index_m+j
            w=[mp.power(mp.mpf(i),-mp.mpf(j)/10) for i in range(1,m+1)]
            p=[x/mp.fsum(w) for x in w]
            truth=m//2+1 if j==0 else next(k for k in range(1,m+1) if mp.fsum(p[:k])>mp.mpf('.5'))
            pp=np.array([float(x) for x in p]);pp=pp/pp.sum()
            # Only analytically exact equalities are allowed near the NC threshold.
            if j>0:
                assert min(abs(mp.fsum(p[:k])-mp.mpf('.5')) for k in range(1,m+1))>mp.mpf('1e-12')
            spec.append(dict(distribution_id=g,support=m,exponent=j/10,true_nc=truth,probabilities=pp.tolist()))
            rows.extend(dict(distribution_id=g,support=m,exponent=j/10,entity_id=i+1,
                             probability=float(pp[i]),true_nc=truth) for i in range(m))
    return pd.DataFrame(rows),spec

def run(out:Path,repetitions:int=REPS):
    out.mkdir(parents=True,exist_ok=True)
    frame,spec=design();frame.to_csv(out/'audit240_distributions.csv',index=False)
    write_json(out/'audit240_design.json',dict(support_sizes=list(SUPPORTS),exponent_numerators=list(range(20)),
        exponent_denominator=10,sample_sizes=list(SIZES),repetitions=repetitions,
        seed_formula='20261019 + 10000019*n + 1000003*distribution_id',
        familywise_comparisons=480,distributions=spec))
    samples={};rows=[]
    for cfg in spec:
        g=cfg['distribution_id'];truth=cfg['true_nc'];p=np.array(cfg['probabilities'])
        for n in SIZES:
            seed=SEED+10000019*n+1000003*g
            rng=np.random.Generator(np.random.PCG64(seed));counts=rng.multinomial(n,p,size=repetitions).astype(np.uint16)
            samples[f'g{g}_n{n}']=counts
            for alpha,label in [(0.05,'published'),(.0125,'alpha00125_diagnostic')]:
                L,U=milad_ranges_many(counts,alpha);success=int(((L<=truth)&(truth<=U)).sum())
                lo,hi=cp_interval(success,repetitions)
                upper_fwer=1. if success==repetitions else float(beta.ppf(1-.05/480,success+1,repetitions-success))
                # FWER flag used in paper is for the primary 0.05 audit only (480 cells).
                rows.append(dict(distribution_id=g,support=cfg['support'],exponent=cfg['exponent'],n=n,
                    method=label,alpha=alpha,seed=seed,repetitions=repetitions,true_nc=truth,
                    coverage_successes=success,coverage=success/repetitions,
                    cp95_lower=lo,cp95_upper=hi,
                    raw_below_095=success/repetitions<.95,
                    simultaneous95_upper=upper_fwer,
                    undercoverage_supported_fwer=upper_fwer<.95,
                    singleton_count=int((L==U).sum()),wrong_singleton_count=int(((L==U)&(L!=truth)).sum())))
        if g%20==19:print(f'Completed support {cfg["support"]}: {g+1}/240 distributions',flush=True)
    pd.DataFrame(rows).to_csv(out/'audit240_results.csv',index=False)
    np.savez_compressed(out/'audit240_counts.npz',**samples)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'results');p.add_argument('--repetitions',type=int,default=REPS)
    a=p.parse_args();run(a.output_dir,a.repetitions)
