#!/usr/bin/env python3
"""Mattner--Tasto confidence limits and sharp HHI intervals.

Construct simultaneous share bounds, invert them for the Nakamoto coefficient,
and combine the sharp HHI envelope with same-producer pair bounds. Candidate
selection and error allocation use the shared functions in inference.py.
No random sampling occurs in this module.
"""
from __future__ import annotations
import functools
import math
import numpy as np
from scipy.special import betaincinv
import inference as legacy

KMAX, DC, DH, GAMMA = legacy.KMAX, legacy.DC, legacy.DH, legacy.GAMMA

@functools.lru_cache(maxsize=None)
def bounds(n: int, alpha: float, method: str = 'mt'):
    if method != 'mt':
        return legacy.bounds(n, alpha, method)
    if n < 2 or not 0 <= alpha <= .25:
        raise ValueError('MT endpoints require n >= 2 and 0 <= alpha <= 1/4.')
    if alpha == 0:
        return np.zeros(n+1), np.ones(n+1)
    y = np.arange(n+1)
    lower = np.zeros(n+1)
    lower[1] = alpha/n
    lower[2:] = betaincinv(y[2:], n-y[2:]+1, alpha)
    return lower, 1-lower[::-1]

def packing(c):
    c = np.asarray(c, dtype=float)
    out = np.zeros_like(c)
    positive = c > 0
    x = c[positive]
    m = np.floor(1/x)
    out[positive] = m*x*x+(1-m*x)**2
    return out

def sharp_upper(shares):
    shares = np.asarray(shares, dtype=float)
    k = np.arange(1, len(shares)+1)
    return np.maximum(shares*shares, packing(shares/k))

def interval_from_counts(counts, family, tail_method='mt', drift=True, *,
                         lower_method='hybrid', hhi_transform='sharp',
                         normalize_history_budget=False):
    counts = np.asarray(counts)
    n = int(counts.sum())
    if n < 2 or np.any(counts < 0):
        raise ValueError('At least two inference blocks and nonnegative counts required.')
    allocation = legacy.topk_error_allocation(lower_method)
    if normalize_history_budget and lower_method != 'history':
        raise ValueError('Only the full-budget historical ablation is normalized.')
    budget_scale = 1/allocation['weight_sum'] if normalize_history_budget else 1.
    if lower_method != 'direct' and (family is None or len(family) != KMAX or any(not x for x in family)):
        raise ValueError('One nonempty historical family per computed rank required.')
    top = np.cumsum(np.sort(np.pad(counts,(0,max(0,KMAX-len(counts)))))[::-1])[:KMAX]
    lower, upper = [], []
    for k in range(1,KMAX+1):
        w = 6/(math.pi**2*k*k)
        if lower_method != 'direct':
            sets = family[k-1]
            lo,_ = bounds(n,DC*w*budget_scale/(2*len(sets)),tail_method)
            lower.append(max(float(lo[int(counts[list(s)].sum())]) for s in sets))
        else:
            lower.append(0.)
        _,up = bounds(n,DC*w*budget_scale/2,tail_method)
        upper.append(float(up[int(top[k-1])]))
    if lower_method != 'history':
        lower = np.maximum(lower,legacy.direct_topk_lower(counts,allocation['direct']))
    lower = np.maximum.accumulate(lower)
    upper = np.minimum.accumulate(upper[::-1])[::-1]
    nc_lower = 1+max([0]+[k for k in range(1,KMAX+1) if upper[k-1] <= .5])
    nc_upper = next((k for k in range(1,KMAX+1) if lower[k-1] > .5),math.inf)
    k = np.arange(1,KMAX+1)
    curve_lower = float(np.max(lower*lower/k))
    if hhi_transform == 'sharp':
        curve_upper = float(np.min(sharp_upper(upper)))
    elif hhi_transform == 'legacy':
        curve_upper = float(np.min(upper*upper+upper*(1-upper)/k))
    else:
        raise ValueError(hhi_transform)
    v = float(np.sum(counts*(counts-1))/(n*(n-1)))
    r = math.sqrt(math.log(1/GAMMA)/(2*(n//2)))
    pair_lower = max(0.,v-r)
    pair_upper = min(1.,1/n+(1-1/n)*(v+r)) if drift else min(1.,v+r)
    hhi_lower,hhi_upper = max(curve_lower,pair_lower),min(curve_upper,pair_upper)
    r_bd = math.sqrt(2*math.log(1/GAMMA)/n)
    bd_lower = max(curve_lower,0.,v-r_bd)
    bd_upper = min(curve_upper,1.,1/n+(1-1/n)*(v+r_bd)) if drift else min(curve_upper,1.,v+r_bd)
    if bd_lower > bd_upper:
        bd_lower,bd_upper = 0.,1.
    fallback = hhi_lower > hhi_upper
    if fallback:
        hhi_lower,hhi_upper = 0.,1.
    return dict(nc_lower=nc_lower,nc_upper=nc_upper,
        nc_size=max(0.,nc_upper-nc_lower+1),nc_empty=int(nc_lower>nc_upper),
        hhi_lower=hhi_lower,hhi_upper=hhi_upper,hhi_width=hhi_upper-hhi_lower,
        hhi_pair_lower=pair_lower,hhi_pair_upper=pair_upper,
        hhi_curve_lower=curve_lower,hhi_curve_upper=curve_upper,
        hhi_fallback=int(fallback),pair_frequency=v,
        hhi_mcd_lower=bd_lower,hhi_mcd_upper=bd_upper,hhi_mcd_width=bd_upper-bd_lower,
        hhi_pair_width=pair_upper-pair_lower,hhi_curve_width=max(0.,curve_upper-curve_lower),
        candidates_k1=len(family[0]) if lower_method!='direct' else 0,
        candidates_k10=len(family[-1]) if lower_method!='direct' else 0,
        band_L=lower,band_U=upper)

def tv_interval_from_counts(counts):
    """Non-IID total-variation comparator using all observed ranks.

    Its radius is empirical Phi + sqrt(2 log(1/DC)/n); all observed ranks
    are evaluated. The HHI pair interval and transformation match Hybrid.
    """
    counts = np.asarray(counts)
    n = int(counts.sum())
    positive = np.sort(counts[counts>0])[::-1]
    radius = min(1.,float(np.sqrt(positive).sum()/n)+math.sqrt(2*math.log(1/DC)/n))
    top = np.cumsum(positive)/n
    lower, upper = np.maximum(0.,top-radius), np.minimum(1.,top+radius)
    nc_lower = 1+max([0]+[k for k,u in enumerate(upper,1) if u<=.5])
    nc_upper = next((k for k,l in enumerate(lower,1) if l>.5),math.inf)
    curve_lower = float(np.max(lower*lower/np.arange(1,len(lower)+1)))
    curve_upper = float(np.min(sharp_upper(upper)))
    v = float(np.sum(counts*(counts-1))/(n*(n-1)))
    r = math.sqrt(math.log(1/GAMMA)/(2*(n//2)))
    pair_lower, pair_upper = max(0.,v-r), min(1.,1/n+(1-1/n)*(v+r))
    hhi_lower,hhi_upper = max(curve_lower,pair_lower),min(curve_upper,pair_upper)
    fallback = hhi_lower>hhi_upper
    if fallback:
        hhi_lower,hhi_upper = 0.,1.
    return dict(nc_lower=nc_lower,nc_upper=nc_upper,
        nc_size=max(0.,nc_upper-nc_lower+1),nc_empty=int(nc_lower>nc_upper),
        hhi_lower=hhi_lower,hhi_upper=hhi_upper,hhi_width=hhi_upper-hhi_lower,
        hhi_pair_lower=pair_lower,hhi_pair_upper=pair_upper,
        hhi_curve_lower=curve_lower,hhi_curve_upper=curve_upper,
        hhi_fallback=int(fallback),pair_frequency=v,
        hhi_mcd_lower=np.nan,hhi_mcd_upper=np.nan,hhi_mcd_width=np.nan,
        hhi_pair_width=pair_upper-pair_lower,hhi_curve_width=max(0.,curve_upper-curve_lower),
        candidates_k1=0,candidates_k10=0,tv_radius=radius,
        band_L=lower,band_U=upper)
