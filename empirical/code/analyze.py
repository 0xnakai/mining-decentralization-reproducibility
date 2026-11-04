#!/usr/bin/env python3
"""Deterministic monthly empirical inference from frozen, height-sorted block arrays.

All statistical settings are declared in empirical/design.json. There is no
network access or latent-ground-truth coverage calculation. Results condition on
the supplied label partition; empirical monthly rates are descriptive.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, math, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'experiments'/'code'))
from inference import candidates, interval_from_counts, topk_error_allocation, RS, DC, DH, GAMMA, KMAX

def dump(path:Path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def iso(sec):return datetime.fromtimestamp(int(sec),timezone.utc).isoformat()

def months(first,last):
 return pd.period_range(first,last,freq='M').astype(str).tolist()

def month_seconds(month):
 p=pd.Period(month,freq='M');s=p.start_time.tz_localize('UTC');e=(p+1).start_time.tz_localize('UTC')
 return int(s.timestamp()),int(e.timestamp())

def write_csv(path,rows):
 pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g',lineterminator='\n')

def quant(x,q):
 # Linear interpolation of the order statistics, with infinite values retained.
 a=np.sort(np.asarray(x,dtype=float));position=(len(a)-1)*q;lo=int(np.floor(position));hi=int(np.ceil(position))
 if lo==hi or a[lo]==a[hi]:return float(a[lo])
 if np.isinf(a[hi]):return math.inf
 return float(a[lo]+(position-lo)*(a[hi]-a[lo]))

def summarize(rows,out):
 df=pd.DataFrame(rows);summary=[]
 for (chain,a,method),g in df.groupby(['chain','anchor_length','method'],sort=True):
  nonempty=g.nc_empty==0
  rec={'chain':chain,'anchor_length':int(a),'method':method,'months':len(g),
    'singleton_months':int(np.sum(nonempty&(g.nc_size==1))),
    'nc_ge2_months':int(np.sum(nonempty&(g.nc_lower>=2))),
    'nc_ge3_months':int(np.sum(nonempty&(g.nc_lower>=3))),
    'nc_unbounded_months':int(np.sum(np.isinf(g.nc_upper))),
    'nc_empty_months':int(g.nc_empty.sum()),'hhi_fallback_months':int(g.hhi_fallback.sum()),
    'hhi_below020_months':int(np.sum(g.hhi_upper<.20)),'hhi_above020_months':int(np.sum(g.hhi_lower>.20)),
    'hhi_unresolved020_months':int(np.sum((g.hhi_lower<=.20)&(g.hhi_upper>=.20))),
    'duration_days_median':quant(g.duration_days,.5),'duration_days_q05':quant(g.duration_days,.05),'duration_days_q95':quant(g.duration_days,.95),
    'nc_lower_median':quant(g.nc_lower,.5),'nc_upper_median':quant(g.nc_upper,.5),'nc_size_median':quant(g.nc_size,.5),'nc_size_q95':quant(g.nc_size,.95),'nc_size_mean':float(g.nc_size.mean()),
    'nc_plugin_median':quant(g.nc_plugin,.5),'hhi_plugin_median':quant(g.hhi_plugin,.5),'hhi_lower_median':quant(g.hhi_lower,.5),'hhi_upper_median':quant(g.hhi_upper,.5),
    'hhi_width_mean':float(g.hhi_width.mean()),'hhi_width_median':quant(g.hhi_width,.5),'hhi_width_q95':quant(g.hhi_width,.95),
    'candidates_k1_median':quant(g.candidates_k1,.5),'candidates_k10_median':quant(g.candidates_k10,.5),
    'unseen_anchor_share_mean':float(g.unseen_anchor_share.mean()),'unknown_prefix_share_mean':float(g.unknown_prefix_share.mean())}
  for event in ['singleton','nc_ge2','nc_ge3','nc_unbounded','hhi_below020','hhi_above020','hhi_unresolved020']:
   rec[event+'_fraction']=rec[event+'_months']/len(g)
  summary.append(rec)
 write_csv(out/'monthly_summary.csv',summary)
 extreme=[]
 for (ch,a,m),g in df.groupby(['chain','anchor_length','method'],sort=True):
  for _,r in g.sort_values(['hhi_width','month'],ascending=[False,True]).head(3).iterrows():extreme.append(r.to_dict())
 write_csv(out/'widest_hhi_windows.csv',extreme)
 paired=[]
 for (ch,a),g in df.groupby(['chain','anchor_length'],sort=True):
  for left,right in [('proposed_multiscale','proposed_short128'),('proposed_multiscale','proposed_long16384')]:
   l=g[g.method==left].set_index('month');r=g[g.method==right].set_index('month');assert list(l.index)==list(r.index)
   ul=l.nc_upper.to_numpy();ur=r.nc_upper.to_numpy();finite=np.isfinite(ul)&np.isfinite(ur)
   paired.append({'chain':ch,'anchor_length':int(a),'left':left,'right':right,'months':len(l),'same_nc_lower_months':int(np.sum(l.nc_lower==r.nc_lower)),
    'smaller_nc_upper_months':int(np.sum(ul<ur)),'equal_nc_upper_months':int(np.sum(ul==ur)),'larger_nc_upper_months':int(np.sum(ul>ur)),
    'both_finite_months':int(np.sum(finite)),'mean_nc_upper_difference_both_finite':float(np.mean(ul[finite]-ur[finite])) if np.any(finite) else math.nan,
    'mean_hhi_width_difference':float(np.mean(l.hhi_width-r.hhi_width)),
    'fewer_nc_infinite_months':int(np.sum(np.isinf(ur))-np.sum(np.isinf(ul)))})
 write_csv(out/'history_paired_descriptive.csv',paired)

def run(data,out,design):
 out.mkdir(parents=True,exist_ok=True)
 cfg=json.loads(design.read_text());assert cfg['history_lengths']==list(RS)
 assert cfg['delta_C']==DC and cfg['delta_H']==DH and cfg['gamma_one_sided']==GAMMA and cfg['kmax']==KMAX
 rows=[];bands=[];counts_rows=[];hist_rows=[];endpoints=[];dataset=[];monthly_counts=[];candidate_records=[]
 for chspec in cfg['chains']:
  ch=chspec['chain'];meta=json.loads((data/f'{ch}_input_manifest.json').read_text())
  assert sha(data/f'{ch}_blocks.npz')==meta['compact_npz_sha256'];assert sha(data/f'{ch}_labels.json')==meta['compact_labels_sha256']
  d=np.load(data/f'{ch}_blocks.npz',allow_pickle=False);h=d['height'];t=d['timestamp_utc_s'];eid=d['entity_code'];source_lines=d['source_line'];status=d['status_code']
  labeldata=json.loads((data/f'{ch}_labels.json').read_text());labels=labeldata['labels'];status_labels=labeldata['status_labels'];m=max(len(labels),KMAX)
  assert np.all(np.diff(h)==1) and len(h)==meta['record_count'];assert min(eid)>=0 and max(eid)<len(labels)
  unknown_codes=np.array([i for i,x in enumerate(labels) if x.startswith('unknown_cluster_')],dtype=int)
  target_start,_=month_seconds(chspec['first_month']);_,target_end=month_seconds(chspec['last_month']);period=(t>=target_start)&(t<target_end)
  pc=np.bincount(eid[period],minlength=m);status_counts={s:int(np.sum(status[period]==i)) for i,s in enumerate(status_labels)}
  ds={'chain':ch,'first_month':chspec['first_month'],'last_month':chspec['last_month'],'saved_records':len(h),'period_records':int(np.sum(period)),
      'period_labels_observed':int(np.sum(pc>0)),'period_unknown_prefix_blocks':int(pc[unknown_codes].sum()),'period_unknown_prefix_fraction':float(pc[unknown_codes].sum()/np.sum(period)),
      'min_height_full':int(h[0]),'max_height_full':int(h[-1]),'period_min_height':int(h[period].min()),'period_max_height':int(h[period].max()),'status_counts':json.dumps(status_counts,sort_keys=True,separators=(',',':'))}
  dataset.append(ds)
  for month in months(chspec['first_month'],chspec['last_month']):
   s,e=month_seconds(month);sel=np.flatnonzero((t>=s)&(t<e));assert len(sel)>0
   # max saved height whose UTC timestamp belongs to the given calendar month
   ix=int(sel[-1]);height=int(h[ix]);assert ix>=max(RS)+max(cfg['anchor_lengths'])
   mc=np.bincount(eid[sel],minlength=m)
   for i in np.flatnonzero(mc):monthly_counts.append({'chain':ch,'month':month,'entity_code':int(i),'entity_label':labels[i],'block_count':int(mc[i]),'month_total':int(len(sel))})
   endpoints.append({'chain':ch,'month':month,'report_height':height,'report_timestamp_utc':iso(t[ix]),'report_source_line':int(source_lines[ix]),
      'calendar_month_block_count':int(len(sel)),'calendar_month_min_height':int(h[sel[0]]),'calendar_month_max_height':height,
      'calendar_month_height_range_count':int(h[sel[-1]]-h[sel[0]]+1),
      'future_height_with_timestamp_before_month_end_count':int(np.sum(t[ix+1:]<e))})
   for a in cfg['anchor_lengths']:
    start=ix-a+1;hist_end=start
    block_counts=np.bincount(eid[start:ix+1],minlength=m).astype(np.int64)
    histories={r:np.bincount(eid[start-r:start],minlength=m).astype(np.int64) for r in RS}
    for r,c in histories.items():
     for i in np.flatnonzero(c):hist_rows.append({'chain':ch,'month':month,'anchor_length':a,'history_length':r,'entity_code':int(i),'count':int(c[i])})
    for i in np.flatnonzero(block_counts):counts_rows.append({'chain':ch,'month':month,'anchor_length':a,'entity_code':int(i),'count':int(block_counts[i])})
    ordered=np.sort(block_counts)[::-1];plug=int(np.flatnonzero(2*np.cumsum(ordered)>a)[0]+1)
    plugin_hhi=float(np.sum(block_counts*block_counts)/(a*a))
    common={'chain':ch,'month':month,'anchor_length':a,'lower_method':cfg['lower_method'],'report_height':height,'anchor_first_height':int(h[start]),'anchor_preceding_height':int(h[start-1]),
      'history_first_height':int(h[start-max(RS)]),'history_last_height':int(h[start-1]),'report_timestamp_utc':iso(t[ix]),
      'anchor_first_timestamp_utc':iso(t[start]),'anchor_preceding_timestamp_utc':iso(t[start-1]),
      'duration_days':float((t[ix]-t[start-1])/86400),'first_last_span_days':float((t[ix]-t[start])/86400),
      'min_max_timestamp_span_days':float((np.max(t[start:ix+1])-np.min(t[start:ix+1]))/86400),
      'backward_timestamp_steps':int(np.sum(np.diff(t[start:ix+1])<0)),'anchor_observed_labels':int(np.sum(block_counts>0)),
      'history_observed_labels':int(np.sum(histories[max(RS)]>0)),'nc_plugin':plug,'hhi_plugin':plugin_hhi,
      'unknown_prefix_share':float(block_counts[unknown_codes].sum()/a),'unseen_anchor_share':float(block_counts[histories[max(RS)]==0].sum()/a)}
    assert common['duration_days']>0
    for mode,method in [('multiscale','proposed_multiscale'),('short128','proposed_short128'),('long16384','proposed_long16384')]:
     family=candidates(histories,mode);res=interval_from_counts(block_counts,family,lower_method=cfg['lower_method'])
     bands_L=res.pop('band_L');bands_U=res.pop('band_U')
     for k,sets in enumerate(family,1):
      w=6/(math.pi**2*k*k);alpha=DC*w/(2*len(sets));beta=DC*w/2
      direct_alpha=float(topk_error_allocation(cfg['lower_method'])['direct'][k-1])
      bands.append({'chain':ch,'month':month,'anchor_length':a,'method':method,'lower_method':cfg['lower_method'],'k':k,'candidate_count':len(sets),'alpha_per_candidate':alpha,'beta':beta,'direct_alpha':direct_alpha,'lower':bands_L[k-1],'upper':bands_U[k-1]})
      windowset=[128] if mode=='short128' else [16384] if mode=='long16384' else list(RS)
      item={'chain':ch,'month':month,'anchor_length':a,'method':method,'lower_method':cfg['lower_method'],'k':k,'alpha_per_candidate':alpha,'beta':beta,'direct_alpha':direct_alpha,'candidates':[]}
      for candidate in sets:
       origins=[]
       for r in windowset:
        c=histories[r];ids=np.flatnonzero(c>0);ids=ids[np.lexsort((ids,-c[ids]))];origin=tuple(sorted(map(int,ids[:k])))
        if origin==candidate:origins.append(r)
       item['candidates'].append({'entity_codes':list(candidate),'history_lengths':origins,'anchor_count':int(block_counts[list(candidate)].sum())})
      candidate_records.append(item)
     rows.append({**common,'method':method,**res})
  print(ch,'ANALYZED',len(months(chspec['first_month'],chspec['last_month'])),'months',flush=True)
  d.close()
 write_csv(out/'monthly_intervals.csv',rows);write_csv(out/'monthly_bands.csv',bands)
 write_csv(out/'anchor_counts.csv',counts_rows);write_csv(out/'history_counts.csv',hist_rows);write_csv(out/'reporting_heights.csv',endpoints)
 write_csv(out/'dataset_summary.csv',dataset);write_csv(out/'monthly_label_counts.csv',monthly_counts)
 with (out/'candidate_sets.jsonl.gz').open('wb') as f:
  with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as g:
   for record in candidate_records:g.write((json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode())
 summarize(rows,out)
 df=pd.DataFrame(rows)
 # Identical anchor statistics imply identical NC lower endpoints in all three candidate variants.
 assert df.groupby(['chain','month','anchor_length']).nc_lower.nunique().max()==1
 expected=3*36*len(cfg['anchor_lengths']);assert len(endpoints)==108 and len(rows)==expected*3
 dump(out/'analysis_checks.json',{'reporting_points':len(endpoints),'anchor_conditions':expected,'method_outputs':len(rows),'band_rows':len(bands),
    'nc_empty_outputs':int(df.nc_empty.sum()),'hhi_fallback_outputs':int(df.hhi_fallback.sum()),'nc_unbounded_outputs':int(np.isinf(df.nc_upper).sum()),
    'three_history_variants_same_nc_lower':True,'lower_method':cfg['lower_method'],'scope':'Descriptive empirical inference conditional on saved labels; no empirical coverage is estimated.',
    'design_sha256':sha(design),'input_manifest_sha256':sha(data/'input_manifest.json')})
 dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'inference_code_sha256':sha(ROOT.parent/'experiments/code/inference.py')})
 print('COMPLETE',len(rows),'method outputs',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'results');p.add_argument('--design',type=Path,default=ROOT/'design.json');a=p.parse_args();run(a.data,a.output,a.design)
if __name__=='__main__':main()
