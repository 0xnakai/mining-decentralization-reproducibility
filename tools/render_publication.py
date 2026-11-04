#!/usr/bin/env python3
"""Publication figure and table generation from verified numerical records.
Read per-trial records, verify all displayed values, then render publication figures and tables.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

RELEASE=Path(__file__).resolve().parents[1]
ROOT=RELEASE
import shutil
from verify_publication import verify
from build_all_scenario_table import build as build_all_scenario_table
COLORS=['#254B70','#C95032','#23826C','#8263A3']
FOCUS=[0,3,7,8]
SCENARIOS=['stationary','history_concentration_2048','history_rank_swap_1024',
 'anchor_concentration_30pp_midpoint','anchor_rank_swap_midpoint','anchor_unseen18_midpoint',
 'anchor_unseen70_midpoint','anchor_unseen80_quarter','stationary_nc_boundary','anchor_gradual_concentration']
CHAIN_NAMES={'btc':'Bitcoin','ltc':'Litecoin','ethpow':'pre-Merge Ethereum'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(text,encoding='utf-8');tmp.replace(path)

def figure(fig,path):
    """Never expose a partly written or stale PDF to the concurrent TeX build."""
    path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):
        target=path.with_suffix('.'+ext)
        temp=target.with_name(target.stem+f'.{os.getpid()}.tmp.'+ext)
        fig.savefig(temp,format=ext,dpi=200,facecolor='white')
        if ext=='pdf':
            with fitz.open(temp) as doc:
                assert len(doc)==1 and doc[0].rect.width>100 and len(doc[0].get_text())>20
        else:
            assert temp.stat().st_size>10000
        temp.replace(target)
    plt.close(fig)

def run(synthetic, empirical, baseline, output):
    output.mkdir(parents=True,exist_ok=True)
    verify(synthetic,empirical,baseline,output/'numeric_data')
    syn=pd.read_csv(output/'numeric_data/synthetic_summary.csv')
    emp=pd.read_csv(output/'numeric_data/empirical_summary.csv')
    emp['method']='proposed_multiscale'
    out=output/'generated';figs=output/'figures'
    out.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
      'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,
      'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
      'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.5})
    plot_records=[]
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.65),layout='constrained')
    x=np.arange(len(FOCUS))
    for method,label,color,offset,marker in [('proposed_hybrid','Proposed method',COLORS[0],-.11,'o'),
      ('Milad_published_alpha005',"Milad et al.'s NC-range method",COLORS[1],.11,'s')]:
        selected=syn[(syn.anchor_length==512)&(syn.method==method)].set_index('scenario').loc[[SCENARIOS[j] for j in FOCUS]]
        y=selected.nc_covered_rate.to_numpy()
        low=selected.nc_covered_cp95_lower.to_numpy();high=selected.nc_covered_cp95_upper.to_numpy()
        axes[0].errorbar(x+offset,y,yerr=np.array([y-low,high-y]),fmt=marker,color=color,
          capsize=3,markersize=4,label=label,linestyle='none')
        for j,(_,r) in zip(FOCUS,selected.iterrows()):
            plot_records.append({'figure':'synthetic','panel':'NC coverage','scenario':'S'+str(j),
             'method':method,'anchor_length':512,'value':float(r.nc_covered_rate),
             'lower':float(r.nc_covered_cp95_lower),'upper':float(r.nc_covered_cp95_upper)})
    axes[0].axhline(.975,color='#666666',linestyle='--',linewidth=.9,label='97.5% NC guarantee')
    axes[0].set(xticks=x,xticklabels=['S'+str(j) for j in FOCUS],ylim=(.45,1.055),
      xlabel='Scenario (512-block target window)',ylabel='NC coverage',title='(a) NC coverage')
    axes[0].legend(loc='lower left',frameon=False)
    axes[0].grid(axis='y',alpha=.18)
    for j,color in zip(FOCUS,COLORS):
        g=syn[(syn.method=='proposed_hybrid')&(syn.scenario==SCENARIOS[j])].sort_values('anchor_length')
        axes[1].plot(g.anchor_length,g.hhi_width_mean,marker='o',markersize=3.4,color=color,label='S'+str(j))
        plot_records += [{'figure':'synthetic','panel':'HHI width','scenario':'S'+str(j),
          'anchor_length':int(r.anchor_length),'value':float(r.hhi_width_mean)} for _,r in g.iterrows()]
    axes[1].set_xscale('log',base=2)
    axes[1].set(xticks=[128,256,512,1024,2048],xticklabels=['128','256','512','1024','2048'],
      xlabel='Target-window block count',ylabel='Mean HHI confidence-interval width',title='(b) Proposed method: HHI width')
    axes[1].legend(frameon=False,ncol=2,loc='upper right');axes[1].grid(alpha=.18)
    figure(fig,figs/'compact_synthetic')
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.65),layout='constrained')
    for ch,color in zip(CHAIN_NAMES,COLORS):
        g=emp[(emp.chain==ch)&(emp.method=='proposed_multiscale')].sort_values('anchor_length')
        for i,col in enumerate(('duration_days_median','hhi_width_median')):
            axes[i].plot(g.anchor_length,g[col],marker='o',markersize=3.5,color=color,label=CHAIN_NAMES[ch])
            plot_records += [{'figure':'empirical','panel':col,'chain':ch,'anchor_length':int(r.anchor_length),
                              'value':float(r[col])} for _,r in g.iterrows()]
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([128,256,512,1024,2048,4096],['128','256','512','1024','2048','4096'])
        ax.set_xlabel('Target-window block count');ax.grid(alpha=.18)
    axes[0].set_yscale('log');axes[0].set(ylabel='Median averaging duration (days)',title='(a) Averaging duration')
    axes[1].set(ylabel='Median HHI confidence-interval width',title='(b) Proposed method: HHI width')
    axes[0].legend(loc='upper left',frameon=False)
    figure(fig,figs/'compact_empirical')
    table_files=[]
    def table(name, rows):
        source=RELEASE/'publication_templates/generated'/name
        template=source.read_text()
        start=template.index('\\midrule')+len('\\midrule')
        end=template.index('\\bottomrule',start)
        rendered=template[:start]+'\n'+'\n'.join(rows)+'\n'+template[end:]
        # Compare full cells, not just displayed precisions individually.
        oldcells=[' '.join(x.split()) for x in template[start:end].strip().splitlines()]
        newcells=[' '.join(x.split()) for x in rows]
        assert oldcells==newcells,(name,oldcells,newcells)
        write(out/name,rendered);table_files.append(name)
    rr=[]
    exact=pd.read_csv(output/'numeric_data/table2_data.csv')
    for m,n,nc in [(2,100,2),(4,100,3)]:
        g=exact[(exact.entities==m)&(exact.n==n)]
        v=float(g[g.implementation=='published'].coverage.iloc[0])*100
        d=float(g[g.implementation=='inclusive_endpoint_diagnostic'].coverage.iloc[0])*100
        p='$(1/2,1/2)$' if m==2 else '$(1/4,1/4,1/4,1/4)$'
        rr.append(f'{p} & {n} & {nc} & {v:.4f} & {d:.2f} '+r'\\')
    table('exact.tex',rr)
    descriptions={0:'Stationary reference',3:'Within-window concentration',7:'Dominant entrant',8:'Strict-majority boundary'}
    r4=pd.read_csv(output/'numeric_data/table5_data.csv')
    rr=[]
    for j in FOCUS:
        r=r4[r4.scenario.eq('S'+str(j))].iloc[0]
        rr.append(f'S{j} & {descriptions[j]} & {int(r.true_nc)} & {r.true_hhi:.4f} & {int(r.joint_covered)}/600 & {r.nc_size_mean:.3f} & {r.hhi_width_mean:.4f} '+r'\\')
    table('compact_scenarios.tex',rr)
    r5=pd.read_csv(output/'numeric_data/table6_data.csv');rr=[]
    for ch in CHAIN_NAMES:
        for a in [512,4096]:
            r=r5[r5.chain.eq(ch)&r5.anchor_length.eq(a)].iloc[0]
            rr.append(f'{CHAIN_NAMES[ch]} & {a} & {r.duration_days_median:.4f} & {int(r.nc_lower_median)} & {int(r.nc_upper_median)} & {int(r.singleton_months)} & {r.hhi_width_median:.4f} & {r.hhi_width_q95:.4f} '+r'\\')
    table('compact_empirical_results.tex',rr)
    r2=pd.read_csv(output/'numeric_data/tableS2_data.csv');rr=[]
    for _,r in r2.iterrows():
        rr.append(f'{int(r.n)} & {100*r.mean_coverage:.2f} & {100*r.median_coverage:.2f} & {int(r.raw_count)}/240 & {int(r.fwer_count)}/240 '+r'\\')
    table('compact_grid_audit.tex',rr)
    ds=pd.read_csv(output/'numeric_data/tableS4_data.csv');rr=[]
    for ch in CHAIN_NAMES:
        r=ds[ds.chain.eq(ch)].iloc[0]
        rr.append(f'{CHAIN_NAMES[ch]} & {r.period_records:,} & {r.period_labels_observed} & {100*r.period_unknown_prefix_fraction:.2f} '+r'\\')
    table('compact_empirical_sources.tex',rr)
    for name in ['comparison_design.tex','compact_all_scenarios.tex','scenario_overview.tex']:
        shutil.copy2(RELEASE/'publication_templates/generated'/name,out/name)
    for name in ['method_overview.pdf','method_overview.svg']:
        shutil.copy2(RELEASE/'publication_templates/figures'/name,figs/name)
    all_scenarios = build_all_scenario_table(synthetic, output, RELEASE/'publication_templates/generated/compact_scenarios.tex')
    assert (output/'generated/all_scenario_results.tex').read_bytes() == (RELEASE/'publication_templates/generated/all_scenario_results.tex').read_bytes()
    table_files.append('all_scenario_results.tex')
    record=dict(all_scenario_table=all_scenarios, status='PASS',numeric_tables=table_files,all_table_cells_match_publication=True,
        regenerated_result_figures=['compact_synthetic.pdf','compact_empirical.pdf'],
        plot_values_checked=64,static_assets_copied=['method_overview.pdf','method_overview.svg','comparison_design.tex','compact_all_scenarios.tex','scenario_overview.tex'],
        note='Reproduces values and publication labels. This numerical companion excludes manuscript PDFs; generated figures may differ in metadata and font placement.',
        outputs_sha256={str(p.relative_to(output)):sha(p) for p in sorted(out.glob('*.tex'))+sorted(figs.glob('*.pdf'))})
    write(output/'ASSET_VERIFICATION.json',json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',type=Path,default=ROOT/'experiments/results/primary')
    p.add_argument('--empirical',type=Path,default=ROOT/'empirical/results_primary')
    p.add_argument('--baseline',type=Path,default=ROOT/'experiments/results')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.synthetic,a.empirical,a.baseline,a.output)
