#!/usr/bin/env python3
"""Paper reproducibility entry point. Outputs go to a new, empty directory.

verify: check supplied sources, reaggregate all current paper claims and assets.
replay: recompute current MT/sharp intervals from saved samples and block arrays.
full: additionally regenerate all synthetic samples, baseline outputs and stationary audit.

All modes are offline after dependencies are installed. Upstream acquisition and
entity attribution are outside the numerical replay boundary. This curated companion retains the numerical inputs needed to reproduce the paper.
"""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
UP=ROOT

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode',choices=['verify','replay','full'])
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();out=a.output.expanduser().resolve()
    if out==ROOT or any(out.is_relative_to(ROOT/x) for x in ('experiments','empirical','tools','documentation','environment')) or out.is_relative_to(ROOT/'publication_templates'):
        p.error('Output must not overwrite supplied sources or manuscript files.')
    if out.exists() and any(out.iterdir()):p.error('Output directory must be new or empty.')
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir()
    env=os.environ.copy();env.update(MPLCONFIGDIR=str(out/'matplotlib-cache'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
    jobs=[];start=time.monotonic()
    def run(label: str, args: list[str|Path]) -> None:
        cmd=[sys.executable]+[str(x) for x in args]
        log=out/'logs'/f'{len(jobs)+1:02d}_{label}.log'
        print(f'[{label}]',flush=True);t=time.monotonic()
        with log.open('w') as f:
            proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        record=dict(step=label,command=cmd,returncode=proc.returncode,log=str(log.relative_to(out)),elapsed_seconds=round(time.monotonic()-t,3))
        jobs.append(record)
        if proc.returncode:
            print(log.read_text()[-8000:],file=sys.stderr)
            raise RuntimeError(f'{label} failed; inspect {log}')
    b=UP/'experiments/results';s=b/'primary';e=UP/'empirical/results_primary'
    result=dict(status='RUNNING',mode=a.mode,python=sys.version,platform=platform.platform(),steps=jobs)
    try:
        run('source_integrity',[ROOT/'tools/check_integrity.py','--report',out/'source_integrity.json'])
        if a.mode=='full':
            b=out/'baseline_regenerated'
            run('regenerate_synthetic',[UP/'experiments/code/inference.py','--output-dir',b,'--repetitions','600'])
            run('regenerate_grid',[UP/'experiments/code/grid_audit.py','--output-dir',b,'--repetitions','2000'])
            run('check_all_generated_samples',[UP/'experiments/tests/verify_artifact.py','--compare-to',b,'--report',out/'input_regeneration_verification.json'])
        if a.mode in ['replay','full']:
            s=out/'synthetic_primary';e=out/'empirical_primary'
            run('recompute_primary_synthetic',[UP/'experiments/code/run_primary.py','--baseline-results',b,'--output',s])
            run('recompute_empirical',[UP/'empirical/code/analyze_primary.py','--output',e])
            run('recompute_attribution_sensitivity',[UP/'empirical/code/attribution_primary.py','--output',e])
            run('compare_all_primary_intervals',[UP/'experiments/tests/verify_primary.py','--compare-synthetic',s,'--compare-empirical',e,'--report',out/'primary_endpoint_verification.json'])
        else:
            run('check_supplied_primary',[UP/'experiments/tests/verify_primary.py','--report',out/'supplied_primary_verification.json'])
        run('publication_claims_and_figures',[ROOT/'tools/render_publication.py','--synthetic',s,'--empirical',e,'--baseline',b,'--output',out/'publication_assets'])
        result.update(status='PASS',elapsed_seconds=round(time.monotonic()-start,3),
            fresh_random_sample_generation=(a.mode=='full'),fresh_primary_interval_computation=(a.mode in ['replay','full']),
            manuscript_modified=False,upstream_acquisition_or_label_assignment_reproduced=False,
            statement='Validation is conditional on the supplied frozen entity partition. No live blockchain acquisition is performed.')
    except Exception as exc:
        result.update(status='FAIL',error=str(exc),elapsed_seconds=round(time.monotonic()-start,3))
        raise
    finally:
        (out/'RUN_STATUS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(f'PASS: {out}/RUN_STATUS.json',flush=True)
if __name__=='__main__': main()
