#!/usr/bin/env python3
"""BTC sensitivity to two documented composite pool labels.

The primary partition is the frozen ``sok_entity`` dictionary.  Alternatives
resolve Bitdeer/NiceHash and Bitdeer/CKPool/NiceHash to one of their constituent
pool labels.  The six maps are fixed here before computing interval outputs.
They examine ambiguity exposed by saved address matching, not a claim that the
named pools share ownership.  Every unknown cluster and all other labels remain
unchanged; all histories and anchors are recounted under each complete map.

No upstream assignment is reconstructed, no external attribution source is
queried, and no cross-partition coverage guarantee is asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'experiments' / 'code'))
from inference import KMAX, RS, candidates, interval_from_counts

COMPOSITES = {
    'Bitdeer/NiceHash': ('Bitdeer', 'NiceHash'),
    'Bitdeer/CKPool/NiceHash': ('Bitdeer', 'CKPool', 'NiceHash'),
}
DECISIONS = ('singleton', 'nc_ge2', 'nc_ge3', 'hhi_below020', 'hhi_above020')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: object) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.17g', lineterminator='\n')


def variants() -> list[tuple[str, dict[str, str]]]:
    result = [('saved_labels', {})]
    for first, second in itertools.product(*COMPOSITES.values()):
        result.append((f'resolve_{first.lower()}_{second.lower()}', dict(zip(COMPOSITES, (first, second)))))
    return result


def partition(labels: list[str], mapping: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    remapped = [mapping.get(x, x) for x in labels]
    dictionary = sorted(set(remapped))
    code = {label: i for i, label in enumerate(dictionary)}
    return np.array([code[label] for label in remapped], dtype=np.int32), dictionary


def finite_endpoint_shift(a: float, b: float) -> float:
    # Equality covers the common +infinity endpoint without inf-inf NaN.
    return 0. if a == b else abs(a - b)


def render_table(summary: pd.DataFrame, pairs: pd.DataFrame, out: Path) -> None:
    """A compact six-window summary; alternatives are summarized as ranges."""
    lines = [r'\begin{table}[!t]', r'\centering', r"\caption{Bitcoin sensitivity to fixed resolutions of the two composite labels. Each row has 36 monthly anchors. $s_u^{\max}$ is the maximum retained unknown-cluster share; $s_c^{\max}$ is the maximum composite-label share. Changed NC counts month--resolution pairs whose NC set differs from the saved-label result (six resolutions per month). $\rho_H^{\max}$ is the largest HHI endpoint displacement divided by the saved-label HHI width.}",
             r'\label{tab:attribution-sensitivity}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.1}',
             r'\begin{tabular}{@{}rrrrr@{}}', r'\toprule',
             r'$a$ & $s_u^{\max}$ (\%) & $s_c^{\max}$ (\%) & Changed NC & $\rho_H^{\max}$ (\%) \\', r'\midrule']
    for a in sorted(summary.anchor_length.unique()):
        b = summary[(summary.anchor_length == a) & (summary.variant == 'saved_labels')].iloc[0]
        p = pairs[pairs.anchor_length == a]
        lines.append(f"{int(a)} & {100*b.unknown_share_max:.3f} & {100*b.composite_share_max:.3f} & {int(p.nc_set_changed.sum())}/216 & {100*p.hhi_endpoint_shift_over_baseline_width.max():.3f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')


def render_narrative(checks: dict, generated: Path, method: str) -> None:
    """Fill authored TeX templates with verified sensitivity result values."""
    # The fixed narrative makes these qualitative claims, so reject stale
    # claims if an altered input ever changes one of the reported decisions.
    assert checks['nc_set_changed_pairs'] == 0 and checks['nc_plugin_changed_pairs'] == 0
    assert all(n == 0 for n in checks['decision_changed_pairs'].values())
    a4096 = checks['anchor4096']
    decisions = a4096['baseline_decisions']
    mantissa, exponent = f"{checks['max_hhi_endpoint_absolute_shift']:.3e}".split('e')
    small_numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
    ncge3 = decisions['nc_ge3']
    values = {
        'ANCHORS': str(checks['anchor_conditions']), 'COMPARISONS': str(checks['paired_comparisons']),
        'SINGLETON4096': str(decisions['singleton']), 'NCGE24096': str(decisions['nc_ge2']),
        'NCGE34096_WORD': small_numbers[ncge3] if 0 <= ncge3 < len(small_numbers) else str(ncge3),
        'HHIBELOW4096': str(decisions['hhi_below020']),
        'HHI_ABSOLUTE_TEX': mantissa + r'\times10^{' + str(int(exponent)) + '}',
        'HHI_RELATIVE_PERCENT': f"{100*checks['max_hhi_endpoint_shift_over_baseline_width']:.3f}",
        'UNKNOWN_PERCENT': f"{100*checks['max_unknown_share']:.3f}",
        'UNKNOWN4096_PERCENT': f"{100*a4096['unknown_share_max']:.3f}",
        'COMPOSITE_PERCENT': f"{100*checks['max_composite_share']:.3f}",
    }
    generated.mkdir(parents=True, exist_ok=True)
    for template in sorted((ROOT/'templates').glob('attribution_*.tex.in')):
        text = template.read_text(encoding='utf-8')
        for key, value in values.items():
            text = text.replace('@@'+key+'@@', value)
        if method != 'hybrid':
            text = text.replace('proposed combined NC and HHI outputs', method+'-component NC and HHI outputs')
        assert '@@' not in text, template
        (generated/template.name.removesuffix('.in')).write_text(text, encoding='utf-8')


def run(data: Path, output: Path, design: Path, method: str, generated: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(design.read_text())
    meta = json.loads((data / 'btc_input_manifest.json').read_text())
    assert sha(data / 'btc_blocks.npz') == meta['compact_npz_sha256']
    assert sha(data / 'btc_labels.json') == meta['compact_labels_sha256']
    assert tuple(cfg['history_lengths']) == RS
    spec = next(c for c in cfg['chains'] if c['chain'] == 'btc')
    months = pd.period_range(spec['first_month'], spec['last_month'], freq='M').astype(str).tolist()
    labels = json.loads((data / 'btc_labels.json').read_text())['labels']
    with np.load(data / 'btc_blocks.npz', allow_pickle=False) as d:
        heights = d['height'].copy()
        times = d['timestamp_utc_s'].copy()
        entity = d['entity_code'].copy()
    assert np.all(np.diff(heights) == 1)
    for label, constituents in COMPOSITES.items():
        assert label in labels and all(c in labels for c in constituents)
    unknown_mask = np.array([x.startswith('unknown_cluster_') for x in labels])
    composite_mask = np.array([x in COMPOSITES for x in labels])
    maps = variants()
    mapped = {}
    for variant, mapping in maps:
        code, dic = partition(labels, mapping)
        mapped[variant] = (code[entity], dic)
        assert all(dic[code[i]] == label for i, label in enumerate(labels) if unknown_mask[i])
    map_rows = [{'variant': v, 'source_label': s, 'target_label': t} for v, m in maps for s, t in m.items()]
    write_csv(output / 'attribution_sensitivity_mappings.csv', map_rows)
    rows = []
    condition_rows = []
    for month in months:
        period = pd.Period(month, freq='M')
        start_sec = int(period.start_time.tz_localize('UTC').timestamp())
        end_sec = int((period + 1).start_time.tz_localize('UTC').timestamp())
        select = np.flatnonzero((times >= start_sec) & (times < end_sec))
        assert len(select)
        end = int(select[-1])
        for a in cfg['anchor_lengths']:
            begin = end - a + 1
            assert begin >= max(RS)
            raw = np.bincount(entity[begin:end+1], minlength=len(labels))
            hraw = np.bincount(entity[begin-max(RS):begin], minlength=len(labels))
            common = {
                'chain': 'btc', 'month': month, 'anchor_length': a,
                'report_height': int(heights[end]), 'anchor_first_height': int(heights[begin]),
                'history_first_height': int(heights[begin-max(RS)]),
                'unknown_blocks': int(raw[unknown_mask].sum()), 'unknown_share': float(raw[unknown_mask].sum()/a),
                'composite_blocks': int(raw[composite_mask].sum()), 'composite_share': float(raw[composite_mask].sum()/a),
                'history_unknown_blocks': int(hraw[unknown_mask].sum()),
                'history_unknown_share': float(hraw[unknown_mask].sum()/max(RS)),
                'history_composite_blocks': int(hraw[composite_mask].sum()),
                'history_composite_share': float(hraw[composite_mask].sum()/max(RS)),
            }
            condition_rows.append(common)
            for variant, mapping in maps:
                eid, dictionary = mapped[variant]
                m = max(len(dictionary), KMAX)
                counts = np.bincount(eid[begin:end+1], minlength=m).astype(np.int64)
                history = {r: np.bincount(eid[begin-r:begin], minlength=m).astype(np.int64) for r in RS}
                assert int(counts.sum()) == a and all(int(c.sum()) == r for r, c in history.items())
                family = candidates(history, 'multiscale')
                if method == 'history':
                    result = interval_from_counts(counts, family)
                else:
                    result = interval_from_counts(counts, family, lower_method=method)
                # Bands are available in the primary empirical output; this
                # focused file retains endpoints and the complete map needed
                # to regenerate every sensitivity band and candidate family.
                result = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
                cumulative = np.cumsum(np.sort(counts)[::-1])
                nc_plugin = int(np.flatnonzero(2*cumulative > a)[0]+1)
                hhi_plugin = float(np.sum(counts*counts)/(a*a))
                nonempty = not bool(result['nc_empty'])
                row = {
                    **common, 'variant': variant, 'lower_method': method,
                    'anchor_observed_labels': int(np.sum(counts > 0)),
                    'nc_plugin': nc_plugin, 'hhi_plugin': hhi_plugin, **result,
                    'singleton': int(nonempty and result['nc_size'] == 1),
                    'nc_ge2': int(nonempty and result['nc_lower'] >= 2),
                    'nc_ge3': int(nonempty and result['nc_lower'] >= 3),
                    'hhi_below020': int(result['hhi_upper'] < .20),
                    'hhi_above020': int(result['hhi_lower'] > .20),
                }
                rows.append(row)
    frame = pd.DataFrame(rows)
    baseline = frame[frame.variant == 'saved_labels'].set_index(['month', 'anchor_length'])
    pairs = []
    for row in rows:
        if row['variant'] == 'saved_labels':
            continue
        base = baseline.loc[(row['month'], row['anchor_length'])]
        endpoint_shift = max(abs(row['hhi_lower'] - base.hhi_lower), abs(row['hhi_upper'] - base.hhi_upper))
        pair = {k: row[k] for k in ['chain', 'month', 'anchor_length', 'report_height', 'variant', 'lower_method']}
        pair.update({
            'nc_set_changed': int(row['nc_empty'] != base.nc_empty or row['nc_lower'] != base.nc_lower or row['nc_upper'] != base.nc_upper),
            'nc_lower_shift': float(row['nc_lower'] - base.nc_lower),
            'nc_upper_absolute_shift': finite_endpoint_shift(row['nc_upper'], base.nc_upper),
            'nc_plugin_shift': int(row['nc_plugin'] - base.nc_plugin),
            'hhi_plugin_shift': float(row['hhi_plugin'] - base.hhi_plugin),
            'hhi_lower_shift': float(row['hhi_lower'] - base.hhi_lower),
            'hhi_upper_shift': float(row['hhi_upper'] - base.hhi_upper),
            'hhi_width_shift': float(row['hhi_width'] - base.hhi_width),
            'baseline_hhi_width': float(base.hhi_width),
            'hhi_endpoint_absolute_shift': float(endpoint_shift),
            'hhi_endpoint_shift_over_baseline_width': float(endpoint_shift / base.hhi_width),
            'hhi_plugin_shift_over_baseline_width': float(abs(row['hhi_plugin'] - base.hhi_plugin) / base.hhi_width),
        })
        pair.update({event+'_changed': int(row[event] != base[event]) for event in DECISIONS})
        pairs.append(pair)
    paired = pd.DataFrame(pairs)
    summaries = []
    for (a, variant), g in frame.groupby(['anchor_length', 'variant'], sort=True):
        result = {'chain': 'btc', 'anchor_length': int(a), 'variant': variant, 'lower_method': method, 'months': len(g),
                  'unknown_share_mean': float(g.unknown_share.mean()), 'unknown_share_max': float(g.unknown_share.max()),
                  'composite_share_mean': float(g.composite_share.mean()), 'composite_share_max': float(g.composite_share.max()),
                  'nc_empty_months': int(g.nc_empty.sum()), 'nc_unbounded_months': int(np.isinf(g.nc_upper).sum()),
                  'hhi_width_median': float(g.hhi_width.median()), 'hhi_plugin_median': float(g.hhi_plugin.median()),
                  'hhi_fallback_months': int(g.hhi_fallback.sum())}
        result.update({event+'_months': int(g[event].sum()) for event in DECISIONS})
        summaries.append(result)
    summary = pd.DataFrame(summaries)
    write_csv(output / 'attribution_sensitivity_conditions.csv', condition_rows)
    write_csv(output / 'attribution_sensitivity_intervals.csv', frame)
    write_csv(output / 'attribution_sensitivity_paired.csv', paired)
    write_csv(output / 'attribution_sensitivity_summary.csv', summary)
    expected = len(months) * len(cfg['anchor_lengths'])
    assert len(rows) == expected * 7 and len(pairs) == expected * 6
    # Every alternative is a deterministic coarsening of the frozen partition.
    # This gives these checks independently of any confidence construction.
    assert (paired.nc_plugin_shift <= 0).all()
    assert (paired.hhi_plugin_shift >= -1e-14).all()
    assert (frame.unknown_blocks == frame.unknown_share * frame.anchor_length).all()
    evidence = [
        'empirical/provenance/mapping_reference.py',
        'empirical/provenance/README.md',
        'empirical/data/btc_labels.json',
    ]
    dump(output / 'attribution_sensitivity_design.json', {
        'chain': 'btc', 'first_month': spec['first_month'], 'last_month': spec['last_month'],
        'anchor_lengths': cfg['anchor_lengths'], 'history_lengths': list(RS), 'history_variant': 'multiscale',
        'lower_method': method, 'baseline': 'saved_labels',
        'variants': [{'name': name, 'map': mapping} for name, mapping in maps],
        'choice_rule': 'All 2 x 3 fixed whole-label resolutions of the two whitelisted composite labels; no output-dependent choice.',
        'interpretation': 'An ambiguity check within saved public-attribution information, not independently verified organizational attribution, raw remapping, or a comparison of external sources.',
        'unknown_policy': 'Retain every distinct unknown_cluster label and every block in every alternative.',
        'evidence': {p: sha(ROOT.parent / p) for p in evidence},
        'scope_limitations': [
            'The final upstream augmentation code and complete evidence dictionary are unavailable.',
            'Slash concatenation in the saved address-mapping code records multiple pool candidates, not a corporate merger.',
            'The two labels are explicitly whitelisted; other names containing punctuation are not parsed or split.',
            'A resolution is fixed for the entire composite label across every history and anchor; individual-block resolutions are not enumerated.',
            'Separate interval statements condition on each fixed partition. No simultaneous confidence claim across alternative partitions or calendar reports is made.',
        ],
        'design_sha256': sha(design), 'input_npz_sha256': sha(data/'btc_blocks.npz'),
        'inference_sha256': sha(ROOT.parent/'experiments/code/inference.py'),
    })
    main4096 = summary[(summary.anchor_length == 4096) & (summary.variant == 'saved_labels')].iloc[0]
    p4096 = paired[paired.anchor_length == 4096]
    checks = {
        'anchor_conditions': expected, 'partitions': 7, 'interval_outputs': len(frame), 'paired_comparisons': len(paired),
        'nc_set_changed_pairs': int(paired.nc_set_changed.sum()),
        'distinct_conditions_with_nc_set_change': int(paired[paired.nc_set_changed == 1].groupby(['month', 'anchor_length']).ngroups),
        'nc_plugin_changed_pairs': int((paired.nc_plugin_shift != 0).sum()),
        'decision_changed_pairs': {event: int(paired[event+'_changed'].sum()) for event in DECISIONS},
        'max_hhi_endpoint_absolute_shift': float(paired.hhi_endpoint_absolute_shift.max()),
        'max_hhi_endpoint_shift_over_baseline_width': float(paired.hhi_endpoint_shift_over_baseline_width.max()),
        'max_hhi_plugin_shift': float(paired.hhi_plugin_shift.max()),
        'max_unknown_share': float(baseline.unknown_share.max()),
        'max_composite_share': float(baseline.composite_share.max()),
        'anchor4096': {
            'unknown_share_max': float(main4096.unknown_share_max),
            'composite_share_max': float(main4096.composite_share_max),
            'baseline_decisions': {event: int(main4096[event+'_months']) for event in DECISIONS},
            'nc_set_changed_pairs': int(p4096.nc_set_changed.sum()),
            'max_hhi_endpoint_absolute_shift': float(p4096.hhi_endpoint_absolute_shift.max()),
            'max_hhi_endpoint_shift_over_baseline_width': float(p4096.hhi_endpoint_shift_over_baseline_width.max()),
        },
        'counts_conserved': True, 'unknown_labels_unchanged': True,
        'coarsening_plugin_checks_passed': True,
    }
    dump(output / 'attribution_sensitivity_checks.json', checks)
    render_table(summary, paired, generated / 'attribution_sensitivity_table.tex')
    render_narrative(checks, generated, method)
    print(json.dumps(checks, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data')
    p.add_argument('--output', type=Path, default=ROOT/'results')
    p.add_argument('--design', type=Path, default=ROOT/'design.json')
    p.add_argument('--lower-method', choices=['history', 'hybrid', 'direct'], default='hybrid')
    p.add_argument('--generated', type=Path, default=ROOT.parent/'manuscript/generated')
    args = p.parse_args()
    run(args.data, args.output, args.design, args.lower_method, args.generated)


if __name__ == '__main__':
    main()
