# ============================================================================
# adapter_table.py -- the paired comparisons of the article (Tables 1, 5 and 6), from saved results
#   Reads checkpoints/results_<row>_<task>_<subset>.npz, written by train_rba.py for every trained row,
#   and prints for each setting: the adapter table, the paired comparisons (Holm-corrected Wilcoxon),
#   the size-accuracy front, the 2 x 2 factorial of the two branches, and the claims with their verdicts.
#   Rows that are not trained are skipped, so it can be run at any time.  python adapter_table.py
#   RBA = Read-out Basis Adapter (rows H and H1; called "HQ read-out" in the code):
#     m_q = Up(<w_q, F>) + <U w_q, G(x)>,  U zero-initialised (rank 32), G = image-only maps at stride 2,
#     plus a rank-1 LoRA branch in the encoder (row H1 = RBA; row H = its read-out branch alone).
#   Primary metric: Dice@4 at t = 0.50 on the 50 test images, paired by image.
#   Rules fixed BEFORE the runs, so the result cannot be tuned:
#   25 Sep 2026
#     beats   : B - A > 0 and Wilcoxon p < 0.05
#     matches : not significant and the whole 95% CI of B - A inside +-0.003
#     An HQ row must also beat B1 (LoRA r=1, about the same size), otherwise
#     there is no claim for the HQ read-out.
#   30 Sep 2026
#     Two rows are only compared if they were trained for the same number of
#     epochs (read from each file).
#   1 Oct 2026, before any GT_Vein run
#     - The p-values are Holm-corrected over all comparisons made in one task;
#       beats / worse use the corrected p (the raw p is printed next to it).
#     - A claim is written only if it holds on every task it was tested on.
#     - RiDA becomes Paper 1's method only if it beats or matches H1 (it has
#       fewer adapter parameters). Otherwise H1 stays the method.
#   2 Oct 2026: RiDA lost to H1 on GT_LargeVessel. If it wins on GT_Capillary only, the
#     REPLICATION block reports it as task-dependent: RiDA is then the thin-vessel adapter,
#     not Paper 1's general method.
#   9 Oct 2026, before rows B, AF and B_lora_hq were trained
#     - Comparisons are grouped in two families, RBA (the main claims) and RiDA (its ablation);
#       the Holm correction runs within each family and setting.
#     - "RBA beats every encoder adapter": H1 beats B1, B (LoRA r=16), C (Conv-LoRA r=16) and
#       AF (AdaptFormer, same size as RBA). If B or AF beats or matches H1, that is reported,
#       and the claim becomes "plug-in / efficiency" only.
#     - "RBA is a plug-in": its read-out improves every encoder adapter it is added to
#       (B1 -> H1, C -> D, and B -> B_lora_hq if that optional row is trained).
#   2 Oct 2026, before row A was trained (FACTORIAL block)
#     Each design is a 2 x 2: encoder path (no / yes) x read-out path (no / yes).
#       RBA design : A (neither), B1 (encoder), H (read-out), H1 (both)
#       RiDA design: A (neither), R_no_readout (encoder), R_no_encoder (read-out), R_rida (both)
#     interaction per image = (both - encoder only) - (read-out only - neither)
#       complementary : interaction > 0 and Wilcoxon p (Holm over the designs) < 0.05
#       overlapping   : interaction < 0 and p < 0.05
#       additive      : not significant and the whole 95% CI inside +-0.003
# ============================================================================
import os
import json
import math

import numpy as np


class Config:
    CHECKPOINT_DIR = "checkpoints"
    SETTINGS = [('GT_LargeVessel', 'OCTA_3mm'), ('GT_Vein', 'OCTA_3mm'), ('GT_Artery', 'OCTA_3mm'),
                ('GT_FAZ', 'OCTA_3mm'), ('GT_Capillary', 'OCTA_3mm'),
                ('GT_LargeVessel', 'OCTA_6mm'), ('GT_Vein', 'OCTA_6mm'), ('GT_Artery', 'OCTA_6mm'),
                ('GT_FAZ', 'OCTA_6mm'), ('GT_RoseVessel', 'ROSE_O'), ('GT_RoseFAZ', 'ROSE_O')]
    ROWS = [                                   # (row, what it is)
        ('A_decoder_only',  'no adapter (decoder only)'),
        ('B1_lora1_ep100',  'LoRA r=1'),
        ('B_lora',          'LoRA r=16'),
        ('C_convlora',      'Conv-LoRA r=16'),
        ('AF_adaptformer',  'AdaptFormer (RBA size)'),
        ('H_hq_only_ep100', 'RBA read-out branch only'),
        ('H1_lora1_hq',     'RBA (ours)'),
        ('D_convlora_hq',   'Conv-LoRA r=16 + RBA read-out'),
        ('B_lora_hq',       'LoRA r=16 + RBA read-out'),
        ('R_rida',          'RiDA (full)'),
        ('R_no_ridge',      'RiDA, no ridge operator'),
        ('R_no_readout',    'RiDA, no read-out path'),
        ('R_no_encoder',    'RiDA, no encoder path'),
    ]
    # rows planned in each setting. A row that is not planned in a setting gets no "not trained yet"
    # line there, and a claim that needs it is marked "not run" instead of "waiting".
    # A row you train anyway is always used.
    _ALL = {'A_decoder_only', 'B1_lora1_ep100', 'B_lora', 'C_convlora', 'AF_adaptformer', 'H_hq_only_ep100',
            'H1_lora1_hq', 'D_convlora_hq', 'B_lora_hq', 'R_rida', 'R_no_ridge', 'R_no_readout', 'R_no_encoder'}
    PLANNED = {('GT_LargeVessel', 'OCTA_3mm'): _ALL,                  # rows run in the article, per setting
               ('GT_Vein', 'OCTA_3mm'): {'A_decoder_only', 'B1_lora1_ep100', 'C_convlora', 'H_hq_only_ep100',
                                         'H1_lora1_hq', 'D_convlora_hq'},
               ('GT_Artery', 'OCTA_3mm'): {'C_convlora', 'H_hq_only_ep100', 'H1_lora1_hq'},
               ('GT_FAZ', 'OCTA_3mm'): {'C_convlora', 'H_hq_only_ep100', 'H1_lora1_hq'},
               ('GT_Capillary', 'OCTA_3mm'): {'H1_lora1_hq', 'R_rida', 'R_no_ridge'},
               ('GT_LargeVessel', 'OCTA_6mm'): {'C_convlora', 'H1_lora1_hq'},
               ('GT_Vein', 'OCTA_6mm'): {'C_convlora', 'H1_lora1_hq'},
               ('GT_Artery', 'OCTA_6mm'): {'H1_lora1_hq'},
               ('GT_FAZ', 'OCTA_6mm'): {'H1_lora1_hq'},
               ('GT_RoseVessel', 'ROSE_O'): {'C_convlora', 'H1_lora1_hq'},
               ('GT_RoseFAZ', 'ROSE_O'): {'C_convlora', 'H1_lora1_hq'}}
    OPTIONAL_ROWS = {'B_lora_hq'}              # used when trained; never waited for
    KNOWN_EPOCHS = {('GT_LargeVessel', 'OCTA_3mm', 'D_convlora_hq'): 100}   # the first D file carries no Config
    # adapter sizes from the RiDA design note, used only to cross-check the files
    KNOWN_ADAPTER_PARAMS = {'R_rida': 292_420, 'R_no_ridge': 290_460,
                            'R_no_readout': 271_032, 'R_no_encoder': 29_452,
                            'AF_adaptformer': 451_488}
    COMPARISONS = [                            # (A, B, family): paired B - A
        # RBA family: the main claims
        ('B1_lora1_ep100',  'H1_lora1_hq',     'RBA'),   # RBA vs LoRA r=1 (= the read-out added to LoRA r=1)
        ('B_lora',          'H1_lora1_hq',     'RBA'),   # RBA vs LoRA r=16
        ('C_convlora',      'H1_lora1_hq',     'RBA'),   # RBA vs Conv-LoRA r=16
        ('AF_adaptformer',  'H1_lora1_hq',     'RBA'),   # RBA vs AdaptFormer of the same size
        ('D_convlora_hq',   'H1_lora1_hq',     'RBA'),   # rank-1 vs rank-16 encoder branch, both with the read-out
        ('C_convlora',      'D_convlora_hq',   'RBA'),   # the read-out added to Conv-LoRA
        ('B_lora',          'B_lora_hq',       'RBA'),   # the read-out added to LoRA r=16 (optional row)
        ('C_convlora',      'H_hq_only_ep100', 'RBA'),   # the read-out branch alone vs Conv-LoRA
        ('B1_lora1_ep100',  'H_hq_only_ep100', 'RBA'),   #   and vs LoRA r=1, about the same size
        # RiDA family: the alternative read-out design and its ablation
        ('H1_lora1_hq',     'R_rida',          'RiDA'),  # RiDA vs RBA
        ('C_convlora',      'R_rida',          'RiDA'),
        ('B1_lora1_ep100',  'R_rida',          'RiDA'),
        ('B1_lora1_ep100',  'R_no_ridge',      'RiDA'),  # a second, 10x smaller read-out on LoRA r=1 (no ridge)
        ('R_no_ridge',      'R_rida',          'RiDA'),  # does the ridge operator matter?
        ('R_no_readout',    'R_rida',          'RiDA'),  # does the read-out path matter?
        ('R_no_encoder',    'R_rida',          'RiDA'),  # does the encoder path matter?
        ('B1_lora1_ep100',  'R_no_readout',    'RiDA'),  # the ridge inside the encoder path alone vs plain LoRA r=1
        ('H_hq_only_ep100', 'R_no_encoder',    'RiDA'),  # the two read-outs alone: RiDA's (29 K) vs RBA's (0.20 M)
    ]
    FACTORIAL = [                              # (design, neither, encoder only, read-out only, both)
        ('RBA',  'A_decoder_only', 'B1_lora1_ep100', 'H_hq_only_ep100', 'H1_lora1_hq'),
        ('RiDA', 'A_decoder_only', 'R_no_readout',   'R_no_encoder',    'R_rida'),
    ]
    MARGIN = 0.003
    P_LEVEL = 0.05
    MODEL_PARAMS = 346.1e6                     # SEEM focal-L total, used if a file has no count


cfg = Config()


def is_optional(setting, row):
    return row in cfg.OPTIONAL_ROWS or row not in cfg.PLANNED.get(setting, set())


def short(setting):
    task, subset = setting
    return f"{task.replace('GT_', '')}-{subset.replace('OCTA_', '')}"


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def results_path(row, setting):
    task, subset = setting
    return os.path.join(cfg.CHECKPOINT_DIR, f"results_{row}_{task}_{subset}.npz")


def adapter_params(row, params):
    """Adapter size = trainable - decoder (same for every trainer version); else the explicit parts."""
    if 'trainable' in params and 'decoder' in params:
        n = int(params['trainable']) - int(params['decoder'])
    else:
        n = int(params.get('encoder', 0)) + int(params.get('hq', 0)) + int(params.get('click_adapter', 0))
    known = cfg.KNOWN_ADAPTER_PARAMS.get(row)
    note = None
    if known is not None and params and n != known:
        note = f"{row}: the file reports {n:,} adapter params, the design note says {known:,}"
    return n, note


def load_row(row, setting):
    path = results_path(row, setting)
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    k = int(z['k_report']) if 'k_report' in z.files else 4
    params = json.loads(str(z['params'])) if 'params' in z.files else {}
    saved_cfg = json.loads(str(z['cfg'])) if 'cfg' in z.files else {}
    epochs = saved_cfg.get('EPOCHS', cfg.KNOWN_EPOCHS.get((*setting, row)))
    n_adapt, note = adapter_params(row, params)
    c05, cts = z['curves_giir_t05'], z['curves_giir_tstar']
    crand = z['curves_random_tstar'] if 'curves_random_tstar' in z.files else np.zeros(0)
    cl = z['cldice_t05'] if 'cldice_t05' in z.files else np.zeros(0)
    return dict(row=row, sids=[str(s) for s in z['sids']], k=k, epochs=epochs, seed=saved_cfg.get('SEED'),
                curves05=c05, dice1=c05[:, 0], dicek=c05[:, k - 1], dicek_tstar=cts[:, k - 1],
                giir_tstar_k=cts[:, k - 1], random_tstar_k=crand[:, k - 1] if crand.size else None,
                t_star=float(z['t_star']), cldice=cl if cl.size else None,
                adapter=n_adapt, total=float(params.get('total', 0)) or cfg.MODEL_PARAMS, note=note)


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def paired_d(d):
    """Mean of per-image values d, 95% CI (t), images > 0, two-sided Wilcoxon p."""
    d = np.asarray(d, float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    try:
        from scipy import stats
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float('nan')
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ImportError:
        half, p = 1.96 * sd / math.sqrt(max(n, 1)), float('nan')
    return dict(diff=mean, lo=mean - half, hi=mean + half, improved=int((d > 0).sum()), n=n, p=p)


def paired(a, b):
    """B - A over images."""
    return paired_d(np.asarray(b, float) - np.asarray(a, float))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [1.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def verdict(s, p):
    if p < cfg.P_LEVEL:
        return 'beats' if s['diff'] > 0 else 'worse'
    if -cfg.MARGIN <= s['lo'] and s['hi'] <= cfg.MARGIN:
        return 'matches'
    return 'unclear'


def noc(curve, tau):
    hits = np.where(np.asarray(curve) >= tau)[0]
    return (int(hits[0]) + 1, False) if len(hits) else (len(curve), True)


def size_text(n_a, n_b):
    if n_a == n_b:
        return "same size"
    if n_b == 0 or n_a == 0:
        return f"{n_b / 1e6:.2f}M vs {n_a / 1e6:.2f}M adapter params"
    return f"{n_a / n_b:.1f}x fewer adapter params" if n_b < n_a else f"{n_b / n_a:.1f}x more adapter params"


def comparable(rows, names):
    """None if the rows can be paired; otherwise the reason they cannot."""
    first = rows[names[0]]
    for nm in names[1:]:
        r = rows[nm]
        if r['sids'] != first['sids']:
            return "the files hold different test images"
    eps = {rows[nm]['epochs'] for nm in names}
    if None in eps or len(eps) > 1:
        return "trained for different numbers of epochs (" + ", ".join(
            f"{nm} {rows[nm]['epochs']}" for nm in names) + ")"
    return None


# ----------------------------------------------------------------------------
# one task
# ----------------------------------------------------------------------------
def task_table(setting):
    task, subset = setting
    rows, notes = {}, []
    print(f"\n{'=' * 100}\nADAPTER TABLE  {task}  {subset}   (GIIR clicks, t = 0.50 unless marked)\n{'=' * 100}")
    print(f"{'row':16s} {'adapter':29s} {'params':>7s} {'% SEEM':>6s} {'ep':>4s} {'Dice@1':>7s} {'Dice@4':>7s} "
          f"{'clDice':>7s}  {'Dice@4 at t*':>16s}  {'NoC85/90':>11s}  {'clicks 1->4':>17s}  {'GIIR-random@4':>17s}")
    for row, what in cfg.ROWS:
        r = load_row(row, setting)
        if r is None:
            if not is_optional(setting, row):
                print(f"{row:16s} {what:29s} not trained yet ({os.path.basename(results_path(row, setting))})")
            continue
        rows[row] = r
        if r['note']:
            notes.append(r['note'])
        cl = f"{r['cldice'].mean():.4f}" if r['cldice'] is not None else "-"
        g = paired(r['dice1'], r['dicek'])
        n85 = np.mean([noc(c, 0.85)[0] for c in r['curves05']])
        n90 = np.mean([noc(c, 0.90)[0] for c in r['curves05']])
        if r['random_tstar_k'] is not None:
            gr = paired(r['random_tstar_k'], r['giir_tstar_k'])
            gr_txt = f"{gr['diff']:+.4f} (p={gr['p']:.0e})"
        else:
            gr_txt = "-"
        ep = str(r['epochs']) if r['epochs'] is not None else '?'
        print(f"{row:16s} {what:29s} {r['adapter'] / 1e6:6.2f}M {100 * r['adapter'] / r['total']:5.2f}% {ep:>4s} "
              f"{r['dice1'].mean():7.4f} {r['dicek'].mean():7.4f} {cl:>7s}  "
              f"{r['dicek_tstar'].mean():.4f} (t*={r['t_star']:.2f})  {n85:5.2f}/{n90:5.2f}  "
              f"{g['diff']:+.4f} (p={g['p']:.0e})  {gr_txt:>17s}")
    print("  clicks 1->4 = Dice@4 - Dice@1 at t = 0.50;  GIIR-random@4 = GIIR minus the random-in-error control at t*;"
          "  both two-sided Wilcoxon p")
    seeds = {r['seed'] for r in rows.values() if r['seed'] is not None}
    if len(seeds) > 1:
        notes.append(f"rows were trained with different seeds {sorted(seeds)}: compare like with like")
    for n in notes:
        print(f"  note: {n}")

    # paired comparisons, Holm-corrected within each family
    tested, skipped = [], []
    for a, b, fam in cfg.COMPARISONS:
        if a not in rows or b not in rows:
            continue
        why = comparable(rows, (a, b))
        if why:
            skipped.append((a, b, why))
            continue
        tested.append((a, b, fam, paired(rows[a]['dicek'], rows[b]['dicek'])))
    results = {}
    for fam in ('RBA', 'RiDA'):
        group = [t for t in tested if t[2] == fam]
        if not group:
            continue
        adj = holm([s_['p'] for _, _, _, s_ in group])
        print(f"\nPAIRED COMPARISONS, {fam} family: Dice@4 (t = 0.50), B - A over the same test images, "
              f"{len(group)} test{'s' if len(group) != 1 else ''} Holm-corrected")
        print(f"  rule: beats / worse = Holm p < {cfg.P_LEVEL};  matches = 95% CI inside +-{cfg.MARGIN};"
              f"  only rows trained for the same number of epochs")
        for (a, b, _, s_), p_adj in zip(group, adj):
            v = verdict(s_, p_adj)
            results[(a, b)] = (s_, v)
            print(f"  {b:15s} vs {a:15s}: {s_['diff']:+.4f} [{s_['lo']:+.4f}, {s_['hi']:+.4f}]  "
                  f"improved {s_['improved']:2d}/{s_['n']}  p={s_['p']:.1e} (Holm {p_adj:.1e})  -> {v.upper():8s} "
                  f"({size_text(rows[a]['adapter'], rows[b]['adapter'])})")
    for a, b, why in skipped:
        print(f"  {b:15s} vs {a:15s}: NOT COMPARABLE: {why}")
    waiting = [(a, b) for a, b, _ in cfg.COMPARISONS if (a not in rows or b not in rows)
               and not any(is_optional(setting, x) and x not in rows for x in (a, b))]
    if waiting:
        print(f"  waiting for: " + ", ".join(f"{b} vs {a}" for a, b in waiting))
    pareto_block(setting, rows)

    fact = factorial_block(setting, rows)
    return rows, results, fact


def factorial_block(setting, rows):
    """2 x 2 per design: encoder path x read-out path, per-image contrasts on Dice@4 (t = 0.50)."""
    print(f"\nFACTORIAL  {short(setting)}: encoder path (no/yes) x read-out path (no/yes), Dice@4 at t = 0.50")
    ready, fact = [], {}
    for design, none, enc, ro, both in cfg.FACTORIAL:
        names = (none, enc, ro, both)
        missing = [n for n in names if n not in rows]
        if missing:
            optional_only = all(is_optional(setting, n) for n in missing)
            fact[design] = dict(status='not run' if optional_only else 'waiting', missing=missing)
            print(f"  {design:5s}: {'not run' if optional_only else 'waiting for'} {', '.join(missing)}")
            continue
        why = comparable(rows, names)
        if why:
            fact[design] = dict(status='not comparable', why=why)
            print(f"  {design:5s}: NOT COMPARABLE: {why}")
            continue
        D = {k: rows[n]['dicek'] for k, n in zip(('none', 'enc', 'ro', 'both'), names)}
        eff = dict(enc_alone=paired(D['none'], D['enc']), enc_with_ro=paired(D['ro'], D['both']),
                   ro_alone=paired(D['none'], D['ro']), ro_with_enc=paired(D['enc'], D['both']))
        inter = paired_d((D['both'] - D['enc']) - (D['ro'] - D['none']))
        ready.append(design)
        fact[design] = dict(status='tested', eff=eff, inter=inter, names=names)
    adj = holm([fact[d]['inter']['p'] for d in ready]) if ready else []
    for design, p_adj in zip(ready, adj):
        f = fact[design]
        none, enc, ro, both = f['names']
        e, it = f['eff'], f['inter']
        if p_adj < cfg.P_LEVEL:
            v = 'complementary' if it['diff'] > 0 else 'overlapping'
        elif -cfg.MARGIN <= it['lo'] and it['hi'] <= cfg.MARGIN:
            v = 'additive'
        else:
            v = 'unclear'
        f['verdict'] = v
        print(f"  {design} design: neither {none} {rows[none]['dicek'].mean():.4f} | encoder only {enc} "
              f"{rows[enc]['dicek'].mean():.4f} | read-out only {ro} {rows[ro]['dicek'].mean():.4f} | "
              f"both {both} {rows[both]['dicek'].mean():.4f}")
        for key, label in (('enc_alone', 'encoder path, without read-out'), ('enc_with_ro', 'encoder path, with read-out'),
                           ('ro_alone', 'read-out path, without encoder'), ('ro_with_enc', 'read-out path, with encoder')):
            s = e[key]
            print(f"    {label:32s}: {s['diff']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]  improved {s['improved']:2d}/{s['n']}"
                  f"  p={s['p']:.1e}")
        print(f"    {'interaction':32s}: {it['diff']:+.4f} [{it['lo']:+.4f}, {it['hi']:+.4f}]  "
              f"p={it['p']:.1e} (Holm {p_adj:.1e})  -> {v.upper()}")
    return fact


def pareto_block(setting, rows):
    """Adapter size vs Dice@4: a row is on the front if no row with fewer adapter params scores higher."""
    if not rows:
        return
    order = sorted(rows.values(), key=lambda r: (r['adapter'], -r['dicek'].mean()))
    print(f"\nSIZE vs ACCURACY  {short(setting)}  (adapter params, Dice@4 at t = 0.50; * = on the front)")
    best = -1.0
    for r in order:
        d = float(r['dicek'].mean())
        front = d > best
        best = max(best, d)
        print(f"  {'*' if front else ' '} {r['row']:16s} {r['adapter'] / 1e6:6.2f}M  {d:.4f}")


# ----------------------------------------------------------------------------
# claims: each returns (status, text); status in holds / fails / unclear / waiting
# ----------------------------------------------------------------------------
def _v(results, a, b):
    return results.get((a, b), (None, None))[1]


def _ratio(rows, a, b):
    return rows[a]['adapter'] / max(rows[b]['adapter'], 1)


def claim_h1(rows, results, fact):
    vc, vb = _v(results, 'C_convlora', 'H1_lora1_hq'), _v(results, 'B1_lora1_ep100', 'H1_lora1_hq')
    if vc is None or vb is None:
        return 'waiting', "needs C, B1 and H1"
    if vb != 'beats':
        return 'fails', f"does not beat LoRA r=1 ({vb}) -> no claim for the RBA read-out"
    if vc in ('beats', 'matches'):
        return 'holds', f"{vc} Conv-LoRA with {_ratio(rows, 'C_convlora', 'H1_lora1_hq'):.0f}x fewer adapter params"
    return ('fails' if vc == 'worse' else 'unclear'), f"{vc} against Conv-LoRA"


def claim_whole(rows, results, fact):
    v = _v(results, 'C_convlora', 'H1_lora1_hq')
    if v is None:
        return 'waiting', "needs C and H1"
    if v in ('beats', 'matches'):
        return 'holds', f"RBA {v} Conv-LoRA with {_ratio(rows, 'C_convlora', 'H1_lora1_hq'):.0f}x fewer adapter params"
    return ('fails' if v == 'worse' else 'unclear'), f"RBA is {v} against Conv-LoRA"


def claim_rank1(rows, results, fact):
    v = _v(results, 'D_convlora_hq', 'H1_lora1_hq')
    if v is None:
        return 'waiting', "needs D and H1"
    if v in ('beats', 'matches'):
        return 'holds', f"LoRA r=1 {v} Conv-LoRA r=16 when both have the RBA read-out"
    return ('fails' if v == 'worse' else 'unclear'), f"LoRA r=1 is {v} against Conv-LoRA r=16 (both with HQ)"


def claim_h(rows, results, fact):
    vc, vb = _v(results, 'C_convlora', 'H_hq_only_ep100'), _v(results, 'B1_lora1_ep100', 'H_hq_only_ep100')
    if vc is None or vb is None:
        return 'waiting', "needs C, B1 and H"
    if vb != 'beats':
        return 'fails', f"does not beat LoRA r=1 ({vb})"
    if vc in ('beats', 'matches'):
        return 'holds', f"{vc} Conv-LoRA with {_ratio(rows, 'C_convlora', 'H_hq_only_ep100'):.0f}x fewer adapter params"
    return ('fails' if vc == 'worse' else 'unclear'), f"{vc} against Conv-LoRA"


def claim_second_readout(rows, results, fact):
    v = _v(results, 'B1_lora1_ep100', 'R_no_ridge')
    if v is None:
        return 'waiting', "needs B1 and R_no_ridge"
    if v == 'beats':
        return 'holds', ("RiDA's read-out (no ridge, about 20 K params) on LoRA r=1 beats LoRA r=1 alone: "
                         "the read-out effect does not depend on the RBA module")
    return ('fails' if v in ('worse', 'matches') else 'unclear'), f"{v} against LoRA r=1"


def claim_rida(rows, results, fact):
    v = _v(results, 'H1_lora1_hq', 'R_rida')
    if v is None:
        return 'waiting', "needs H1 and R_rida"
    if v in ('beats', 'matches'):
        return 'holds', f"RiDA {v} H1 in this setting, with {_ratio(rows, 'H1_lora1_hq', 'R_rida'):.1f}x fewer adapter params"
    if v == 'worse':
        return 'fails', "RiDA is worse than H1 in this setting"
    return 'unclear', "RiDA vs H1 is unclear (not significant, CI wider than +-0.003)"


def part_claim(a_row, part):
    def f(rows, results, fact):
        v = _v(results, a_row, 'R_rida')
        if v is None:
            return 'waiting', f"needs {a_row} and R_rida"
        word = {'beats': 'MATTERS', 'matches': 'adds nothing measurable', 'worse': 'HURTS'}.get(v, 'unclear')
        return ('holds' if v == 'beats' else ('unclear' if v == 'unclear' else 'fails')), f"{part}: {word}"
    return f


def claim_ridge_encoder(rows, results, fact):
    v = _v(results, 'B1_lora1_ep100', 'R_no_readout')
    if v is None:
        return 'waiting', "needs B1 and R_no_readout"
    word = {'beats': 'helps', 'matches': 'adds nothing measurable', 'worse': 'hurts'}.get(v, 'unclear')
    return ('holds' if v == 'beats' else ('unclear' if v == 'unclear' else 'fails')), \
        f"the ridge inside an encoder adapter alone (vs plain LoRA r=1): {word}"


def factorial_claim(design):
    def f(rows, results, fact):
        st = fact.get(design, {})
        if st.get('status') != 'tested':
            return 'waiting', st.get('why') or f"needs {', '.join(st.get('missing', []))}"
        v = st['verdict']
        it = st['inter']
        text = f"interaction {it['diff']:+.4f} [{it['lo']:+.4f}, {it['hi']:+.4f}] -> {v}"
        return ('holds' if v == 'complementary' else ('unclear' if v == 'unclear' else 'fails')), text
    return f


def claim_rba_best(rows, results, fact):
    vs = [('B1_lora1_ep100', 'LoRA r=1'), ('B_lora', 'LoRA r=16'), ('C_convlora', 'Conv-LoRA r=16'),
          ('AF_adaptformer', 'AdaptFormer')]
    got = {name: _v(results, a, 'H1_lora1_hq') for a, name in vs}
    missing = [n for n, v in got.items() if v is None]
    if missing:
        return 'waiting', "needs " + ", ".join(missing)
    text = ", ".join(f"{v} {n}" for n, v in got.items())
    if all(v == 'beats' for v in got.values()):
        return 'holds', f"RBA {text}"
    if any(v == 'worse' for v in got.values()):
        return 'fails', f"RBA: {text}"
    return 'unclear', f"RBA: {text}"


def claim_plug_in(rows, results, fact):
    pairs = [('B1_lora1_ep100', 'H1_lora1_hq', 'LoRA r=1'), ('C_convlora', 'D_convlora_hq', 'Conv-LoRA r=16'),
             ('B_lora', 'B_lora_hq', 'LoRA r=16')]
    got = [(n, _v(results, a, b)) for a, b, n in pairs if _v(results, a, b) is not None]
    if len(got) < 2:
        return 'waiting', "needs at least two encoder adapters with and without the read-out"
    text = "; ".join(f"on {n}: {v}" for n, v in got)
    if all(v == 'beats' for _, v in got):
        return 'holds', f"the read-out improves every encoder adapter tested ({len(got)}): {text}"
    return ('fails' if any(v in ('worse', 'matches') for _, v in got) else 'unclear'), text


CLAIMS = [   # (claim, test, rows it needs)
    ('RBA beats every encoder adapter tested',   claim_rba_best,
     ('B1_lora1_ep100', 'B_lora', 'C_convlora', 'AF_adaptformer', 'H1_lora1_hq')),
    ('RBA read-out is a plug-in',               claim_plug_in,
     ('B1_lora1_ep100', 'H1_lora1_hq', 'C_convlora', 'D_convlora_hq')),
    ('RBA vs Conv-LoRA, whole adapter',         claim_whole, ('C_convlora', 'H1_lora1_hq')),
    ('RBA (H1) vs Conv-LoRA',                   claim_h1,    ('C_convlora', 'B1_lora1_ep100', 'H1_lora1_hq')),
    ('rank 1 is enough in RBA',                 claim_rank1, ('D_convlora_hq', 'H1_lora1_hq')),
    ('RBA read-out branch alone vs Conv-LoRA',  claim_h,     ('C_convlora', 'B1_lora1_ep100', 'H_hq_only_ep100')),
    ('read-out effect, second design',            claim_second_readout, ('B1_lora1_ep100', 'R_no_ridge')),
    ('RiDA as the method (vs H1)',                claim_rida,  ('H1_lora1_hq', 'R_rida')),
    ('RiDA: ridge operator',  part_claim('R_no_ridge', 'the ridge operator'),  ('R_no_ridge', 'R_rida')),
    ('RiDA: read-out path',   part_claim('R_no_readout', 'the read-out path'), ('R_no_readout', 'R_rida')),
    ('RiDA: encoder path',    part_claim('R_no_encoder', 'the encoder path'),  ('R_no_encoder', 'R_rida')),
    ('ridge in the encoder path alone',           claim_ridge_encoder, ('B1_lora1_ep100', 'R_no_readout')),
    ('paths complementary, RBA design',           factorial_claim('RBA'),
     ('A_decoder_only', 'B1_lora1_ep100', 'H_hq_only_ep100', 'H1_lora1_hq')),
    ('paths complementary, RiDA design',          factorial_claim('RiDA'),
     ('A_decoder_only', 'R_no_readout', 'R_no_encoder', 'R_rida')),
]


def main():
    per_task = {}
    for setting in cfg.SETTINGS:
        rows, results, fact = task_table(setting)
        print("\nCLAIMS")
        status = {}
        for name, fn, needs in CLAIMS:
            st, text = fn(rows, results, fact)
            missing = [r for r in needs if r not in rows]
            if st == 'waiting' and missing and all(is_optional(setting, r) for r in missing):
                st, text = 'not run', f"not planned in {short(setting)} ({', '.join(missing)})"
            elif st == 'waiting' and not missing:
                text = "the rows exist but are not comparable (different epochs or test images, see above)"
            status[name] = st
            print(f"  {name:40s}: [{st:7s}] {text}")
        per_task[setting] = (rows, results, status)

    print(f"\n{'=' * 100}\nREPLICATION  (a claim is written only if it holds in every setting it was tested in)\n{'=' * 100}")
    print(f"  {'':40s}  " + "  ".join(f"{short(st):16s}" for st in cfg.SETTINGS))
    for name, _, _ in CLAIMS:
        sts = {t: per_task[t][2][name] for t in cfg.SETTINGS}
        ran = {t: s for t, s in sts.items() if s != 'not run'}
        skipped = [t for t, s in sts.items() if s == 'not run']
        if any(s == 'waiting' for s in ran.values()):
            overall = "waiting"
        elif not ran:
            overall = "not run"
        elif all(s == 'holds' for s in ran.values()):
            overall = ("HOLDS IN EVERY SETTING -> can be written" if not skipped else
                       ("holds in the one setting tested -> state it for that setting" if len(ran) == 1 else
                        f"holds in all {len(ran)} settings tested -> state it for those"))
        elif any(s == 'holds' for s in ran.values()):
            overall = "setting-dependent -> state it per setting, not as a general claim"
        elif any(s == 'unclear' for s in ran.values()):
            overall = "unclear -> more seeds, or no claim"
        else:
            overall = "does not hold"
        per = "  ".join(f"{s:16s}" for s in sts.values())
        print(f"  {name:40s}  {per}  => {overall}")
    return per_task


if __name__ == "__main__":
    main()
