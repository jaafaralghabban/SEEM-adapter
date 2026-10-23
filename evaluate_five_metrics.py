# ============================================================================
# evaluate_five_metrics.py -- Dice, IoU, ACC, SE and SP of saved checkpoints (no training)
#   Re-tests each trained row on its test images after four clicks at threshold 0.5, scored on the
#   native label grid, and averages the five metrics over the images (the format of the OCTA-500
#   paper, Tables S4 and S6-S8; Supplementary Tables S1-S5 of our article).
#
#   python evaluate_five_metrics.py                                   # every row and setting below
#   python evaluate_five_metrics.py --rows H1_lora1_hq --settings GT_Vein:OCTA_3mm
#   Output: checkpoints/five_metrics.csv (rows already in it are skipped).
# ============================================================================
import os
import csv
import gc
import argparse

import numpy as np

import train_rba as T


class Config:
    ROWS = ('C_convlora', 'H1_lora1_hq')
    SETTINGS = (
        ('GT_LargeVessel', 'OCTA_3mm'), ('GT_Artery', 'OCTA_3mm'), ('GT_Vein', 'OCTA_3mm'), ('GT_FAZ', 'OCTA_3mm'),
        ('GT_LargeVessel', 'OCTA_6mm'), ('GT_Artery', 'OCTA_6mm'), ('GT_Vein', 'OCTA_6mm'), ('GT_FAZ', 'OCTA_6mm'),
        ('GT_RoseVessel', 'ROSE_O'), ('GT_RoseFAZ', 'ROSE_O'),
    )
    THRESHOLD = 0.5
    OUT_CSV = os.path.join('checkpoints', 'five_metrics.csv')


FIELDS = ('task', 'subset', 'row', 'n', 'dice', 'iou', 'acc', 'se', 'sp')


def five_metrics(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    tp = float((pred & gt).sum()); tn = float((~pred & ~gt).sum())
    fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
    s = 1e-6                                                    # as calc_metrics in train_rba.py
    return dict(dice=2 * tp / (pred.sum() + gt.sum() + s), iou=tp / (tp + fp + fn + s),
                acc=(tp + tn) / (tp + tn + fp + fn), se=tp / (tp + fn + s), sp=tn / (tn + fp + s))


def run_one(row, task, subset, thr):
    T.configure(row=row, task=task, subset=subset)
    got = T._load_row_for_check(row)                           # None if there is no matching checkpoint
    if got is None:
        return None
    model, hq, cad = got
    _, _, test_ids = T.make_splits()
    dl = T.DataLoader(T.OCTADataset(test_ids), batch_size=1, shuffle=False, num_workers=T.cfg.NUM_WORKERS)
    out = T.evaluate(model, hq, cad, dl, (thr,), T.cfg.K_ROUNDS, 'giir', k_report=T.cfg.K_ROUNDS,
                     return_preds=True, desc=f"five metrics {row} {task} {subset}")
    o = out[thr]
    per = [five_metrics(p, g) for p, g in zip(o['preds'], o['gts'])]
    del model, hq, cad, out
    T._release()
    gc.collect()
    m = {k: float(np.mean([x[k] for x in per])) for k in ('dice', 'iou', 'acc', 'se', 'sp')}
    if abs(m['dice'] - float(o['dice'])) > 1e-6:
        print(f"  WARNING: Dice {m['dice']:.4f} differs from the trainer's {float(o['dice']):.4f}")
    return dict(task=task, subset=subset, row=row, n=len(per), **m)


def main():
    c = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', nargs='*', default=list(c.ROWS))
    ap.add_argument('--settings', nargs='*', help="TASK:SUBSET, e.g. GT_Vein:OCTA_3mm")
    a = ap.parse_args()
    settings = [tuple(s.split(':')) for s in a.settings] if a.settings else list(c.SETTINGS)
    done = {}
    if os.path.exists(c.OUT_CSV):
        with open(c.OUT_CSV, newline='', encoding='utf-8') as f:
            done = {(r['task'], r['subset'], r['row']): r for r in csv.DictReader(f)}
    for task, subset in settings:
        for row in a.rows:
            if (task, subset, row) in done:
                continue
            r = run_one(row, task, subset, c.THRESHOLD)
            if r is None:
                continue
            done[(task, subset, row)] = r
            os.makedirs(os.path.dirname(c.OUT_CSV), exist_ok=True)
            with open(c.OUT_CSV, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                for k in sorted(done):
                    w.writerow({x: done[k][x] for x in FIELDS})
    print(f"\n{'task':15s} {'subset':9s} {'row':16s} {'n':>3s} " + " ".join(f"{m:>7s}" for m in ('Dice', 'IoU', 'ACC', 'SE', 'SP')))
    for (task, subset, row), r in sorted(done.items()):
        print(f"{task:15s} {subset:9s} {row:16s} {int(r['n']):3d} "
              + " ".join(f"{100 * float(r[m]):7.2f}" for m in ('dice', 'iou', 'acc', 'se', 'sp')))
    print(f"\n-> {os.path.abspath(c.OUT_CSV)}")


if __name__ == "__main__":
    main()
