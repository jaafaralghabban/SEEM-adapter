# ============================================================================
# prepare_data.py -- put OCTA-500 (3 mm, 6 mm) and ROSE-O in the layout train_rba.py reads (no training)
#   The trainer reads every dataset the same way: three projection folders that share one file name per
#   image, one label folder per target, and a split file listing the image names:
#     <DATA_DIR>/<SUBSET>/OCTA(FULL) | OCTA(ILM_OPL) | OCTA(OPL_BM)/<id>.<ext>
#     <DATA_DIR>/Label/Label/<TASK>/<id>.<ext>
#     checkpoints/<SPLIT_FILE>   = {'train': [...], 'val': [...], 'test': [...]}
#   OCTA-500 3 mm : official split 10301-10440 / 10441-10450 / 10451-10500 (140 / 10 / 50)
#   OCTA-500 6 mm : official split 10001-10240 / 10241-10250 / 10251-10300 (240 / 10 / 50)
#   ROSE-O        : SVC, DVC, IVC -> the three input channels; 30 training subjects -> 25 train + 5 validation
#                   (seed 42), 9 test subjects; vessel and FAZ targets.
#   The datasets are not redistributed here: get OCTA-500 from IEEE DataPort and ROSE-O from its authors,
#   then set the folders in Config. Original folders are only read; files are copied, never moved.
#   python prepare_data.py          -> prints READY or what is missing for each dataset
# ============================================================================
import os
import random
import shutil

import numpy as np
import cv2
import torch


class Config:
    CHECKPOINT_DIR = 'checkpoints'
    SEED = 42
    # ---- OCTA-500 ----------------------------------------------------------------
    # OCTA-500 as downloaded: octa/OCTA_3mm/OCTA(FULL|ILM_OPL|OPL_BM)/<id>.bmp and octa/Label/Label/<TASK>/<id>.bmp
    OCTA_DIR = 'octa'                                         # <-- the folder that holds OCTA_3mm and Label
    THREE_SUBSET = 'OCTA_3mm'
    THREE_TASKS = ('GT_LargeVessel', 'GT_Artery', 'GT_Vein', 'GT_FAZ', 'GT_Capillary')
    THREE_SPLIT_FILE = 'data_split_paper.pth'
    THREE_MM_SPLIT_FILE = THREE_SPLIT_FILE                    # checked: no 3 mm image may enter the 6 mm split
    THREE_TRAIN = (10301, 10440)
    THREE_VAL = (10441, 10450)
    THREE_TEST = (10451, 10500)
    SIX_SUBSET = 'OCTA_6mm'
    SIX_SOURCE_CANDIDATES = (os.path.join('octa', 'OCTA_6mm'), os.path.join('octa', 'OCTA_6M'),
                             os.path.join('octa', 'Projection Maps'),
                             os.path.join('octa', 'OCTA_6mm', 'Projection Maps'),
                             os.path.join('octa', 'OCTA_6M', 'Projection Maps'),
                             'OCTA_6mm', 'OCTA_6M')            # <-- the folder that holds the 6 mm OCTA(FULL)
    SIX_TASK = 'GT_LargeVessel'
    SIX_TASKS = ('GT_Vein', 'GT_Artery', 'GT_FAZ')
    SIX_SPLIT_FILE = 'data_split_paper_6mm.pth'
    SIX_TRAIN = (10001, 10240)
    SIX_VAL = (10241, 10250)
    SIX_TEST = (10251, 10300)
    # ---- ROSE-O -------------------------------------------------------------------
    # ROSE-O as released: ROSE-O/{train,test}/img/{SVC,DVC,IVC} and ROSE-O/{train,test}/gt/{vessel,FAZ}
    ROSE_ROOT_CANDIDATES = ('ROSE-O', os.path.join('data', 'ROSE-O'), os.path.join('datasets', 'ROSE-O'),
                            os.path.join('..', 'ROSE-O'))    # <-- add the full path of your ROSE-O folder
    ROSE_CHANNELS = ('SVC', 'DVC', 'IVC')                     # -> OCTA(FULL), OCTA(ILM_OPL), OCTA(OPL_BM)
    ROSE_GT_SUBDIR = 'vessel'
    ROSE_FAZ_SUBDIR = 'FAZ'
    ROSE_N_VAL = 5
    ROSE_DATA_DIR = 'octa_rose'
    ROSE_SUBSET = 'ROSE_O'
    ROSE_TASK = 'GT_RoseVessel'
    ROSE_FAZ_TASK = 'GT_RoseFAZ'
    ROSE_SPLIT_FILE = 'data_split_rose_vessel.pth'
    ROSE_FAZ_SPLIT_FILE = 'data_split_rose_faz.pth'
    ROSE_VESSEL_SPLIT_FILE = 'data_split_rose_vessel.pth'


PROJ = ('OCTA(FULL)', 'OCTA(ILM_OPL)', 'OCTA(OPL_BM)')
IMG_EXTS = ('.tif', '.tiff', '.png', '.bmp', '.jpg')


def _find(base, stem, exts=IMG_EXTS):
    for e in exts:
        p = os.path.join(base, stem + e)
        if os.path.exists(p):
            return p
    return None


def _read_grey(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f"cannot read {p}")
    return im


def _load_split(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


# ----------------------------------------------------------------------------
# ROSE-O FAZ
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# OCTA-500 3 mm
# ----------------------------------------------------------------------------
def prepare_three(cfg):
    print("=" * 72 + "\n OCTA-500 3 mm\n" + "=" * 72)
    ids = list(range(cfg.THREE_TRAIN[0], cfg.THREE_TEST[1] + 1))
    dirs = [os.path.join(cfg.OCTA_DIR, cfg.THREE_SUBSET, p) for p in PROJ]
    if not all(os.path.isdir(d) for d in dirs):
        print("  NOT FOUND: " + ", ".join(os.path.abspath(d) for d in dirs if not os.path.isdir(d))
              + "\n  Set Config.OCTA_DIR to the folder that holds OCTA_3mm and Label.")
        return False
    first = _find(dirs[0], str(ids[0]))
    if first is None:
        print(f"  no image {ids[0]}.* in {os.path.abspath(dirs[0])}")
        return False
    ext = os.path.splitext(first)[1]
    have_img = [i for i in ids if all(os.path.exists(os.path.join(d, f"{i}{ext}")) for d in dirs)]
    print(f"  {len(have_img)}/{len(ids)} images with all three projections ({ext})")
    ok_any = False
    for t in cfg.THREE_TASKS:
        gt_dir = os.path.join(cfg.OCTA_DIR, 'Label', 'Label', t)
        n = sum(os.path.exists(os.path.join(gt_dir, f"{i}{ext}")) for i in have_img)
        print(f"  {t:15s}: {n}/{len(have_img)} labels" + ("" if n == len(have_img) else f"  (in {os.path.abspath(gt_dir)})"))
        ok_any |= n == len(have_img)
    rng = lambda a, b: [f"{i}{ext}" for i in range(a, b + 1) if i in set(have_img)]
    split = {'train': rng(*cfg.THREE_TRAIN), 'val': rng(*cfg.THREE_VAL), 'test': rng(*cfg.THREE_TEST)}
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    torch.save(split, os.path.join(cfg.CHECKPOINT_DIR, cfg.THREE_SPLIT_FILE))
    print(f"  split -> {os.path.join(cfg.CHECKPOINT_DIR, cfg.THREE_SPLIT_FILE)}: train {len(split['train'])} | "
          f"val {len(split['val'])} | test {len(split['test'])}  (paper: 140 / 10 / 50)")
    print("  OCTA-500 3 mm READY" if ok_any else "  OCTA-500 3 mm: labels missing")
    return ok_any


# ----------------------------------------------------------------------------
# OCTA-500 6 mm
# ----------------------------------------------------------------------------
def prepare_six(cfg):
    print("=" * 72 + "\n OCTA-500 6 mm\n" + "=" * 72)
    ids = list(range(cfg.SIX_TRAIN[0], cfg.SIX_TEST[1] + 1))
    target = [os.path.join(cfg.OCTA_DIR, cfg.SIX_SUBSET, p) for p in PROJ]
    first = str(ids[0])
    ext = None
    if all(os.path.isdir(d) for d in target) and _find(target[0], first):
        ext = os.path.splitext(_find(target[0], first))[1]
        print(f"  projections already in {os.path.abspath(os.path.join(cfg.OCTA_DIR, cfg.SIX_SUBSET))}")
    else:
        src = next((c for c in cfg.SIX_SOURCE_CANDIDATES
                    if os.path.isdir(os.path.join(c, PROJ[0])) and _find(os.path.join(c, PROJ[0]), first)), None)
        if src is None:
            print("  NOT FOUND. Looked for a folder containing OCTA(FULL)/10001.* under:\n    "
                  + "\n    ".join(os.path.abspath(c) for c in cfg.SIX_SOURCE_CANDIDATES)
                  + "\n  Add the folder that holds the 6 mm OCTA(FULL), OCTA(ILM_OPL) and OCTA(OPL_BM) to "
                    "Config.SIX_SOURCE_CANDIDATES and run again.")
            return False
        ext = os.path.splitext(_find(os.path.join(src, PROJ[0]), first))[1]
        print(f"  source: {os.path.abspath(src)} -> copying the 6 mm projections into "
              f"{os.path.abspath(os.path.join(cfg.OCTA_DIR, cfg.SIX_SUBSET))}")
        n_copied, missing = 0, []
        for p, d in zip(PROJ, target):
            os.makedirs(d, exist_ok=True)
            for i in ids:
                s = os.path.join(src, p, f"{i}{ext}")
                if os.path.exists(s):
                    if not os.path.exists(os.path.join(d, f"{i}{ext}")):
                        shutil.copy2(s, d)
                        n_copied += 1
                else:
                    missing.append(f"{p}/{i}{ext}")
        print(f"  copied {n_copied} files" + (f" | MISSING {len(missing)}: {missing[:5]}" if missing else ""))
    gt_dir = os.path.join(cfg.OCTA_DIR, 'Label', 'Label', cfg.SIX_TASK)
    have_img = [i for i in ids if all(os.path.exists(os.path.join(d, f"{i}{ext}")) for d in target)]
    have_gt = [i for i in have_img if os.path.exists(os.path.join(gt_dir, f"{i}{ext}"))]
    print(f"  {len(have_img)}/{len(ids)} images with all three projections | "
          f"{len(have_gt)}/{len(ids)} with a label in {os.path.abspath(gt_dir)}")
    if len(have_gt) < len(ids):
        lack = sorted(set(ids) - set(have_gt))
        print(f"  MISSING image or label for {len(lack)} IDs, e.g. {lack[:6]}. The 6 mm labels of OCTA-500 "
              f"go in {gt_dir} with the same file names as the images.")
        if not have_gt:
            return False
    img = _read_grey(os.path.join(target[0], f"{have_gt[0]}{ext}"))
    gt = _read_grey(os.path.join(gt_dir, f"{have_gt[0]}{ext}"))
    print(f"  image {img.shape} | label {gt.shape}" + ("" if img.shape == gt.shape else "  <-- SIZES DIFFER"))
    if img.shape != gt.shape:
        print("  STOP: the trainer needs image and label on the same grid.")
        return False
    rng = lambda a, b: [f"{i}{ext}" for i in range(a, b + 1) if i in set(have_gt)]
    split = {'train': rng(*cfg.SIX_TRAIN), 'val': rng(*cfg.SIX_VAL), 'test': rng(*cfg.SIX_TEST)}
    p3 = os.path.join(cfg.CHECKPOINT_DIR, cfg.THREE_MM_SPLIT_FILE)
    if os.path.exists(p3):
        try:
            s3 = torch.load(p3, map_location='cpu', weights_only=False)
        except TypeError:
            s3 = torch.load(p3, map_location='cpu')
        both = set(sum((list(v) for v in s3.values()), [])) & set(sum(split.values(), []))
        print("  no overlap with the 3 mm split" if not both else f"  WARNING: {len(both)} IDs also in the 3 mm split")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    torch.save(split, os.path.join(cfg.CHECKPOINT_DIR, cfg.SIX_SPLIT_FILE))
    print(f"  split -> {os.path.join(cfg.CHECKPOINT_DIR, cfg.SIX_SPLIT_FILE)}: "
          f"train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}  (paper: 240 / 10 / 50)")
    print(f"  trainer settings: DATA_DIR='{cfg.OCTA_DIR}', SUBSET='{cfg.SIX_SUBSET}', TASK='{cfg.SIX_TASK}', "
          f"SPLIT_FILE='{cfg.SIX_SPLIT_FILE}'  (set by train_rba.py --subset OCTA_6mm)")
    print("  OCTA-500 6 mm READY")
    return True


def check_six(cfg):
    print("=" * 72 + "\n OCTA-500 6 mm: veins, arteries, FAZ\n" + "=" * 72)
    ps = os.path.join(cfg.CHECKPOINT_DIR, cfg.SIX_SPLIT_FILE)
    if not os.path.exists(ps):
        print(f"  NOT FOUND: {ps}. prepare_six writes the 6 mm split first; see the messages above.")
        return {t: False for t in cfg.SIX_TASKS}
    split = _load_split(ps)
    ids = list(split['train']) + list(split['val']) + list(split['test'])
    img_dir = os.path.join(cfg.OCTA_DIR, cfg.SIX_SUBSET, PROJ[0])
    shape = _read_grey(os.path.join(img_dir, ids[0])).shape
    print(f"  6 mm split: train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])} "
          f"| images {shape}")
    ok = {}
    for t in cfg.SIX_TASKS:
        d = os.path.join(cfg.OCTA_DIR, 'Label', 'Label', t)
        if not os.path.isdir(d):
            print(f"  {t:10s}: NOT FOUND {os.path.abspath(d)}")
            ok[t] = False
            continue
        present = [i for i in ids if os.path.exists(os.path.join(d, i))]
        missing = sorted(set(ids) - set(present))
        gshape = _read_grey(os.path.join(d, present[0])).shape if present else None
        good = not missing and gshape == shape
        frac = np.mean([(_read_grey(os.path.join(d, i)) > 0).mean() for i in ids[:20]]) if good else 0.0
        print(f"  {t:10s}: {len(ids) - len(missing)}/{len(ids)} labels"
              + (f" | MISSING e.g. {missing[:5]}" if missing else "")
              + (f" | label {gshape}" if gshape else "")
              + (f" | foreground {100 * frac:.1f}% | READY" if good else " | NOT READY"))
        ok[t] = good
    return ok


# ----------------------------------------------------------------------------
# ROSE-O
# ----------------------------------------------------------------------------
def prepare_rose(cfg):
    print("=" * 72 + "\n ROSE-O\n" + "=" * 72)
    root = next((r for r in cfg.ROSE_ROOT_CANDIDATES
                 if os.path.isdir(os.path.join(r, 'train', 'img', cfg.ROSE_CHANNELS[0]))), None)
    if root is None:
        print("  NOT FOUND. Looked for <root>/train/img/SVC under:\n    " + "\n    ".join(
            os.path.abspath(r) for r in cfg.ROSE_ROOT_CANDIDATES)
              + "\n  Add the full path of your ROSE-O folder to Config.ROSE_ROOT_CANDIDATES and run again.")
        return False
    print(f"  source: {os.path.abspath(root)}")
    out_img = [os.path.join(cfg.ROSE_DATA_DIR, cfg.ROSE_SUBSET, p) for p in PROJ]
    out_gt = os.path.join(cfg.ROSE_DATA_DIR, 'Label', 'Label', cfg.ROSE_TASK)
    for d in out_img + [out_gt]:
        os.makedirs(d, exist_ok=True)
    stems, sizes, resized, fractions = {}, set(), 0, []
    for split in ('train', 'test'):
        d0 = os.path.join(root, split, 'img', cfg.ROSE_CHANNELS[0])
        all_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(d0) if f.lower().endswith(IMG_EXTS))
        ok, missing = [], []
        for s in all_stems:
            chans = [_find(os.path.join(root, split, 'img', c), s) for c in cfg.ROSE_CHANNELS]
            gt_p = _find(os.path.join(root, split, 'gt', cfg.ROSE_GT_SUBDIR), s, ('.png', '.tif', '.tiff', '.bmp'))
            if None in chans or gt_p is None:
                missing.append(s)
                continue
            projs = [_read_grey(p) for p in chans]
            h0, w0 = projs[0].shape
            projs = [p if p.shape == (h0, w0) else cv2.resize(p, (w0, h0), interpolation=cv2.INTER_LINEAR)
                     for p in projs]                       # as your earlier ROSE loader
            gt = (_read_grey(gt_p) > 0).astype(np.uint8)
            if gt.shape != (h0, w0):                       # keep the label grid; bring the image to it
                gh, gw = gt.shape
                projs = [cv2.resize(p, (gw, gh), interpolation=cv2.INTER_AREA if gh < h0 else cv2.INTER_LINEAR)
                         for p in projs]
                resized += 1
            sizes.add(gt.shape)
            fractions.append(float(gt.mean()))
            name = f"{split}_{s}.png"
            for d, p in zip(out_img, projs):
                cv2.imwrite(os.path.join(d, name), p)
            cv2.imwrite(os.path.join(out_gt, name), gt * 255)
            ok.append(s)
        stems[split] = ok
        print(f"  {split:5s}: {len(ok)} subjects with all of SVC, DVC, IVC and the vessel label"
              + (f" | MISSING something for {len(missing)}: {missing[:6]}" if missing else ""))
    if not stems['train'] or not stems['test']:
        print("  STOP: no complete training or test subjects.")
        return False
    random.seed(cfg.SEED)                                  # the same 25 / 5 split as your earlier ROSE runs
    shuf = stems['train'][:]
    random.shuffle(shuf)
    val, train = shuf[:cfg.ROSE_N_VAL], shuf[cfg.ROSE_N_VAL:]
    split = {'train': [f"train_{s}.png" for s in train], 'val': [f"train_{s}.png" for s in val],
             'test': [f"test_{s}.png" for s in stems['test']]}
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    torch.save(split, os.path.join(cfg.CHECKPOINT_DIR, cfg.ROSE_SPLIT_FILE))
    print(f"  split -> {os.path.join(cfg.CHECKPOINT_DIR, cfg.ROSE_SPLIT_FILE)}: "
          f"train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")
    print(f"  label grid {sorted(sizes)} | vessel pixels {100 * np.mean(fractions):.1f}% of the image"
          + (f" | {resized} images resized to their label grid" if resized else ""))
    print(f"  trainer settings: DATA_DIR='{cfg.ROSE_DATA_DIR}', SUBSET='{cfg.ROSE_SUBSET}', "
          f"TASK='{cfg.ROSE_TASK}', SPLIT_FILE='{cfg.ROSE_SPLIT_FILE}'  (set by train_rba.py --subset ROSE_O)")
    print("  ROSE-O READY")
    return True


def prepare_rose_faz(cfg):
    print("=" * 72 + "\n ROSE-O FAZ\n" + "=" * 72)
    root = next((r for r in cfg.ROSE_ROOT_CANDIDATES
                 if os.path.isdir(os.path.join(r, 'train', 'img', cfg.ROSE_CHANNELS[0]))), None)
    if root is None:
        print("  NOT FOUND. Looked for <root>/train/img/SVC under:\n    " + "\n    ".join(
            os.path.abspath(r) for r in cfg.ROSE_ROOT_CANDIDATES)
              + "\n  Add the full path of your ROSE-O folder to Config.ROSE_ROOT_CANDIDATES and run again.")
        return False
    sub = cfg.ROSE_FAZ_SUBDIR
    if not os.path.isdir(os.path.join(root, 'train', 'gt', sub)):
        alt = [d for d in os.listdir(os.path.join(root, 'train', 'gt')) if 'faz' in d.lower()]
        if not alt:
            print(f"  NO FAZ LABELS: {os.path.abspath(os.path.join(root, 'train', 'gt'))} has no FAZ folder.")
            return False
        sub = alt[0]
    print(f"  source: {os.path.abspath(root)} | FAZ labels in gt/{sub}")
    out_img = [os.path.join(cfg.ROSE_DATA_DIR, cfg.ROSE_SUBSET, p) for p in PROJ]
    out_gt = os.path.join(cfg.ROSE_DATA_DIR, 'Label', 'Label', cfg.ROSE_FAZ_TASK)
    for d in out_img + [out_gt]:
        os.makedirs(d, exist_ok=True)
    complete, no_faz, sizes, area = {}, [], set(), []
    for split in ('train', 'test'):
        d0 = os.path.join(root, split, 'img', cfg.ROSE_CHANNELS[0])
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(d0) if f.lower().endswith(IMG_EXTS))
        complete[split] = []
        for s in stems:
            chans = [_find(os.path.join(root, split, 'img', c), s) for c in cfg.ROSE_CHANNELS]
            if None in chans:
                continue
            complete[split].append(s)                      # the stem list the vessel split was drawn from
            faz_p = _find(os.path.join(root, split, 'gt', sub), s, ('.png', '.tif', '.tiff', '.bmp'))
            if faz_p is None:
                no_faz.append(f"{split}/{s}")
                continue
            name = f"{split}_{s}.png"
            gt = (_read_grey(faz_p) > 0).astype(np.uint8)
            if not all(os.path.exists(os.path.join(d, name)) for d in out_img):
                projs = [_read_grey(p) for p in chans]
                h0, w0 = projs[0].shape
                projs = [p if p.shape == (h0, w0) else cv2.resize(p, (w0, h0), interpolation=cv2.INTER_LINEAR)
                         for p in projs]
                if gt.shape != (h0, w0):
                    gh, gw = gt.shape
                    projs = [cv2.resize(p, (gw, gh), interpolation=cv2.INTER_AREA if gh < h0 else cv2.INTER_LINEAR)
                             for p in projs]
                for d, p in zip(out_img, projs):
                    cv2.imwrite(os.path.join(d, name), p)
            img_shape = _read_grey(os.path.join(out_img[0], name)).shape
            if gt.shape != img_shape:
                print(f"  STOP: {name}: FAZ label {gt.shape} vs image {img_shape}")
                return False
            cv2.imwrite(os.path.join(out_gt, name), gt * 255)
            sizes.add(gt.shape)
            area.append(float(gt.mean()))
    random.seed(cfg.SEED)                                  # exactly the draw of the vessel split
    shuf = complete['train'][:]
    random.shuffle(shuf)
    val, train = shuf[:cfg.ROSE_N_VAL], shuf[cfg.ROSE_N_VAL:]
    has = lambda sp, s: f"{sp}/{s}" not in no_faz
    split = {'train': [f"train_{s}.png" for s in train if has('train', s)],
             'val': [f"train_{s}.png" for s in val if has('train', s)],
             'test': [f"test_{s}.png" for s in complete['test'] if has('test', s)]}
    if no_faz:
        print(f"  {len(no_faz)} subjects have no FAZ label and are left out: {no_faz}")
    pv = os.path.join(cfg.CHECKPOINT_DIR, cfg.ROSE_VESSEL_SPLIT_FILE)
    if os.path.exists(pv):
        sv = _load_split(pv)
        same = all(list(sv[k]) == split[k] for k in ('train', 'val', 'test'))
        print("  same subjects as your ROSE-O vessel runs" if same else
              "  NOTE: differs from the vessel split only by the subjects without a FAZ label")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    torch.save(split, os.path.join(cfg.CHECKPOINT_DIR, cfg.ROSE_FAZ_SPLIT_FILE))
    print(f"  split -> {os.path.join(cfg.CHECKPOINT_DIR, cfg.ROSE_FAZ_SPLIT_FILE)}: "
          f"train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")
    print(f"  label grid {sorted(sizes)} | FAZ covers {100 * np.mean(area):.1f}% of the image on average")
    print("  ROSE-O FAZ READY")
    return True


# ----------------------------------------------------------------------------
# OCTA-500 6 mm: vein, artery, FAZ labels
# ----------------------------------------------------------------------------
def main():
    cfg = Config()
    steps = (('OCTA-500 3 mm', prepare_three), ('OCTA-500 6 mm', prepare_six), ('6 mm targets', check_six),
             ('ROSE-O vessels', prepare_rose), ('ROSE-O FAZ', prepare_rose_faz))
    ok = {}
    for name, fn in steps:
        try:
            r = fn(cfg)
            ok[name] = all(r.values()) if isinstance(r, dict) else bool(r)
        except Exception as e:
            print(f"  {name}: stopped by an error -- {type(e).__name__}: {e}")
            ok[name] = False
    print("\n" + "=" * 72)
    for name, r in ok.items():
        print(f" {name:15s}: " + ("READY" if r else "not ready (see above)"))
    print("=" * 72)


if __name__ == "__main__":
    main()
