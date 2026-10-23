# ============================================================================
# setup_seem.py -- get SEEM v1.0, add our two patched files, download SEEM's Focal-L weights (run once)
#   1. clones https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once (branch v1.0) into ./SEEM
#   2. copies seem_patches/seem_v0.py and seem_patches/attention_data_struct_seemv0.py into it
#      (the originals are kept as *.orig), and checks that the patches are the right files
#   3. downloads seem_focall_v0.pt (about 1.3 GB) into ./SEEM
#   python setup_seem.py              # then: python train_rba.py --help
#   If you already have a working SEEM folder, point to it instead:  set SEEM_DIR=C:\path\to\seem
# ============================================================================
import os
import sys
import shutil
import subprocess
import urllib.request


class Config:
    SEEM_DIR = os.environ.get('SEEM_DIR', 'SEEM')
    SEEM_REPO = 'https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git'
    SEEM_BRANCH = 'v1.0'
    WEIGHTS_URL = 'https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt'
    WEIGHTS = 'seem_focall_v0.pt'
    CONFIG_FILE = os.path.join('configs', 'seem', 'focall_unicl_lang_demo.yaml')
    PATCHES = {   # file in seem_patches/  ->  place inside the SEEM folder, text that must be in it
        'seem_v0.py': (os.path.join('modeling', 'interface', 'seem_v0.py'), ('spatial_pos_prompt', 'content_gate')),
        'attention_data_struct_seemv0.py': (os.path.join('modeling', 'interface', 'prototype',
                                                         'attention_data_struct_seemv0.py'), ()),
    }
    REQUIRED = ('seem_v0.py',)     # the learned positional click prompt lives here; the checkpoints need it
    MIN_WEIGHT_BYTES = 100 * 1024 * 1024


cfg = Config()
HERE = os.path.dirname(os.path.abspath(__file__))


def clone():
    if os.path.isdir(os.path.join(cfg.SEEM_DIR, 'modeling')):
        print(f"[1/3] SEEM already in {os.path.abspath(cfg.SEEM_DIR)}")
        return True
    print(f"[1/3] cloning SEEM {cfg.SEEM_BRANCH} into {os.path.abspath(cfg.SEEM_DIR)}")
    rc = subprocess.call(['git', 'clone', '--depth', '1', '--branch', cfg.SEEM_BRANCH, cfg.SEEM_REPO, cfg.SEEM_DIR])
    if rc != 0:
        print("      git clone failed: install git (https://git-scm.com) or download the v1.0 branch as a ZIP "
              f"from {cfg.SEEM_REPO[:-4]}/tree/{cfg.SEEM_BRANCH} and unzip it to {cfg.SEEM_DIR}")
        return False
    return True


def patch():
    print("[2/3] applying the patched SEEM files from seem_patches/")
    ok = True
    for name, (rel, markers) in cfg.PATCHES.items():
        src = os.path.join(HERE, 'seem_patches', name)
        dst = os.path.join(cfg.SEEM_DIR, rel)
        if not os.path.exists(src):
            need = name in cfg.REQUIRED
            print(f"      {'MISSING' if need else 'not given (SEEM original kept)'}: seem_patches/{name}"
                  + (f" -- copy it from the SEEM folder the paper was run in (<that folder>/{rel.replace(os.sep, '/')});"
                     " see seem_patches/README.md" if need else ""))
            ok = ok and not need
            continue
        text = open(src, encoding='utf-8', errors='ignore').read()
        lacking = [m for m in markers if m not in text]
        if lacking:
            print(f"      seem_patches/{name} does not contain {lacking}: this is not the patched file")
            ok = False
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and not os.path.exists(dst + '.orig'):
            shutil.copy2(dst, dst + '.orig')
        shutil.copy2(src, dst)
        print(f"      {name} -> {dst}")
    return ok


def download():
    path = os.path.join(cfg.SEEM_DIR, cfg.WEIGHTS)
    if os.path.exists(path) and os.path.getsize(path) >= cfg.MIN_WEIGHT_BYTES:
        print(f"[3/3] weights already in {path}")
        return True
    print(f"[3/3] downloading {cfg.WEIGHTS_URL}\n      -> {path} (about 1.3 GB)")

    def hook(n, block, total):
        if total > 0 and n % 200 == 0:
            sys.stdout.write(f"\r      {min(100.0, 100.0 * n * block / total):5.1f}%")
            sys.stdout.flush()
    try:
        urllib.request.urlretrieve(cfg.WEIGHTS_URL, path, hook)
        print()
    except Exception as e:
        print(f"\n      download failed ({e}); download the file by hand and put it in {cfg.SEEM_DIR}")
        return False
    return os.path.getsize(path) >= cfg.MIN_WEIGHT_BYTES


def main():
    ok = clone()
    ok = patch() and ok if ok else False
    ok = download() and ok
    conf = os.path.join(cfg.SEEM_DIR, cfg.CONFIG_FILE)
    if os.path.isdir(cfg.SEEM_DIR) and not os.path.exists(conf):
        print(f"      note: {conf} not found; set Config.CONFIG_FILE in train_rba.py to SEEM's Focal-L demo config")
        ok = False
    print("\nSEEM is ready: python train_rba.py --help" if ok else "\nSEEM is NOT ready yet: see the messages above")


if __name__ == "__main__":
    main()
