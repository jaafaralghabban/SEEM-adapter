# ============================================================================
# train_rba.py -- RBA: a Read-out Basis Adapter for interactive OCTA vessel segmentation with SEEM
#   Trains and tests one configuration ("row") of the paper on one target and dataset.
#
#   python train_rba.py --row H1_lora1_hq --task GT_LargeVessel --subset OCTA_3mm             # train RBA
#   python train_rba.py --row H1_lora1_hq --task GT_LargeVessel --subset OCTA_3mm --run eval  # re-test it
#
#   Rows: H1_lora1_hq = RBA (ours) | H_hq_only_ep100 = RBA read-out branch only | B1_lora1_ep100 = LoRA r=1
#         B_lora = LoRA r=16 | C_convlora = Conv-LoRA r=16 | AF_adaptformer = AdaptFormer | A_decoder_only
#         D_convlora_hq / B_lora_hq = Conv-LoRA / LoRA r=16 + RBA read-out | R_* = RiDA (alternative design)
#         *_frozendec, H1_adapteronly = all of SEEM frozen (Table 2)
#   --run calibre_check / vein_check : the analyses of Tables 8 and 7 (no training)
#   Needs SEEM v1.0 with the two patched files of seem_patches/ (run setup_seem.py once).
#   Every option can also be changed in the Config class below.
# ============================================================================
# ============================================================================
# train_paper1_hq.py  —  Paper 1: image-only lift of SEEM's read-out basis
# ============================================================================
# Built on train_adaptable_adapters.py. Every change is a flag in Config and is
# tagged in the code so each one can be reported as an ablation row:
#
#   [FIX 1] input normalisation   — backbone() was called on raw 0-255 pixels;
#                                   SEEM normalises inside forward(), which the
#                                   trainer bypasses
#   [FIX 2] augmentation          — dihedral group + mild intensity (train only)
#   [FIX 3] native-resolution grid— loss, clicks and metrics on the 304x304 GT,
#                                   the grid the published OCTA-500 numbers use
#   [FIX 4] low-rank Conv-LoRA    — the conv lives inside the rank-r bottleneck
#                                   (the old ParallelConvLoRALinear had a
#                                   full-rank conv_proj: 172.8 M params)
#   [FIX 5] click adapters off    — click-conditioned fusion is Paper-2 scope;
#                                   if used, prototypes come from the stride-8
#                                   level (ms_feats[0] is the stride-32 map)
#   [FIX 6] deployable calibration— one global threshold chosen on the
#                                   validation set, then frozen for test
#   [FIX 7] schedule              — per-group LR, warmup + cosine, EMA,
#                                   frozen modules in eval mode (no drop-path)
#   [IDEA 1] HQ basis read-out    — lifts the basis the prompt selects from to
#                                   stride 2 (or 1) with image-only features;
#                                   zero-initialised, so step 0 == stock model
#   [IDEA 3] RiDA                 — one rank-r code per site, sharpened by a
#                                   parameter-free ridge (vessel) operator, feeds
#                                   both the encoder and the mask read-out
#
# NO SEEM EDIT NEEDED for USE_HQ (the read-out needs w_q = mask_embed of the
#   spatial queries). ReadoutTap passes 'pred_smaskembs' through
#   AttentionDataStruct.update_spatial_results at run time, with a forward hook
#   on predictor.mask_embed as fallback. preflight() verifies
#   einsum(w, F) == pred_smasks, so a wrong w cannot go unnoticed.
#   Manual alternative, if both ever fail: in attention_data_struct_seemv0.py,
#   update_spatial_results(), add  "pred_smaskembs": v_emb  to the returned dict.
#
# Paper-1 rows (use_row(name) then main()):
#   A_decoder_only  B_lora  C_convlora  D_convlora_hq  D1_hq_stride1
#   D2_hq_no_stem   D3_hq_no_res2  E_input1024
#   adapter test (HQ read-out as the adapter): H_hq_only  H1_lora1_hq  B1_lora1
#   [IDEA 3] RiDA, one low-rank vessel adapter: R_rida  R_no_ridge  R_no_readout  R_no_encoder
# Your current script is the "legacy" row — keep its numbers as they are.
#
# Re-test a trained row without training: set  RUN = 'eval'  in Config (same
# RUN_TAG / TASK / SUBSET as the run). It reports the native-grid and old
# input-grid numbers, ceilings at 300/1000/3000 fit steps, and overlays.
# Adapter table: adapter_table.py. RBA = the read-out basis adapter (rows H, H1).
# ============================================================================

import os
import gc
import math
import json
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from PIL import Image
import cv2
from skimage.morphology import skeletonize
from tqdm import tqdm

import sys
SEEM_DIR = os.environ.get('SEEM_DIR', 'SEEM')            # SEEM v1.0 folder made by setup_seem.py
if os.path.isdir(SEEM_DIR) and os.path.abspath(SEEM_DIR) not in sys.path:
    sys.path.insert(0, os.path.abspath(SEEM_DIR))


def _in_seem(rel):
    """A path inside the SEEM folder if it exists there, else the path as given (running inside a SEEM folder)."""
    p = os.path.join(SEEM_DIR, rel)
    return p if os.path.exists(p) else rel

BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    CONFIG_FILE = _in_seem("configs/seem/focall_unicl_lang_demo.yaml")
    PRETRAINED = os.path.join(SEEM_DIR, "seem_focall_v0.pt") if os.path.isdir(SEEM_DIR) else "seem_focall_v0.pt"
    CHECKPOINT_DIR = "checkpoints"
    RUN_TAG = "H1_lora1_hq"             # <-- this row (or --row)

    DECODER_NAME_OVERRIDE = 'seem_v0'
    ATTENTION_ARCH_OVERRIDE = {
        'VARIABLE': {
            'queries': ['object', 'grounding', 'spatial'],
            'tokens': ['grounding', 'spatial'],
            'memories': ['spatial'],
        },
        'SELF_ATTENTION': {
            'queries': {
                'object': ['queries_object'],
                'grounding': ['queries_grounding', 'tokens_grounding'],
                'spatial': ['queries_spatial', 'tokens_spatial', 'memories_spatial'],
            },
            'tokens': {
                'grounding': ['queries_grounding', 'tokens_grounding'],
                'spatial': ['tokens_spatial'],
            },
            'memories': {'spatial': ['memories_spatial']},
        },
        'CROSS_ATTENTION': {
            'queries': {'object': True, 'grounding': True, 'spatial': True},
            'memories': {'spatial': True},
            'tokens': {'grounding': False, 'spatial': False},
        },
        'MASKING': ['tokens_spatial', 'tokens_grounding'],
        'DUPLICATION': {'queries': {'grounding': 'queries_object', 'spatial': 'queries_object'}},
        'SPATIAL_MEMORIES': 32,
        'QUERY_NUMBER': 3,
    }

    # ---- data -------------------------------------------------------------
    DATA_DIR = "octa"
    SUBSET = "OCTA_3mm"                 # <-- OCTA_3mm | OCTA_6mm | ROSE_O (or --subset)
    LABEL_ROOT_REL = "Label/Label"
    TASK = 'GT_LargeVessel'             # <-- target (or --task)
    SPLIT_FILE = "data_split_paper.pth"

    @property
    def OCTA_FULL_DIR(self):    return os.path.join(self.DATA_DIR, self.SUBSET, "OCTA(FULL)")
    @property
    def OCTA_ILM_OPL_DIR(self): return os.path.join(self.DATA_DIR, self.SUBSET, "OCTA(ILM_OPL)")
    @property
    def OCTA_OPL_BM_DIR(self):  return os.path.join(self.DATA_DIR, self.SUBSET, "OCTA(OPL_BM)")
    @property
    def GT_MASK_DIR(self):      return os.path.join(self.DATA_DIR, self.LABEL_ROOT_REL, self.TASK)

    IMG_SIZE = 512
    CLICK_RADIUS_AT_512 = 8             # disk radius in input px at IMG_SIZE=512 (scaled with IMG_SIZE)

    # ---- [FIX 1..3] -------------------------------------------------------
    NORMALIZE_INPUT = True
    AUGMENT = True
    NATIVE_RES = True

    # ---- [FIX 4] encoder PEFT --------------------------------------------
    ENC_ADAPT = 'lora'                  # <-- 'lora' | 'conv_lora' | 'adaptformer' | 'rida' | 'none'
    LORA_RANK = 1                       # <-- LoRA rank (not used by AdaptFormer)
    LORA_ALPHA = 2                      # <-- LoRA scale alpha / rank = 2 in every row
    LORA_DROPOUT = 0.0
    AF_DIM = 12                         # AdaptFormer (row AF): bottleneck width -> 451,488 params, the size of RBA
    AF_SCALE = 2.0                      # AdaptFormer output scale = alpha / r of the LoRA rows, so the shared
                                        # LR_ENCODER treats it like them (AdaptFormer's own default is 0.1)

    # ---- [FIX 5] click-conditioned adapters (Paper-2 scope) --------------
    CLICK_ADAPTER = 'none'              # 'none' | 'standard' | 'prototype'

    # ---- [IDEA 1] HQ basis read-out --------------------------------------
    USE_HQ = True                       # <-- True = with the RBA read-out
    HQ_STRIDE = 2                       # 2 -> 256x256 at 512 input; 1 -> 512x512
    HQ_RANK = 32
    HQ_USE_STEM = True                  # full-res CNN stem on the image
    HQ_USE_RES2 = True                  # FocalNet stage-1 features (stride 4, pre pixel-decoder)
    HQ_USE_MASKFEAT = True              # the stock read-out basis itself
    MEMORY_FROM_HQ = True               # prev_mask memory = area-downsampled HQ mask
    READOUT = 'hq'                      # <-- 'hq' = RBA's read-out | 'rida' = RiDA's

    # ---- [IDEA 3] RiDA (ENC_ADAPT='rida'; READOUT='rida' with USE_HQ=True) ----
    RIDA_RIDGE = True                   # parameter-free ridge (vessel) operator in the codes
    RIDA_ENC_UPDATE = True              # encoder path (B, gamma); False = read-out path only
    RIDA_TAP_STAGES = (0, 1)            # encoder stages whose site codes the read-out reads (stride 4, 8)
    RIDA_SCALES_ENC = (1.0, 2.0)        # ridge scales at encoder sites (token-grid px)
    RIDA_PIX_RANK = 4                   # rank of the full-resolution pixel code
    RIDA_SCALES_PIX = (1.0, 2.0, 4.0)   # ridge scales of the pixel code (input px at IMG_SIZE)
    RIDA_F_RANK = 4                     # rank of the code of SEEM's mask features F
    RIDA_SCALES_F = (1.0, 2.0)          # ridge scales of the F code (F-grid px, stride 4)

    # ---- [FIX 6] deployable calibration ----------------------------------
    THRESH_GRID = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

    # ---- [FIX 7] optimisation ---------------------------------------------
    EPOCHS = 100                        # keep 100: every row in the table is trained for 100 epochs
    WARMUP_EPOCHS = 3
    SCHEDULER = 'cosine'                # 'cosine' | 'plateau' (old)
    LR_DECODER = 5e-5                   # pretrained SEEM predictor
    LR_ENCODER = 2e-4                   # LoRA / Conv-LoRA
    LR_NEW = 5e-4                       # HQ basis, click adapter (fresh modules)
    WEIGHT_DECAY = 1e-4
    USE_EMA = True
    EMA_DECAY = 0.999
    FROZEN_EVAL_MODE = True             # frozen backbone/pixel-decoder in eval mode (no drop-path 0.3)
    TRAIN_LANG_ENCODER = False          # 63 M params that never receive gradient in the spatial task
    DECODER_TRAIN = 'full'              # <-- what of SEEM's decoder trains:
                                        #   'full'     the whole decoder (40.4 M) - every earlier row
                                        #   'new_only' SEEM frozen; only the learned positional click prompt
                                        #              (566,786, new, not in SEEM's checkpoint) trains with the adapter
                                        #   'none'     only the adapter trains (needs USE_LEARNED_POS_PROMPT = False)

    BATCH_SIZE = 1
    K_ROUNDS = 4                        # clicks per training session and the headline Dice@K
    K_EVAL = 20                         # final-test curve length (NoC@85/90)
    IMPROVE_W = 0.5
    ORACLE_BEST_CANDIDATE = True        # TRAINING ONLY. Never used by evaluate()
    RUN_RANDOM_CONTROL = True           # random-in-error click policy on test (causal control)
    RESUME = True                       # continue from checkpoints/last_<row>_<task>_<subset>.pth (a run that stops continues)
    SAVE_LAST = True                    # write that file after every epoch
    RUN_CEILING = True                  # reachable-set ceiling on test (diagnostic, uses GT by design)
    CEILING_STEPS = 300
    RUN = 'train'                       # <-- 'train' | 'eval' | 'calibre_check' | 'vein_check' (or --run)
    EVAL_CEILING_STEPS = (300, 1000, 3000)   # eval mode: ceiling at several fit lengths (convergence check)
    SAVE_OVERLAYS = True                # eval mode: worst/best test images, TP white / FN blue / FP orange
    NUM_WORKERS = 0

    USE_LEARNED_POS_PROMPT = True
    POS_PROMPT_ON_TOKENS = True
    POS_PROMPT_ON_MEANQ = True


cfg = Config()

ROWS = {
    'A_decoder_only': dict(ENC_ADAPT='none', USE_HQ=False),
    'B_lora':         dict(ENC_ADAPT='lora', USE_HQ=False),
    'C_convlora':     dict(ENC_ADAPT='conv_lora', USE_HQ=False),
    'D_convlora_hq':  dict(ENC_ADAPT='conv_lora', USE_HQ=True, HQ_STRIDE=2),
    'D1_hq_stride1':  dict(ENC_ADAPT='conv_lora', USE_HQ=True, HQ_STRIDE=1),
    'D2_hq_no_stem':  dict(ENC_ADAPT='conv_lora', USE_HQ=True, HQ_USE_STEM=False),
    'D3_hq_no_res2':  dict(ENC_ADAPT='conv_lora', USE_HQ=True, HQ_USE_RES2=False),
    'E_input1024':    dict(ENC_ADAPT='conv_lora', USE_HQ=False, IMG_SIZE=1024),
    # adapter test: the HQ read-out as the adapter, against Conv-LoRA (C) and a same-size LoRA (B1)
    'H_hq_only':      dict(ENC_ADAPT='none', USE_HQ=True),                              # 0.20 M
    'H1_lora1_hq':    dict(ENC_ADAPT='lora', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=True),   # 0.47 M
    'B1_lora1':       dict(ENC_ADAPT='lora', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=False),  # 0.27 M
    # 100-epoch reruns (the first H and B1 runs used EPOCHS = 50; C, D, H1 used 100)
    'H_hq_only_ep100': dict(ENC_ADAPT='none', USE_HQ=True),
    'B1_lora1_ep100':  dict(ENC_ADAPT='lora', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=False),
    # RiDA: one low-rank vessel adapter (full) and its three ablations
    'R_rida':          dict(ENC_ADAPT='rida', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=True, READOUT='rida'),
    'R_no_ridge':      dict(ENC_ADAPT='rida', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=True, READOUT='rida',
                            RIDA_RIDGE=False),
    'R_no_readout':    dict(ENC_ADAPT='rida', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=False, READOUT='rida'),
    'R_no_encoder':    dict(ENC_ADAPT='rida', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=True, READOUT='rida',
                            RIDA_ENC_UPDATE=False),
    # comparison rows for RBA, the read-out basis adapter (= rows H and H1)
    'AF_adaptformer':  dict(ENC_ADAPT='adaptformer', USE_HQ=False),       # AdaptFormer, 451,488 params
    'B_lora_hq':       dict(ENC_ADAPT='lora', USE_HQ=True),               # LoRA r=16 + RBA read-out
    # SEEM frozen (Table 2): only the adapter and the learned positional click prompt train
    'H1_frozendec':         dict(ENC_ADAPT='lora', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=True, DECODER_TRAIN='new_only'),
    'C_convlora_frozendec': dict(ENC_ADAPT='conv_lora', USE_HQ=False, DECODER_TRAIN='new_only'),
    'B1_lora1_frozendec':   dict(ENC_ADAPT='lora', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=False, DECODER_TRAIN='new_only'),
    'H1_adapteronly':       dict(ENC_ADAPT='lora', LORA_RANK=1, LORA_ALPHA=2, USE_HQ=True, DECODER_TRAIN='none',
                                 USE_LEARNED_POS_PROMPT=False),              # only RBA trains (0.47 M)
}


def use_row(name):
    """Reset the ablation-relevant flags to the default and apply one row."""
    defaults = dict(ENC_ADAPT='conv_lora', LORA_RANK=16, LORA_ALPHA=32, USE_HQ=True, HQ_STRIDE=2,
                    HQ_USE_STEM=True, HQ_USE_RES2=True, HQ_USE_MASKFEAT=True, IMG_SIZE=512,
                    READOUT='hq', RIDA_RIDGE=True, RIDA_ENC_UPDATE=True, DECODER_TRAIN='full',
                    USE_LEARNED_POS_PROMPT=True)
    for k, v in {**defaults, **ROWS[name]}.items():
        setattr(cfg, k, v)
    cfg.RUN_TAG = name
    print(f"row {name}: " + ", ".join(f"{k}={getattr(cfg, k)}" for k in sorted(defaults)))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ============================================================================
# DATASET   [FIX 2] augmentation, [FIX 3] native-resolution GT
# ============================================================================
def augment_pair(img_np, gt_np, rng=np.random):
    """Label-preserving for en-face OCTA: 8-element dihedral group + mild intensity."""
    k = rng.randint(4)
    img_np = np.rot90(img_np, k, axes=(0, 1))
    gt_np = np.rot90(gt_np, k, axes=(0, 1))
    if rng.rand() < 0.5:
        img_np = img_np[:, ::-1]
        gt_np = gt_np[:, ::-1]
    img = img_np.astype(np.float32)
    if rng.rand() < 0.8:
        gamma = float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
        img = 255.0 * (img / 255.0) ** gamma
        c = rng.uniform(0.9, 1.1, size=(1, 1, img.shape[2]))
        b = rng.uniform(-10.0, 10.0, size=(1, 1, img.shape[2]))
        img = img * c + b
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(gt_np)


class OCTADataset(Dataset):
    def __init__(self, ids, augment=False):
        self.ids = list(ids)
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _load_3channel(self, sid):
        projs = []
        for d in [cfg.OCTA_FULL_DIR, cfg.OCTA_ILM_OPL_DIR, cfg.OCTA_OPL_BM_DIR]:
            p = cv2.imread(os.path.join(d, sid), cv2.IMREAD_GRAYSCALE)
            if p is None:
                raise FileNotFoundError(f"Missing: {os.path.join(d, sid)}")
            projs.append(p)
        return np.stack(projs, axis=2)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img_np = self._load_3channel(sid)
        gt_np = (np.array(Image.open(os.path.join(cfg.GT_MASK_DIR, sid)).convert("L")) > 0).astype(np.uint8)
        if gt_np.shape != img_np.shape[:2]:
            raise ValueError(f"{sid}: GT {gt_np.shape} vs image {img_np.shape[:2]}")
        if self.augment:
            img_np, gt_np = augment_pair(img_np, gt_np)
        S = cfg.IMG_SIZE
        img_rs = np.array(Image.fromarray(img_np).resize((S, S), BICUBIC))
        img_t = torch.from_numpy(img_rs).permute(2, 0, 1).float()        # 0..255, normalised in encode()
        gt_t = torch.from_numpy(gt_np).float()                            # native grid (304x304 for 3 mm)
        if not cfg.NATIVE_RES:                                            # legacy: GT on the input grid
            gt_t = F.interpolate(gt_t[None, None], size=(S, S), mode='nearest')[0, 0]
        return img_t, gt_t, sid


def make_splits():
    split = safe_torch_load(os.path.join(cfg.CHECKPOINT_DIR, cfg.SPLIT_FILE), map_location='cpu')
    train, val, test = split['train'], split['val'], split['test']
    assert not (set(train) & set(test)) and not (set(train) & set(val)) and not (set(val) & set(test))
    return train, val, test


# ============================================================================
# ENCODER PEFT   [FIX 4]
# ============================================================================
def _as_map(h):
    """[B,H,W,C] -> ([B,C,H,W], restore) ; [B,L,C] with square L -> same ; else None."""
    if h.dim() == 4:
        return h.permute(0, 3, 1, 2), lambda y: y.permute(0, 2, 3, 1)
    if h.dim() == 3:
        B, L, C = h.shape
        s = int(round(math.sqrt(L)))
        if s * s == L:
            return (h.transpose(1, 2).reshape(B, C, s, s),
                    lambda y: y.reshape(B, C, L).transpose(1, 2))
    return None, None


class LoRALinear(nn.Module):
    """W x + (alpha/r) * B( h + DW3x3(h) ),  h = A x.
    conv=False -> plain LoRA. conv=True -> Conv-LoRA: the depthwise 3x3 acts on the
    rank-r bottleneck (Zhong et al., ICLR 2024), so it adds 9*r params per site
    instead of in*out. B and the conv are zero-initialised -> identity at init."""

    def __init__(self, base, r, alpha, dropout, conv=False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Linear(base.in_features, r, bias=False).to(dev, dt)
        self.lora_B = nn.Linear(r, base.out_features, bias=False).to(dev, dt)
        self.lora_dw = nn.Conv2d(r, r, 3, padding=1, groups=r, bias=False).to(dev, dt) if conv else None
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_dw is not None:
            nn.init.zeros_(self.lora_dw.weight)
        self.n_spatial_calls = 0

    def forward(self, x):
        out = self.base(x)
        h = self.lora_A(self.dropout(x))
        if self.lora_dw is not None:
            hm, back = _as_map(h)
            if hm is not None:
                h = h + back(self.lora_dw(hm.contiguous())).contiguous()
                self.n_spatial_calls += 1
        return out + self.lora_B(h) * self.scale


class ParallelConvLoRALinear(nn.Module):
    """The ORIGINAL adapter, kept only as an ablation row ('fullrank_conv').
    conv_proj is a full in_f x out_f matrix: 172,798,848 params on FocalNet-L."""

    def __init__(self, base_layer, r, alpha, dropout):
        super().__init__()
        self.base = base_layer
        dev, dt = base_layer.weight.device, base_layer.weight.dtype
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base_layer.in_features, base_layer.out_features
        self.lora_A = nn.Linear(in_f, r, bias=False).to(dev, dt)
        self.lora_B = nn.Linear(r, out_f, bias=False).to(dev, dt)
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_f, in_f, 3, padding=1, groups=in_f, bias=False).to(dev, dt)
        self.conv_proj = nn.Linear(in_f, out_f, bias=False).to(dev, dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        nn.init.dirac_(self.conv.weight)
        nn.init.zeros_(self.conv_proj.weight)

    def forward(self, x):
        out = self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        xm, back = _as_map(x)
        if xm is not None:
            out = out + self.conv_proj(back(self.conv(xm.contiguous())).contiguous())
        return out


class AdaptMLP(nn.Module):
    """AdaptFormer (Chen et al., NeurIPS 2022): a bottleneck branch parallel to a block's MLP,
         y = MLP(x) + s * up(ReLU(down(x))).
    up is zero-initialised, so every block is unchanged at step 0 (the same rule as LoRA's B)."""

    def __init__(self, mlp, d_in, d_out, bottleneck, scale):
        super().__init__()
        self.mlp = mlp
        self.down = nn.Linear(d_in, bottleneck)
        self.up = nn.Linear(bottleneck, d_out)
        self.scale = float(scale)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.mlp(x) + self.scale * self.up(F.relu(self.down(x)))


def inject_adaptformer(model):
    """Wrap every FocalNet MLP (a module with fc1 and fc2) in an AdaptMLP: 24 blocks on FocalNet-L."""
    sites = []
    for _, parent in model.model.backbone.named_modules():
        for cname, child in parent.named_children():
            if isinstance(getattr(child, 'fc1', None), nn.Linear) and isinstance(getattr(child, 'fc2', None), nn.Linear):
                sites.append((parent, cname, child))
    if not sites:
        raise RuntimeError("AdaptFormer: no MLP with fc1 / fc2 found in the backbone")
    for parent, cname, mlp in sites:                    # replace after collecting (no mutation while iterating)
        setattr(parent, cname, AdaptMLP(mlp, mlp.fc1.in_features, mlp.fc2.out_features, cfg.AF_DIM, cfg.AF_SCALE))
    return {'mlp': len(sites)}


def inject_encoder_adapters(model, kind, r, alpha, dropout):
    model._rida_taps = []                                # RiDA encoder sites whose codes the read-out reads
    if kind == 'none':
        return {}
    if kind == 'adaptformer':
        return inject_adaptformer(model)
    targets = []
    for mname, module in model.model.backbone.named_modules():
        for child_name in ('f', 'fc1', 'fc2', 'proj'):
            child = getattr(module, child_name, None)
            if isinstance(child, nn.Linear):
                targets.append((mname, module, child_name, child))
    counts = {'f': 0, 'fc1': 0, 'fc2': 0, 'proj': 0}
    rida_readout = cfg.USE_HQ and cfg.READOUT == 'rida'
    for mname, module, child_name, child in targets:    # replace after collecting (no mutation while iterating)
        if kind == 'lora':
            new = LoRALinear(child, r, alpha, dropout, conv=False)
        elif kind == 'conv_lora':
            new = LoRALinear(child, r, alpha, dropout, conv=True)
        elif kind == 'fullrank_conv':
            new = ParallelConvLoRALinear(child, r, alpha, dropout)
        elif kind == 'rida':
            tap = rida_readout and _stage_of(mname) in tuple(cfg.RIDA_TAP_STAGES)
            if not cfg.RIDA_ENC_UPDATE and not tap:
                continue                                 # no encoder update and nobody reads this code
            new = RidaLinear(child, r, alpha, cfg.RIDA_SCALES_ENC, ridge=cfg.RIDA_RIDGE, tap=tap,
                             enc_update=cfg.RIDA_ENC_UPDATE)
            if tap:
                model._rida_taps.append(new)
        else:
            raise ValueError(f"unknown ENC_ADAPT {kind}")
        setattr(module, child_name, new)
        counts[child_name] += 1
    if kind == 'rida':
        counts['tapped by the read-out'] = len(model._rida_taps)
    return counts


# ============================================================================
# [IDEA 1] HQ BASIS READ-OUT
# ============================================================================
class HQBasis(nn.Module):
    """Image-only, click-independent lift of the read-out basis.

    Stock SEEM:  logits_q = < w_q , F >                 F   : [512, S/4, S/4]
    Lifted:      logits_q = < w_q , Up(F) + U G >       G   : [r, S/s, S/s]  (image only)
                          = Up(<w_q, F>) + < U^T w_q , G >   U : [512 -> r], zero-init

    Bilinear interpolation is linear and commutes with the channel contraction,
    so with U = 0 the lifted logits equal Up(stock logits) exactly: step 0 is the
    stock model. The prompt still chooses only w_q; what changes is the
    image-determined basis it chooses from, B' = Up(F) + U^T G.

    Precise property (for the paper): for any w in null(U) the lifted mask equals
    the stock mask, so the lifted family keeps a (C - r)-dimensional part of the
    stock family exactly and re-assigns r read-out directions to r image-only,
    high-resolution maps. Whether that trade raises the ceiling is measured, not
    assumed: see reachable_ceiling().
    """

    def __init__(self, c_mask=512, c_res2=192, rank=32, stride=2,
                 use_stem=True, use_res2=True, use_maskfeat=True, c_stem=32, c_mid=64):
        super().__init__()
        assert stride in (1, 2), "HQ_STRIDE must be 1 or 2"
        assert use_stem or use_res2 or use_maskfeat
        self.stride = stride
        self.use_stem, self.use_res2, self.use_maskfeat = use_stem, use_res2, use_maskfeat
        c_in = 0
        if use_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(3, c_stem, 3, stride=1, padding=1), nn.GroupNorm(8, c_stem), nn.GELU(),
                nn.Conv2d(c_stem, c_stem, 3, stride=stride, padding=1), nn.GroupNorm(8, c_stem), nn.GELU())
            c_in += c_stem
        if use_res2:
            self.p_res2 = nn.Conv2d(c_res2, c_mid, 1)
            c_in += c_mid
        if use_maskfeat:
            self.p_mask = nn.Conv2d(c_mask, c_mid, 1)
            c_in += c_mid
        self.fuse = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 3, padding=1), nn.GroupNorm(8, c_mid), nn.GELU(),
            nn.Conv2d(c_mid, c_mid, 3, padding=1), nn.GroupNorm(8, c_mid), nn.GELU(),
            nn.Conv2d(c_mid, rank, 1))
        self.U = nn.Linear(c_mask, rank, bias=False)
        nn.init.zeros_(self.U.weight)

    def basis(self, img_raw, res2, mask_features):
        """img_raw: [B,3,S,S] in 0..255 (own fixed normalisation). Returns G: [B,r,S/s,S/s]."""
        S = img_raw.shape[-2:]
        size = (S[0] // self.stride, S[1] // self.stride)
        parts = []
        if self.use_stem:
            parts.append(self.stem(img_raw / 127.5 - 1.0))
        if self.use_res2:
            parts.append(F.interpolate(self.p_res2(res2), size=size, mode='bilinear', align_corners=False))
        if self.use_maskfeat:
            parts.append(F.interpolate(self.p_mask(mask_features), size=size, mode='bilinear', align_corners=False))
        return self.fuse(torch.cat(parts, dim=1))

    def readout(self, stock_logits, w, G):
        """stock_logits [B,Q,h,w] (stride 4), w [B,Q,512], G [B,r,H,W] -> [B,Q,H,W]."""
        up = F.interpolate(stock_logits, size=G.shape[-2:], mode='bilinear', align_corners=False)
        return up + torch.einsum('bqr,brhw->bqhw', self.U(w), G)


# ============================================================================
# [IDEA 3] RiDA — Ridge-informed Dual-path Adapter (one low-rank vessel adapter)
# ============================================================================
#   Every site computes a rank-r CODE of its input, h = A x, and a parameter-free
#   vessel (ridge) response of that code, V_s(h). The same code feeds two paths:
#     encoder path : y = W x + (alpha/r) B ( h + sum_s gamma_s V_s(h) )   B = 0, gamma = 0 at init
#     read-out path: m_q = Up(<w_q, F>) + < U w_q , G >,  G = Psi(codes)  U = 0 at init
#   With B = U = 0 the model is exactly stock SEEM; with gamma = 0 and no read-out
#   path it is exactly LoRA of rank r.
#   V_s(h) = s^2 ( -lam_min - relu(-lam_max) ), lam = eigenvalues of the Hessian of
#   G_s * h: angle-independent, ridge-selective (a round blob gives ~0), tuned to a
#   width of ~sqrt(2)*s. A linear filter that treats all angles alike (Laplacian-like)
#   responds MORE to blobs than to vessels; this operator does not, and has no weights.
def _stage_of(module_name):
    """'layers.1.blocks.0.mlp' (FocalNet) or 'stages.1.0' (mock) -> 1; unknown -> -1."""
    import re
    m = re.search(r'(?:^|\.)(?:layers|stages)\.(\d+)(?:\.|$)', module_name)
    return int(m.group(1)) if m else -1


def _gauss_deriv_1d(sigma):
    rad = max(1, int(math.ceil(3.5 * sigma)))
    x = torch.arange(-rad, rad + 1, dtype=torch.float32)
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    g1 = -x / sigma ** 2 * g
    g2 = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    return g, g1, g2


class RidgeOperator(nn.Module):
    """Parameter-free multi-scale ridge response of every channel of h: [B,C,H,W] -> list of [B,C,H,W].
    Hessian by separable Gaussian derivatives (replicate padding, so any map size works)."""

    def __init__(self, scales, eps=1e-8):
        super().__init__()
        self.scales = tuple(float(s) for s in scales)
        self.eps = eps
        for i, s in enumerate(self.scales):
            g, g1, g2 = _gauss_deriv_1d(s)
            self.register_buffer(f'kx{i}', torch.stack([g2, g, g1]), persistent=False)   # x-pass: xx, yy, xy
            self.register_buffer(f'ky{i}', torch.stack([g, g2, g1]), persistent=False)   # y-pass

    def forward(self, h):
        B, C, H, W = h.shape
        outs = []
        for i, s in enumerate(self.scales):
            kx, ky = getattr(self, f'kx{i}').to(h.dtype), getattr(self, f'ky{i}').to(h.dtype)
            k = kx.shape[1]
            p = k // 2
            wx = kx.repeat(C, 1).view(3 * C, 1, 1, k)
            wy = ky.repeat(C, 1).view(3 * C, 1, k, 1)
            t = F.conv2d(F.pad(h, (p, p, 0, 0), mode='replicate'), wx, groups=C)      # [B,3C,H,W]
            t = F.conv2d(F.pad(t, (0, 0, p, p), mode='replicate'), wy, groups=3 * C)
            hxx, hyy, hxy = t.view(B, C, 3, H, W).unbind(2)
            tr = 0.5 * (hxx + hyy)
            rad = torch.sqrt(0.25 * (hxx - hyy) ** 2 + hxy ** 2 + self.eps)
            lmin, lmax = tr - rad, tr + rad
            outs.append((s * s) * (-lmin - F.relu(-lmax)))
        return outs


class RidaLinear(nn.Module):
    """RiDA encoder site on a frozen linear layer W (rank r, like LoRA's A and B).
      y = W x + (alpha/r) * B( h + sum_s gamma_s V_s(h) ),  h = A x
    ridge=False     -> no V (the site is plain LoRA; its code is h only)
    tap=True        -> keeps the code z = [h, V_s1(h), ...] ([B, r(1+K), H, W]) for the read-out path
    enc_update=False-> no B, no gamma: the layer is unchanged; A is trained only through the read-out."""

    def __init__(self, base, r, alpha, scales, ridge=True, tap=False, enc_update=True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        dev, dt = base.weight.device, base.weight.dtype
        self.r = r
        self.lora_A = nn.Linear(base.in_features, r, bias=False).to(dev, dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        self.lora_B = None
        if enc_update:
            self.lora_B = nn.Linear(r, base.out_features, bias=False).to(dev, dt)
            nn.init.zeros_(self.lora_B.weight)
        self.ridge = RidgeOperator(scales).to(dev) if (ridge and len(scales) > 0) else None
        self.gamma = (nn.Parameter(torch.zeros(len(scales), device=dev, dtype=dt))
                      if (self.ridge is not None and enc_update) else None)
        self.scale = alpha / r
        self.tap = tap
        self.code_channels = r * (1 + (len(scales) if self.ridge is not None else 0))
        self.last_code = None
        self.n_spatial_calls = 0

    def forward(self, x):
        out = self.base(x)
        h = self.lora_A(x)
        if self.ridge is None and not self.tap:
            return out + self.lora_B(h) * self.scale if self.lora_B is not None else out
        hm, back = _as_map(h)
        if hm is None:                                   # not a spatial token map: plain low-rank update
            self.last_code = None
            return out + self.lora_B(h) * self.scale if self.lora_B is not None else out
        hm = hm.contiguous()
        V = self.ridge(hm) if self.ridge is not None else []
        self.n_spatial_calls += 1
        if self.tap:
            self.last_code = torch.cat([hm] + V, dim=1)
        if self.lora_B is None:
            return out
        if V:
            u = hm + sum(g * v for g, v in zip(self.gamma, V))
            h = back(u).contiguous()
        return out + self.lora_B(h) * self.scale


class RidaReadout(nn.Module):
    """RiDA read-out path. Same interface as HQBasis (basis / readout / U / stride / use_res2),
    so every place that uses the HQ read-out (training, test, ceilings, preflight) works unchanged.
      G = Psi( [ pixel code, F code, codes of the tapped encoder sites ] )      at stride `stride`
      logits_q = Up(<w_q, F>) + < U w_q , G >,   U = 0 at init
    pixel code: h0 = A0 x_img at full resolution (+ ridge maps, scales in input px)
    F code    : hF = AF F      (+ ridge maps, scales in F-grid px)
    site codes: stored by the RidaLinear sites; their A is shared with the encoder path.
    Psi       : per-map GroupNorm -> 1x1 mix -> GELU -> depthwise 3x3."""
    use_res2 = False

    def __init__(self, c_mask, taps, rank=32, stride=2, pix_rank=4, f_rank=4,
                 scales_pix=(1.0, 2.0, 4.0), scales_f=(1.0, 2.0), ridge=True):
        super().__init__()
        assert stride in (1, 2), "HQ_STRIDE must be 1 or 2"
        self.stride = stride
        self._taps = list(taps)          # plain list: the sites belong to the backbone, not to this module
        self.pix = nn.Conv2d(3, pix_rank, 1)
        self.fcode = nn.Conv2d(c_mask, f_rank, 1)
        self.ridge_pix = RidgeOperator(scales_pix) if (ridge and len(scales_pix) > 0) else None
        self.ridge_f = RidgeOperator(scales_f) if (ridge and len(scales_f) > 0) else None
        n_pix = pix_rank * (1 + (len(scales_pix) if self.ridge_pix is not None else 0))
        n_f = f_rank * (1 + (len(scales_f) if self.ridge_f is not None else 0))
        self.c_in = n_pix + n_f + sum(t.code_channels for t in self._taps)
        self.norm = nn.GroupNorm(self.c_in, self.c_in)
        self.mix = nn.Conv2d(self.c_in, rank, 1)
        self.dw = nn.Conv2d(rank, rank, 3, padding=1, groups=rank)
        self.U = nn.Linear(c_mask, rank, bias=False)
        nn.init.zeros_(self.U.weight)

    def basis(self, img_raw, res2, mask_features):
        """img_raw [B,3,S,S] in 0..255; the tapped sites must hold the codes of THIS image
        (encode() runs the backbone first). Returns G [B,rank,S/s,S/s]."""
        S = img_raw.shape[-2:]
        size = (S[0] // self.stride, S[1] // self.stride)
        h0 = self.pix(img_raw / 127.5 - 1.0)
        z0 = torch.cat([h0] + (self.ridge_pix(h0) if self.ridge_pix is not None else []), 1)
        parts = [z0 if tuple(z0.shape[-2:]) == size else F.interpolate(z0, size=size, mode='area')]
        hf = self.fcode(mask_features)
        zf = torch.cat([hf] + (self.ridge_f(hf) if self.ridge_f is not None else []), 1)
        parts.append(F.interpolate(zf, size=size, mode='bilinear', align_corners=False))
        for t in self._taps:
            z = t.last_code
            if z is None or z.shape[0] != img_raw.shape[0]:
                raise RuntimeError("a RiDA site holds no code for this image: run the backbone first")
            parts.append(F.interpolate(z, size=size, mode='bilinear', align_corners=False))
        g = torch.cat(parts, 1)
        return self.dw(F.gelu(self.mix(self.norm(g))))

    def readout(self, stock_logits, w, G):
        up = F.interpolate(stock_logits, size=G.shape[-2:], mode='bilinear', align_corners=False)
        return up + torch.einsum('bqr,brhw->bqhw', self.U(w), G)


# ============================================================================
# CLICK-CONDITIONED ADAPTERS  [FIX 5]  — Paper-2 scope, default OFF
# ============================================================================
class EncoderToPromptAdapter(nn.Module):
    def __init__(self, mask_channels=512, hidden_dim=256):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(mask_channels + 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim), nn.ReLU(inplace=True),        # was BatchNorm at batch size 1
            nn.Conv2d(hidden_dim, mask_channels, 3, padding=1), nn.Sigmoid())
        self.feat_proj = nn.Conv2d(mask_channels, mask_channels, 1)
        nn.init.zeros_(self.feat_proj.weight); nn.init.zeros_(self.feat_proj.bias)  # identity at init

    def forward(self, mask_features, highres_features, pos_mask, neg_mask):
        H, W = mask_features.shape[-2:]
        pos_m = F.interpolate(pos_mask.float(), size=(H, W), mode='bilinear', align_corners=False)
        neg_m = F.interpolate(neg_mask.float(), size=(H, W), mode='bilinear', align_corners=False)
        attn = self.spatial_conv(torch.cat([mask_features, pos_m, neg_m], dim=1))
        return mask_features + self.feat_proj(mask_features) * attn


class PrototypeGuidedAdapter(nn.Module):
    def __init__(self, mask_channels=512, highres_channels=512, hidden_dim=256):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(mask_channels + highres_channels + 4, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, mask_channels, 3, padding=1), nn.Sigmoid())
        self.feat_proj = nn.Conv2d(mask_channels, mask_channels, 1)
        nn.init.zeros_(self.feat_proj.weight); nn.init.zeros_(self.feat_proj.bias)

    def forward(self, mask_features, highres_features, pos_mask, neg_mask):
        H, W = mask_features.shape[-2:]
        pos_m = F.interpolate(pos_mask.float(), size=(H, W), mode='bilinear', align_corners=False)
        neg_m = F.interpolate(neg_mask.float(), size=(H, W), mode='bilinear', align_corners=False)
        hr_m = F.interpolate(highres_features, size=(H, W), mode='bilinear', align_corners=False)
        pos_proto = (hr_m * pos_m).sum((2, 3), keepdim=True) / (pos_m.sum((2, 3), keepdim=True) + 1e-6)
        neg_proto = (hr_m * neg_m).sum((2, 3), keepdim=True) / (neg_m.sum((2, 3), keepdim=True) + 1e-6)
        hr_n = F.normalize(hr_m, dim=1)
        pos_sim = (hr_n * F.normalize(pos_proto, dim=1)).sum(1, keepdim=True) * (pos_m.amax((2, 3), keepdim=True) > 0)
        neg_sim = (hr_n * F.normalize(neg_proto, dim=1)).sum(1, keepdim=True) * (neg_m.amax((2, 3), keepdim=True) > 0)
        attn = self.spatial_conv(torch.cat([mask_features, hr_m, pos_m, neg_m, pos_sim, neg_sim], dim=1))
        return mask_features + self.feat_proj(mask_features) * attn


# ============================================================================
# PRETRAINED WEIGHTS + MODEL BUILD
# ============================================================================
SEEM_URL = "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt"
MIN_CKPT_BYTES = 100 * 1024 * 1024


def ensure_pretrained(path=None, url=SEEM_URL):
    """Download SEEM's Focal-L checkpoint once. setup_seem.py does the same."""
    import urllib.request
    path = path or cfg.PRETRAINED
    if os.path.exists(path) and os.path.getsize(path) >= MIN_CKPT_BYTES:
        return path
    if os.path.exists(path):
        os.remove(path)
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    if not os.path.exists(path) or os.path.getsize(path) < MIN_CKPT_BYTES:
        raise FileNotFoundError("could not fetch the SEEM checkpoint")
    return path


NEW_PARAM_NAMES = set()      # predictor params absent from the checkpoint -> trained at LR_NEW, not LR_DECODER


def report_load(missing, unexpected):
    """load_state_dict(strict=False) is silent. A backbone key in `missing` means
    the encoder is NOT pretrained — that alone would sink every number."""
    def group(keys):
        g = {}
        for k in keys:
            top = '.'.join(k.split('.')[:2])
            g[top] = g.get(top, 0) + 1
        return g
    print(f"  pretrained load: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"    missing by module: {group(missing)}")
        if len(missing) <= 20:
            print(f"    missing (new, trained from scratch): {list(missing)}")
    if unexpected:
        print(f"    unexpected by module: {group(unexpected)}")
        if len(unexpected) <= 20:
            print(f"    unexpected (ignored): {list(unexpected)}")
    bad = [k for k in missing if k.startswith('backbone.') or k.startswith('sem_seg_head.pixel_decoder.')]
    if len(bad) > 10:
        raise RuntimeError(f"{len(bad)} backbone/pixel-decoder weights missing from the checkpoint "
                           f"(e.g. {bad[:3]}) — the encoder would be random.")
    if bad:
        print(f"    WARNING: {len(bad)} encoder keys missing: {bad}")


def build_seem():
    """Real SEEM. Imports are local so the rest of the file can be tested without SEEM."""
    import importlib
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utils.arguments import load_opt_from_config_files
    from utils.distributed import init_distributed
    from utils.constants import COCO_PANOPTIC_CLASSES
    import modeling.interface.seem_v0
    import modeling.interface.prototype.attention_data_struct_seemv0
    importlib.reload(modeling.interface.prototype.attention_data_struct_seemv0)
    importlib.reload(modeling.interface.seem_v0)

    opt = load_opt_from_config_files([cfg.CONFIG_FILE])
    opt['MODEL']['DECODER']['NAME'] = cfg.DECODER_NAME_OVERRIDE
    opt['ATTENTION_ARCH'] = cfg.ATTENTION_ARCH_OVERRIDE
    opt = init_distributed(opt)
    ensure_pretrained()
    model = BaseModel(opt, build_model(opt))
    base = safe_torch_load(cfg.PRETRAINED, map_location='cpu')
    res = model.model.load_state_dict(base.get('model', base), strict=False)
    report_load(res.missing_keys, res.unexpected_keys)
    NEW_PARAM_NAMES.clear()
    NEW_PARAM_NAMES.update(res.missing_keys)
    model.to(cfg.DEVICE)
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    return model


# ============================================================================
# CLICKS / RENDERING  (clicks live on the evaluation grid; rendered on the input grid)
# ============================================================================
def dt_peak(mask_bin):
    m = mask_bin.astype(np.uint8)
    if m.sum() == 0:
        return None
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    idx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return int(idx[1]), int(idx[0])


def seed_first_click(gt_bin):
    p = dt_peak(gt_bin)
    return p if p is not None else (gt_bin.shape[1] // 2, gt_bin.shape[0] // 2)


def _suppress(dt_map, history_xy):
    for (hx, hy) in history_xy:
        dt_map[max(0, hy - 3):hy + 4, max(0, hx - 3):hx + 4] = 0


def select_giir_click(curr_mask_bin, gt_bin, history_xy, rng=np.random):
    fn = np.logical_and(gt_bin, ~curr_mask_bin).astype(np.uint8)
    fp = np.logical_and(curr_mask_bin, ~gt_bin).astype(np.uint8)
    if fn.sum() == 0 and fp.sum() == 0:
        return None
    fn_dt = cv2.distanceTransform(fn, cv2.DIST_L2, 5) if fn.sum() > 0 else np.zeros(fn.shape, np.float32)
    fp_dt = cv2.distanceTransform(fp, cv2.DIST_L2, 5) if fp.sum() > 0 else np.zeros(fp.shape, np.float32)
    _suppress(fn_dt, history_xy); _suppress(fp_dt, history_xy)
    fn_max, fp_max = fn_dt.max(), fp_dt.max()
    if fn_max == 0 and fp_max == 0:
        for region, lbl in ((fn, 1), (fp, 0)):
            if region.sum() > 0:
                ys, xs = np.where(region > 0)
                i = rng.randint(len(ys))
                return int(xs[i]), int(ys[i]), lbl
        return None
    if fn_max >= fp_max:
        idx = np.unravel_index(np.argmax(fn_dt), fn_dt.shape)
        return int(idx[1]), int(idx[0]), 1
    idx = np.unravel_index(np.argmax(fp_dt), fp_dt.shape)
    return int(idx[1]), int(idx[0]), 0


def select_random_error_click(curr_mask_bin, gt_bin, history_xy, rng=np.random):
    """Control arm: a uniformly random pixel of the current error (FN or FP), labelled by GT.
    Same information budget as GIIR; only the placement rule differs."""
    err = np.logical_xor(curr_mask_bin, gt_bin).astype(np.uint8)
    if err.sum() == 0:
        return None
    ys, xs = np.where(err > 0)
    i = rng.randint(len(ys))
    x, y = int(xs[i]), int(ys[i])
    return x, y, int(gt_bin[y, x])


CLICK_POLICIES = {'giir': select_giir_click, 'random_error': select_random_error_click}


def click_radius_px():
    return max(1, int(round(cfg.CLICK_RADIUS_AT_512 * cfg.IMG_SIZE / 512)))


def grid_to_input(xy, grid_hw, S):
    gh, gw = grid_hw
    x, y = xy
    xi = int(round((x + 0.5) * S / gw - 0.5))
    yi = int(round((y + 0.5) * S / gh - 0.5))
    return min(max(xi, 0), S - 1), min(max(yi, 0), S - 1)


def render_disks(clicks_grid, grid_hw, S, device):
    m = np.zeros((S, S), dtype=np.float32)
    R = click_radius_px()
    for xy in clicks_grid:
        xi, yi = grid_to_input(xy, grid_hw, S)
        cv2.circle(m, (xi, yi), R, 1.0, thickness=-1)
    return torch.from_numpy(m)[None].to(device)                          # [1,S,S]


# ============================================================================
# FORWARD
# ============================================================================
def set_spatial_task(model):
    ts = model.model.task_switch
    ts['spatial'] = True
    for k in ('grounding', 'visual', 'audio'):
        if k in ts:
            ts[k] = False


def normalize_input(model, img):
    """[FIX 1] SEEM's meta-architecture does (x - pixel_mean) / pixel_std inside forward();
    calling backbone() directly skips it."""
    mm = model.model
    if not (hasattr(mm, 'pixel_mean') and hasattr(mm, 'pixel_std')):
        raise RuntimeError("model.model has no pixel_mean/pixel_std buffers; set NORMALIZE_INPUT=False "
                           "only after checking how your SEEM build normalises input.")
    return (img - mm.pixel_mean.view(1, -1, 1, 1)) / mm.pixel_std.view(1, -1, 1, 1)


def encode(model, hq, img):
    """Runs ONCE per image (the encoder never sees a click)."""
    x = normalize_input(model, img) if cfg.NORMALIZE_INPUT else img
    features = model.model.backbone(x)
    mask_features, _, ms = model.model.sem_seg_head.pixel_decoder.forward_features(features)
    feats = {'ms': ms, 'mask': mask_features}
    if hq is not None:
        feats['G'] = hq.basis(img, features['res2'] if hq.use_res2 else None, mask_features)
    return feats


def to_grid(logits, grid_hw):
    if tuple(logits.shape[-2:]) == tuple(grid_hw):
        return logits
    return F.interpolate(logits, size=tuple(grid_hw), mode='bilinear', align_corners=False)


def match_selected(cand, sel):
    """Index of the candidate the model selected (prev_mask is a copy of one candidate)."""
    d = (cand - sel).abs().flatten(2).mean(-1)                           # [B,Q]
    dmin, idx = d.min(dim=1)
    return idx, float(dmin.max())


class ReadoutTap:
    """Makes the read-out vectors w_q (mask_embed of the spatial queries) available
    without editing SEEM.
      1) wraps AttentionDataStruct.update_spatial_results so that 'pred_smaskembs'
         (and 'pred_smasks'), which organize_output() has already computed, are
         passed through to the predictor's return dict;
      2) fallback: a forward hook on predictor.mask_embed, sliced with
         attention_data.query_index['queries_spatial'].
    Either way preflight() checks einsum(w, F) == pred_smasks."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.last = None
        self.passthrough = self._install_passthrough(predictor)
        me = getattr(predictor, 'mask_embed', None)
        self.hook = me.register_forward_hook(self._hook) if isinstance(me, nn.Module) else None

    @staticmethod
    def _install_passthrough(predictor):
        ad = getattr(predictor, 'attention_data', None)
        if ad is None or not hasattr(ad, 'update_spatial_results'):
            return False
        cls = type(ad)
        if getattr(cls, '_readout_passthrough', False):
            return True
        orig = cls.update_spatial_results

        def update_spatial_results(self, results, *args, **kwargs):
            extra = orig(self, results, *args, **kwargs)
            if isinstance(extra, dict) and isinstance(results, dict):
                for key in ('pred_smasks', 'pred_smaskembs'):
                    if key not in extra and key in results:
                        extra[key] = results[key]
            return extra

        cls.update_spatial_results = update_spatial_results
        cls._readout_passthrough = True
        return True

    def _hook(self, module, inputs, output):
        self.last = output                      # the final layer's call is the last one per forward

    def from_hook(self, n_cand):
        out = self.last
        if out is None or out.dim() != 3:
            return None
        ad = getattr(self.predictor, 'attention_data', None)
        qi = getattr(ad, 'query_index', {}).get('queries_spatial') if ad is not None else None
        w = out[:, qi[0]:qi[1]] if qi is not None else out
        return w if w.shape[1] == n_cand else None

    def remove(self):
        if self.hook is not None:
            self.hook.remove()


READOUT_TAP = [None]


def run_round(model, hq, cadapter, feats, pos_clicks, neg_clicks, grid_hw, prev_mask, need_all):
    S = cfg.IMG_SIZE
    dev = feats['mask'].device
    pos = render_disks(pos_clicks, grid_hw, S, dev)
    neg = render_disks(neg_clicks, grid_hw, S, dev)
    mf = feats['mask']
    if cadapter is not None:        # [FIX 5] stride-8 level; ms[0] is stride 32 (16x16 at 512)
        mf = cadapter(mf, feats['ms'][-1], pos[:, None], neg[:, None])
    extra = {'spatial_query_pos_mask': [pos.bool()], 'spatial_query_neg_mask': [neg.bool()]}
    if prev_mask is not None:
        extra['prev_mask'] = prev_mask
    preds = model.model.sem_seg_head.predictor(feats['ms'], mf, target_queries=None,
                                               extra=extra, task='spatial')
    raw = preds['prev_mask']                                              # model's own selection
    cand = preds.get('pred_smasks', None)
    out = {'raw': raw, 'cand': cand, 'mf': mf}
    if hq is None:
        out['sel'], out['all'] = raw, cand
        return out
    if cand is None:
        raise KeyError("USE_HQ needs 'pred_smasks' from the predictor (your seem_v0 patch returned it "
                       "for oracle training before). preds keys: " + str(sorted(preds.keys())))
    w, src = preds.get('pred_smaskembs', None), 'pred_smaskembs'
    if w is None and READOUT_TAP[0] is not None:
        w, src = READOUT_TAP[0].from_hook(cand.shape[1]), 'mask_embed hook'
    if w is None:
        raise KeyError("USE_HQ needs the read-out vectors w_q. Neither the pass-through nor the "
                       "mask_embed hook found them. preds keys: " + str(sorted(preds.keys())) +
                       ". Add  \"pred_smaskembs\": v_emb  to the dict returned by "
                       "update_spatial_results() in attention_data_struct_seemv0.py.")
    idx, _ = match_selected(cand, raw)
    w_sel = w[torch.arange(w.shape[0], device=w.device), idx][:, None]   # [B,1,C]
    out['sel'] = hq.readout(raw, w_sel, feats['G'])
    out['all'] = hq.readout(cand, w, feats['G']) if need_all else None
    out['w'], out['w_src'] = w, src
    return out


def memory_from(sup_logits, raw_ref):
    if tuple(sup_logits.shape[-2:]) == tuple(raw_ref.shape[-2:]):
        return sup_logits
    if cfg.MEMORY_FROM_HQ:
        return F.interpolate(sup_logits, size=raw_ref.shape[-2:], mode='area')
    return raw_ref


# ============================================================================
# LOSSES (unchanged: clDice stays OUT of Paper 1's objective — ownership contract)
# ============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1); targets = targets.view(-1)
        inter = (probs * targets).sum()
        return 1 - (2. * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * bce * (1 - p_t) ** self.gamma).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__(); self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1); targets = targets.view(-1)
        tp = (probs * targets).sum(); fn = ((1 - probs) * targets).sum(); fp = (probs * (1 - targets)).sum()
        return 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(); self.focal = FocalLoss(); self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
    def forward(self, logits, targets):
        return 0.4 * self.dice(logits, targets) + 0.3 * self.focal(logits, targets) + 0.3 * self.tversky(logits, targets)


def dice_loss(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


# ============================================================================
# METRICS
# ============================================================================
def calc_metrics(pred_bin, gt_bin):
    TP = np.logical_and(pred_bin, gt_bin).sum(); TN = np.logical_and(~pred_bin, ~gt_bin).sum()
    FP = np.logical_and(pred_bin, ~gt_bin).sum(); FN = np.logical_and(~pred_bin, gt_bin).sum()
    s = 1e-6
    return ((2.0 * TP) / (pred_bin.sum() + gt_bin.sum() + s),
            TP / (TP + FP + FN + s), TP / (TP + FN + s), TN / (TN + FP + s))


def cl_dice(pred_bin, gt_bin):
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0
    sp = skeletonize(pred_bin); sg = skeletonize(gt_bin)
    tprec = np.logical_and(sp, gt_bin).sum() / (sp.sum() + 1e-12)
    tsens = np.logical_and(sg, pred_bin).sum() / (sg.sum() + 1e-12)
    return float(2 * tprec * tsens / (tprec + tsens + 1e-12))


def noc(curve, tau, K):
    hits = np.where(np.asarray(curve) >= tau)[0]
    return int(hits[0]) + 1 if len(hits) else K


def paired_summary(a, b, label):
    """Paired change b - a over images: mean, 95% CI (t), one-sided Wilcoxon, images improved."""
    d = np.asarray(b, float) - np.asarray(a, float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    try:
        from scipy import stats
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float('nan')
        p = float(stats.wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0
    except Exception:
        half, p = 1.96 * sd / math.sqrt(max(n, 1)), float('nan')
    dz = mean / sd if sd > 0 else float('nan')
    print(f"  {label}: {mean:+.5f}  95% CI [{mean - half:+.5f}, {mean + half:+.5f}]  "
          f"improved {int((d > 0).sum())}/{n}  Wilcoxon(one-sided) p={p:.2e}  dz={dz:.2f}")
    return dict(mean=mean, ci=(mean - half, mean + half), p=p, dz=dz, improved=int((d > 0).sum()), n=n)


# ============================================================================
# TRAINING
# ============================================================================
def pick_best_candidate(cand_logits, gt_full):
    """TRAINING ONLY: which candidate to supervise (never used at evaluation)."""
    h, w = cand_logits.shape[-2:]
    gt_s = F.interpolate(gt_full, size=(h, w), mode='nearest')[0, 0].bool()
    pb = torch.sigmoid(cand_logits[0]) > 0.5
    inter = (pb & gt_s).sum(dim=(1, 2)).float()
    denom = pb.sum(dim=(1, 2)).float() + gt_s.sum().float() + 1e-6
    return int((2.0 * inter / denom).argmax().item())


def set_train_mode(model, hq, cadapter):
    model.train()
    if cfg.FROZEN_EVAL_MODE:                     # no stochastic depth in frozen modules
        model.model.backbone.eval()
        model.model.sem_seg_head.pixel_decoder.eval()
    for m in (hq, cadapter):
        if m is not None:
            m.train()


def set_eval_mode(model, hq, cadapter):
    model.eval()
    for m in (hq, cadapter):
        if m is not None:
            m.eval()


class EMA:
    def __init__(self, params, decay):
        self.params = list(params)
        self.shadow = [p.detach().clone() for p in self.params]
        self.decay = decay
        self.steps = 0

    @torch.no_grad()
    def update(self):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1 - d)

    @contextmanager
    def swapped(self):
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p, s in zip(self.params, self.shadow):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)


@contextmanager
def maybe_ema(ema):
    if ema is None:
        yield
    else:
        with ema.swapped():
            yield


GRAD_AUDIT_DONE = [False]


def gradient_audit(model, hq, cadapter):
    """Counts what ACTUALLY receives gradient (Version B reported 43.6 M this way)."""
    got = sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    for m in (hq, cadapter):
        if m is not None:
            got += sum(p.numel() for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for m in (hq, cadapter) if m is not None for p in m.parameters())
    print(f"  gradient audit (first step): {got:,} params received non-zero gradient "
          f"({100.0 * got / total:.2f}% of {total:,}). Zero-init branches (LoRA A, HQ G) get "
          f"gradient only after their partner (B, U) moves.")


def train_one_epoch(model, hq, cadapter, loader, optimizer, scheduler, criterion, ema, epoch):
    set_train_mode(model, hq, cadapter)
    tot = {'L': 0.0, 'mask': 0.0, 'impr': 0.0}
    n = 0
    pbar = tqdm(loader, desc=f"Train {cfg.RUN_TAG} ep{epoch} (K={cfg.K_ROUNDS})")
    for img, gt, sid in pbar:
        img = img.to(cfg.DEVICE)
        gt = gt.to(cfg.DEVICE)[:, None]                                  # [1,1,gh,gw]
        grid_hw = tuple(gt.shape[-2:])
        gt_bin = gt[0, 0].cpu().numpy().astype(bool)
        set_spatial_task(model)
        feats = encode(model, hq, img)

        pos, neg = [seed_first_click(gt_bin)], []
        prev = None
        optimizer.zero_grad(set_to_none=True)
        mask_losses, dice_terms = [], []
        for _ in range(cfg.K_ROUNDS):
            r = run_round(model, hq, cadapter, feats, pos, neg, grid_hw, prev,
                          need_all=cfg.ORACLE_BEST_CANDIDATE)
            if cfg.ORACLE_BEST_CANDIDATE and r['all'] is not None:
                bi = pick_best_candidate(r['all'], gt)
                sup = r['all'][:, bi:bi + 1]
            else:
                sup = r['sel']
            sup_g = to_grid(sup, grid_hw)
            mask_losses.append(criterion(sup_g, gt))
            dice_terms.append(dice_loss(sup_g, gt))
            prev = memory_from(sup, r['raw']).detach()
            with torch.no_grad():
                pred_bin = (torch.sigmoid(sup_g) > 0.5)[0, 0].cpu().numpy()
                nxt = select_giir_click(pred_bin, gt_bin, pos + neg)
                if nxt is not None:
                    (pos if nxt[2] == 1 else neg).append((nxt[0], nxt[1]))

        L_mask = torch.stack(mask_losses).mean()
        if len(dice_terms) > 1:
            L_impr = torch.stack([F.relu(dice_terms[k] - dice_terms[k - 1].detach())
                                  for k in range(1, len(dice_terms))]).mean()
        else:
            L_impr = torch.zeros((), device=cfg.DEVICE)
        loss = L_mask + cfg.IMPROVE_W * L_impr
        loss.backward()
        if not GRAD_AUDIT_DONE[0]:
            gradient_audit(model, hq, cadapter)
            GRAD_AUDIT_DONE[0] = True
        params = [p for g in optimizer.param_groups for p in g['params']]
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        if isinstance(scheduler, LambdaLR):
            scheduler.step()
        if ema is not None:
            ema.update()
        tot['L'] += loss.item(); tot['mask'] += L_mask.item(); tot['impr'] += float(L_impr.item())
        n += 1
        pbar.set_postfix({k: f"{v / n:.4f}" for k, v in tot.items()})
    return {k: v / max(n, 1) for k, v in tot.items()}


# ============================================================================
# EVALUATION — deployable: model's own selection, fixed threshold per run,
# GT only places clicks. Seeds are fixed per image so SEEM's rand_sample
# point subsampling no longer makes repeated evaluations disagree.
# ============================================================================
@torch.no_grad()
def evaluate(model, hq, cadapter, loader, thresholds, K, policy='giir', k_report=None,
             compute_cldice=False, desc="Eval", return_preds=False):
    set_eval_mode(model, hq, cadapter)
    k_report = k_report or min(cfg.K_ROUNDS, K)
    choose = CLICK_POLICIES[policy]
    res = {t: {'curve': [], 'iou': [], 'se': [], 'sp': [], 'cldice': [], 'pred': [], 'gt': []} for t in thresholds}
    sids = []
    for i, (img, gt, sid) in enumerate(tqdm(loader, desc=desc, leave=False)):
        img = img.to(cfg.DEVICE)
        gt = gt.to(cfg.DEVICE)[:, None]
        grid_hw = tuple(gt.shape[-2:])
        gt_bin = gt[0, 0].cpu().numpy().astype(bool)
        sids.append(sid[0] if isinstance(sid, (list, tuple)) else sid)
        set_spatial_task(model)
        feats = encode(model, hq, img)                                   # once per image
        for t in thresholds:
            torch.manual_seed(cfg.SEED + i)
            rng = np.random.RandomState(cfg.SEED + i)
            pos, neg = [seed_first_click(gt_bin)], []
            prev = None
            curve = []
            for k in range(K):
                r = run_round(model, hq, cadapter, feats, pos, neg, grid_hw, prev, need_all=False)
                prev = memory_from(r['sel'], r['raw'])
                prob = torch.sigmoid(to_grid(r['sel'], grid_hw))
                pred_bin = (prob > t)[0, 0].cpu().numpy().astype(bool)
                curve.append(calc_metrics(pred_bin, gt_bin)[0])
                if k + 1 == k_report:
                    _, iou, se, sp = calc_metrics(pred_bin, gt_bin)
                    res[t]['iou'].append(iou); res[t]['se'].append(se); res[t]['sp'].append(sp)
                    if compute_cldice:
                        res[t]['cldice'].append(cl_dice(pred_bin, gt_bin))
                    if return_preds:
                        res[t]['pred'].append(pred_bin.copy()); res[t]['gt'].append(gt_bin)
                nxt = choose(pred_bin, gt_bin, pos + neg, rng)
                if nxt is not None:
                    (pos if nxt[2] == 1 else neg).append((nxt[0], nxt[1]))
            res[t]['curve'].append(curve)
    out = {}
    for t in thresholds:
        c = np.asarray(res[t]['curve'], float)                           # [N,K]
        out[t] = {'curves': c, 'dice': float(c[:, k_report - 1].mean()),
                  'dice_curve': c.mean(0).tolist(),
                  'iou': float(np.mean(res[t]['iou'])), 'se': float(np.mean(res[t]['se'])),
                  'sp': float(np.mean(res[t]['sp'])),
                  'cldice': float(np.mean(res[t]['cldice'])) if compute_cldice else None,
                  'cldice_per_image': np.asarray(res[t]['cldice'], float) if compute_cldice else None,
                  'iou_per_image': np.asarray(res[t]['iou'], float),
                  'preds': res[t]['pred'] if return_preds else None,
                  'gts': res[t]['gt'] if return_preds else None}
    out['sids'] = sids
    return out


def reachable_ceiling(model, hq, loader, steps=300, lr=0.05):
    """DIAGNOSTIC — uses GT by design, never a deployable number.
    Best Dice ANY prompt could reach = the best read-out vector w on the basis,
    fitted per image (no bias, exactly like the read-out). Compares the trained
    model's own stride-4 basis F with its lifted basis Up(F) + U^T G. Channels are
    rescaled by their RMS for the fit (a diagonal rescaling leaves the span unchanged)."""
    set_eval_mode(model, hq, None)
    out = {'stock': [], 'lifted': []}
    for img, gt, sid in tqdm(loader, desc="Reachable-set ceiling", leave=False):
        img = img.to(cfg.DEVICE); gt = gt.to(cfg.DEVICE)[:, None]
        grid = tuple(gt.shape[-2:]); gt_bin = gt[0, 0].cpu().numpy().astype(bool)
        with torch.no_grad():
            feats = encode(model, hq, img)
            bases = {'stock': feats['mask']}
            if hq is not None:
                up = F.interpolate(feats['mask'], size=feats['G'].shape[-2:], mode='bilinear', align_corners=False)
                bases['lifted'] = up + torch.einsum('rc,brhw->bchw', hq.U.weight, feats['G'])
        for name, B in bases.items():
            Bn = (B / (B.pow(2).mean((2, 3), keepdim=True).sqrt() + 1e-6)).detach()
            w = torch.zeros(1, Bn.shape[1], device=Bn.device, requires_grad=True)
            opt = torch.optim.Adam([w], lr=lr)
            with torch.enable_grad():
                for _ in range(steps):
                    lg = to_grid(torch.einsum('bc,bchw->bhw', w, Bn)[:, None], grid)
                    loss = F.binary_cross_entropy_with_logits(lg, gt) + dice_loss(lg, gt)
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                lg = to_grid(torch.einsum('bc,bchw->bhw', w, Bn)[:, None], grid)
                pred = (torch.sigmoid(lg) > 0.5)[0, 0].cpu().numpy()
            out[name].append(calc_metrics(pred, gt_bin)[0])
    return {k: np.asarray(v, float) for k, v in out.items() if v}


def choose_threshold(val_out, thresholds):
    """[FIX 6] one global threshold from the validation set; ties -> closest to 0.5."""
    scores = {t: val_out[t]['dice'] for t in thresholds}
    best = max(scores.values())
    t_star = min([t for t, s in scores.items() if s >= best - 1e-9], key=lambda t: abs(t - 0.5))
    return t_star, scores


# ============================================================================
# PARAMETERS / OPTIMISER / CHECKPOINTS
# ============================================================================
def param_table(model, hq, cadapter):
    enc = sum(p.numel() for n, p in model.model.named_parameters() if p.requires_grad and n.startswith('backbone.'))
    dec = sum(p.numel() for n, p in model.model.named_parameters()
              if p.requires_grad and n.startswith('sem_seg_head.predictor'))
    other = sum(p.numel() for n, p in model.model.named_parameters()
                if p.requires_grad and not n.startswith('backbone.') and not n.startswith('sem_seg_head.predictor'))
    n_hq = sum(p.numel() for p in hq.parameters()) if hq is not None else 0
    n_ca = sum(p.numel() for p in cadapter.parameters()) if cadapter is not None else 0
    total = sum(p.numel() for p in model.parameters()) + n_hq + n_ca
    trainable = enc + dec + other + n_hq + n_ca
    print(f"  encoder adaptation ({cfg.ENC_ADAPT}): {enc:,}")
    print(f"  read-out adapter ({cfg.READOUT if cfg.USE_HQ else 'none'}):{' ' * max(1, 15 - len(cfg.READOUT if cfg.USE_HQ else 'none'))}{n_hq:,}")
    print(f"  click adapter ({cfg.CLICK_ADAPTER}): {n_ca:,}")
    print(f"  decoder (requires_grad):           {dec:,}   [DECODER_TRAIN = '{cfg.DECODER_TRAIN}']")
    if other:
        print(f"  other trainable:                   {other:,}")
    print(f"  trainable {trainable:,} of {total:,} ({100.0 * trainable / total:.2f}%)")
    return dict(encoder=enc, hq=n_hq, click_adapter=n_ca, decoder=dec, trainable=trainable, total=total)


def param_groups(model, hq, cadapter):
    dec, enc, new = [], [], []
    for n, p in model.model.named_parameters():
        if not p.requires_grad:
            continue
        if n in NEW_PARAM_NAMES:                 # not in the SEEM checkpoint: fresh module
            new.append(p)
        elif n.startswith('sem_seg_head.predictor'):
            dec.append(p)
        elif n.startswith('backbone.'):
            enc.append(p)
        else:
            new.append(p)
    for m in (hq, cadapter):
        if m is not None:
            new += [p for p in m.parameters() if p.requires_grad]
    groups = [dict(params=dec, lr=cfg.LR_DECODER, name='decoder'),
              dict(params=enc, lr=cfg.LR_ENCODER, name='encoder'),
              dict(params=new, lr=cfg.LR_NEW, name='new')]
    return [g for g in groups if g['params']]


def make_scheduler(optimizer, steps_per_epoch):
    if cfg.SCHEDULER == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=7)
    total = max(1, cfg.EPOCHS * steps_per_epoch)
    warm = max(1, cfg.WARMUP_EPOCHS * steps_per_epoch)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))
    return LambdaLR(optimizer, lr_lambda)


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.model.named_parameters() if p.requires_grad}


def save_checkpoint(path, model, hq, cadapter, ema, meta):
    with maybe_ema(ema):
        ck = {'trainable': trainable_state(model),
              'hq': hq.state_dict() if hq is not None else None,
              'cadapter': cadapter.state_dict() if cadapter is not None else None,
              'meta': meta}
    torch.save(ck, path)


def load_checkpoint(path, model, hq, cadapter):
    ck = safe_torch_load(path, map_location='cpu') if isinstance(path, str) else path
    res = model.model.load_state_dict(ck['trainable'], strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"checkpoint has keys the model does not: {res.unexpected_keys[:5]}. "
                           "It was trained with different SEEM code (e.g. your own seem_v0.py with the learned "
                           "positional prompt) — load it with that same SEEM code.")
    names = set(n for n, _ in model.model.named_parameters())
    lost = [k for k in ck['trainable'] if k not in names]
    if lost:
        raise RuntimeError(f"trainable keys not restored: {lost[:5]}")
    if hq is not None:
        hq.load_state_dict(ck['hq'])
    if cadapter is not None:
        cadapter.load_state_dict(ck['cadapter'])
    return ck['meta']


# ============================================================================
# PREFLIGHT
# ============================================================================
@torch.no_grad()
def preflight(model, hq, cadapter, loader):
    print("\nPREFLIGHT")
    set_eval_mode(model, hq, cadapter)
    img, gt, sid = next(iter(loader))
    img = img.to(cfg.DEVICE); gt = gt.to(cfg.DEVICE)[:, None]
    grid_hw = tuple(gt.shape[-2:]); gt_bin = gt[0, 0].cpu().numpy().astype(bool)
    set_spatial_task(model)
    x = normalize_input(model, img) if cfg.NORMALIZE_INPUT else img
    print(f"  input range fed to backbone: [{float(x.min()):.2f}, {float(x.max()):.2f}] "
          f"({'normalised' if cfg.NORMALIZE_INPUT else 'RAW 0-255'})")
    feats = encode(model, hq, img)
    print("  multi-scale levels: " + ", ".join(f"ms[{i}] {tuple(m.shape[-2:])}" for i, m in enumerate(feats['ms']))
          + f" | mask_features {tuple(feats['mask'].shape[-2:])}  (ms[0] is the COARSEST level)")
    pos = [seed_first_click(gt_bin)]
    r1 = run_round(model, hq, cadapter, feats, pos, [], grid_hw, None, need_all=True)
    if r1['cand'] is None:
        raise RuntimeError("predictor returned no 'pred_smasks' — ORACLE_BEST_CANDIDATE/HQ need your seem_v0 patch.")
    idx, dmin = match_selected(r1['cand'], r1['raw'])
    print(f"  selected candidate {int(idx[0])} of {r1['cand'].shape[1]} (match residual {dmin:.2e})")
    if dmin > 1e-4:
        print("  WARNING: prev_mask is not one of pred_smasks (your patch changes the selection); "
              "the HQ read-out uses the nearest candidate's w_q.")
    if hq is not None:
        w = r1['w']
        print(f"  read-out vectors w_q from: {r1['w_src']}  {tuple(w.shape)}")
        rec = torch.einsum('bqc,bchw->bqhw', w, r1['mf'])
        rel = float((rec - r1['cand']).abs().max() / (r1['cand'].abs().max() + 1e-8))
        print(f"  einsum(w_q, F) vs pred_smasks: max rel err {rel:.2e}")
        if rel > 1e-2:
            raise RuntimeError(f"w_q from '{r1['w_src']}' are not the read-out vectors of pred_smasks — "
                               "the HQ read-out would be wrong. Add  \"pred_smaskembs\": v_emb  to the dict "
                               "returned by update_spatial_results() in attention_data_struct_seemv0.py.")
        up = F.interpolate(r1['cand'], size=r1['all'].shape[-2:], mode='bilinear', align_corners=False)
        ident = float((r1['all'] - up).abs().max())
        print(f"  HQ identity at init: max|HQ - Up(stock)| = {ident:.2e} "
              f"({'PASS' if ident < 1e-5 or hq.U.weight.abs().sum() > 0 else 'FAIL'})")
    d1 = calc_metrics((torch.sigmoid(to_grid(r1['sel'], grid_hw)) > 0.5)[0, 0].cpu().numpy().astype(bool), gt_bin)[0]
    print(f"  untrained 1-click Dice on '{sid[0]}': {d1:.4f}")


# ============================================================================
# MAIN
# ============================================================================
def cfg_snapshot():
    """Plain-typed Config values, stored in every checkpoint."""
    cls = type(cfg)
    out = {}
    for k in dir(cfg):
        if k.isupper() and not isinstance(getattr(cls, k, None), property):
            v = getattr(cfg, k)
            if v is None or isinstance(v, (bool, int, float, str, tuple, list)):
                out[k] = v
    return out


CFG_MUST_MATCH = ('ENC_ADAPT', 'LORA_RANK', 'LORA_ALPHA', 'USE_HQ', 'HQ_STRIDE', 'HQ_RANK', 'HQ_USE_STEM', 'HQ_USE_RES2',
                  'HQ_USE_MASKFEAT', 'CLICK_ADAPTER', 'IMG_SIZE', 'NORMALIZE_INPUT', 'TASK', 'SUBSET',
                  'READOUT', 'RIDA_RIDGE', 'RIDA_ENC_UPDATE', 'RIDA_TAP_STAGES', 'RIDA_SCALES_ENC',
                  'RIDA_PIX_RANK', 'RIDA_SCALES_PIX', 'RIDA_F_RANK', 'RIDA_SCALES_F',
                  'AF_DIM', 'AF_SCALE', 'DECODER_TRAIN', 'USE_LEARNED_POS_PROMPT')


def _same_setting(a, b):
    """Config values as saved vs current (a tuple saved through JSON comes back as a list)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_setting(x, y) for x, y in zip(a, b))
    return a == b


def set_decoder_trainable(model):
    """[FROZEN DECODER] Which parameters of SEEM's decoder (sem_seg_head.predictor) train.
    cfg.DECODER_TRAIN:
      'full'     every predictor parameter except the language encoder (all earlier rows);
      'new_only' only predictor parameters absent from the SEEM checkpoint, i.e. the learned
                 positional click prompt; SEEM's own weights stay frozen;
      'none'     no predictor parameter; only the adapter trains.
    Returns the names of the predictor parameters set trainable."""
    mode = cfg.DECODER_TRAIN
    if mode not in ('full', 'new_only', 'none'):
        raise ValueError(f"DECODER_TRAIN must be 'full', 'new_only' or 'none', not {mode!r}")
    if mode == 'none' and cfg.USE_LEARNED_POS_PROMPT:
        raise ValueError("DECODER_TRAIN='none' would freeze the learned positional prompt at its random start. "
                         "Set USE_LEARNED_POS_PROMPT = False (SEEM's own click encoding) or use 'new_only'.")
    names = []
    for name, p in model.model.named_parameters():
        if not name.startswith('sem_seg_head.predictor'):
            continue
        if 'lang_encoder' in name and not cfg.TRAIN_LANG_ENCODER:
            continue
        if mode == 'full' or (mode == 'new_only' and name in NEW_PARAM_NAMES):
            p.requires_grad = True
            names.append(name)
    if mode != 'full':
        chosen = set(names)
        n_train = sum(p.numel() for name, p in model.model.named_parameters() if name in chosen)
        n_frozen = sum(p.numel() for name, p in model.model.named_parameters()
                       if name.startswith('sem_seg_head.predictor') and name not in chosen)
        parts = sorted({nm.split('.')[2] for nm in names})
        print(f"  [frozen decoder] mode '{mode}': {n_train:,} decoder params train "
              f"({len(names)} tensors: {', '.join(parts) if parts else 'none'}); {n_frozen:,} stay frozen")
        if mode == 'new_only' and not names:
            print("  [frozen decoder] note: no decoder parameter is new to SEEM, so only the adapter trains")
    return names


def build_all():
    """SEEM + encoder adapters + trainable decoder + HQ basis + click adapter + read-out tap."""
    model = build_seem()
    for p in model.parameters():
        p.requires_grad = False
    counts = inject_encoder_adapters(model, cfg.ENC_ADAPT, cfg.LORA_RANK, cfg.LORA_ALPHA, cfg.LORA_DROPOUT)
    model.to(cfg.DEVICE)
    print(f"encoder adapters ({cfg.ENC_ADAPT}): {counts}")
    set_decoder_trainable(model)                                         # [FROZEN DECODER] cfg.DECODER_TRAIN

    predictor = model.model.sem_seg_head.predictor
    if hasattr(predictor, 'use_learned_pos_prompt'):
        predictor.use_learned_pos_prompt = cfg.USE_LEARNED_POS_PROMPT
        predictor.pos_prompt_on_tokens = cfg.POS_PROMPT_ON_TOKENS
        predictor.pos_prompt_on_meanq = cfg.POS_PROMPT_ON_MEANQ

    with torch.no_grad():                                                # channel sizes from the real model
        probe = torch.zeros(1, 3, 64, 64, device=cfg.DEVICE)
        fprobe = model.model.backbone(probe)
        c_res2 = fprobe['res2'].shape[1]
        mprobe, _, msprobe = model.model.sem_seg_head.pixel_decoder.forward_features(fprobe)
        c_mask, c_hr = mprobe.shape[1], msprobe[-1].shape[1]
        del fprobe, mprobe, msprobe

    if cfg.READOUT not in ('hq', 'rida'):
        raise ValueError(f"READOUT must be 'hq' or 'rida', not {cfg.READOUT!r}")
    hq = None
    if cfg.USE_HQ and cfg.READOUT == 'rida':
        taps = getattr(model, '_rida_taps', [])
        if cfg.ENC_ADAPT != 'rida':
            print("  note: READOUT='rida' without ENC_ADAPT='rida' -> the read-out uses the pixel and F codes only")
        hq = RidaReadout(c_mask, taps, rank=cfg.HQ_RANK, stride=cfg.HQ_STRIDE,
                         pix_rank=cfg.RIDA_PIX_RANK, f_rank=cfg.RIDA_F_RANK,
                         scales_pix=cfg.RIDA_SCALES_PIX, scales_f=cfg.RIDA_SCALES_F,
                         ridge=cfg.RIDA_RIDGE).to(cfg.DEVICE)
        print(f"RiDA read-out: {hq.c_in} code maps -> G rank {cfg.HQ_RANK} at stride {cfg.HQ_STRIDE} "
              f"({len(taps)} encoder sites tapped, ridge {'on' if cfg.RIDA_RIDGE else 'off'})")
    elif cfg.USE_HQ:
        hq = HQBasis(c_mask=c_mask, c_res2=c_res2, rank=cfg.HQ_RANK, stride=cfg.HQ_STRIDE,
                     use_stem=cfg.HQ_USE_STEM, use_res2=cfg.HQ_USE_RES2,
                     use_maskfeat=cfg.HQ_USE_MASKFEAT).to(cfg.DEVICE)
    cadapter = None
    if cfg.CLICK_ADAPTER == 'prototype':
        cadapter = PrototypeGuidedAdapter(c_mask, c_hr, 256).to(cfg.DEVICE)
    elif cfg.CLICK_ADAPTER == 'standard':
        cadapter = EncoderToPromptAdapter(c_mask, 256).to(cfg.DEVICE)
    elif cfg.CLICK_ADAPTER != 'none':
        raise ValueError(cfg.CLICK_ADAPTER)

    # always: the original SEEM returns only prev_mask; every row needs pred_smasks (oracle-candidate
    # training) and the HQ rows also need pred_smaskembs
    if READOUT_TAP[0] is not None:
        READOUT_TAP[0].remove()
    READOUT_TAP[0] = ReadoutTap(predictor)
    print(f"read-out tap: pass-through {'installed' if READOUT_TAP[0].passthrough else 'NOT available'}, "
          f"mask_embed hook {'installed' if READOUT_TAP[0].hook is not None else 'NOT available'}")
    return model, hq, cadapter


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    set_seed(cfg.SEED)
    GRAD_AUDIT_DONE[0] = False
    print("=" * 72)
    print(f" PAPER 1 | row {cfg.RUN_TAG} | task {cfg.TASK} | {cfg.SUBSET} | input {cfg.IMG_SIZE}")
    print(f" norm={cfg.NORMALIZE_INPUT} aug={cfg.AUGMENT} native={cfg.NATIVE_RES} enc={cfg.ENC_ADAPT} "
          f"HQ={cfg.USE_HQ}(s{cfg.HQ_STRIDE}, r{cfg.HQ_RANK}) read-out={cfg.READOUT} click_adapter={cfg.CLICK_ADAPTER}")
    if cfg.ENC_ADAPT == 'rida':
        print(f" RiDA: rank {cfg.LORA_RANK}, ridge {cfg.RIDA_RIDGE}, encoder path {cfg.RIDA_ENC_UPDATE}, "
              f"taps from stages {tuple(cfg.RIDA_TAP_STAGES)}, scales enc {tuple(cfg.RIDA_SCALES_ENC)} "
              f"pix {tuple(cfg.RIDA_SCALES_PIX)} F {tuple(cfg.RIDA_SCALES_F)}")
    print("=" * 72)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    if not os.path.isdir(cfg.GT_MASK_DIR):
        raise FileNotFoundError(f"TASK='{cfg.TASK}' -> {cfg.GT_MASK_DIR} does not exist.")

    train_ids, val_ids, test_ids = make_splits()
    train_loader = DataLoader(OCTADataset(train_ids, augment=cfg.AUGMENT), batch_size=cfg.BATCH_SIZE,
                              shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(OCTADataset(val_ids), batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(OCTADataset(test_ids), batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)
    print(f"split: train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")

    model, hq, cadapter = build_all()

    print("\nPARAMETERS")
    ptab = param_table(model, hq, cadapter)
    preflight(model, hq, cadapter, val_loader)

    criterion = CombinedLoss().to(cfg.DEVICE)
    groups = param_groups(model, hq, cadapter)
    optimizer = AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, len(train_loader))
    ema = EMA([p for g in groups for p in g['params']], cfg.EMA_DECAY) if cfg.USE_EMA else None
    print("optimiser groups: " + ", ".join(
        f"{g['name']} {sum(p.numel() for p in g['params']):,} @ {g.get('initial_lr', g['lr']):.0e}"
        for g in optimizer.param_groups))

    thresholds = tuple(sorted(set(cfg.THRESH_GRID) | {0.5}))
    best, best_t = -1.0, 0.5
    tag = f"{cfg.RUN_TAG}_{cfg.TASK}_{cfg.SUBSET}"
    best_path = os.path.join(cfg.CHECKPOINT_DIR, f"best_{tag}.pth")
    last_path = os.path.join(cfg.CHECKPOINT_DIR, f"last_{tag}.pth")
    history = []
    start_epoch = 1
    if cfg.RESUME and os.path.exists(last_path):
        done, best, best_t, history = load_last(last_path, model, hq, cadapter, optimizer, scheduler, ema)
        start_epoch = done + 1
        GRAD_AUDIT_DONE[0] = True
        print(f"RESUMED from {last_path}: {done} epochs done, best val {best:.4f} at t*={best_t:.2f}")
    elif os.path.exists(last_path):
        print(f"RESUME is off: starting {tag} from epoch 1 (the old {last_path} will be overwritten)")
    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        tr = train_one_epoch(model, hq, cadapter, train_loader, optimizer, scheduler, criterion, ema, epoch)
        with maybe_ema(ema):
            v = evaluate(model, hq, cadapter, val_loader, thresholds, cfg.K_ROUNDS, desc="Val")
        t_star, scores = choose_threshold(v, thresholds)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(v[t_star]['dice'])
        curve = " ".join(f"{d:.3f}" for d in v[t_star]['dice_curve'])
        print(f"ep {epoch:3d} | L {tr['L']:.4f} (mask {tr['mask']:.4f}, impr {tr['impr']:.4f}) | "
              f"val Dice@{cfg.K_ROUNDS} t=0.50 {v[0.5]['dice']:.4f} | t*={t_star:.2f} {v[t_star]['dice']:.4f} "
              f"| curve [{curve}]")
        history.append(dict(epoch=epoch, **tr, val_05=v[0.5]['dice'], t_star=t_star, val_tstar=v[t_star]['dice']))
        if v[t_star]['dice'] > best:
            best, best_t = v[t_star]['dice'], t_star
            save_checkpoint(best_path, model, hq, cadapter, ema,
                            meta=dict(epoch=epoch, t_star=t_star, val_scores=scores, row=cfg.RUN_TAG,
                                      cfg=cfg_snapshot()))
            print(f"  saved (val {best:.4f} at t*={t_star:.2f})")
        if cfg.SAVE_LAST:
            save_last(last_path, epoch, model, hq, cadapter, optimizer, scheduler, ema, best, best_t, history)

    # ---------------------------------------------------------------- test
    print(f"\nFINAL TEST — loading {best_path}")
    meta = load_checkpoint(best_path, model, hq, cadapter)
    t_star = meta['t_star']
    K, kr = cfg.K_EVAL, cfg.K_ROUNDS
    te = evaluate(model, hq, cadapter, test_loader, tuple(sorted({0.5, t_star})), K, 'giir',
                  k_report=kr, compute_cldice=True, desc="Test (GIIR)")
    rc = None
    if cfg.RUN_RANDOM_CONTROL:
        rc = evaluate(model, hq, cadapter, test_loader, (t_star,), K, 'random_error', k_report=kr,
                      desc="Test (random control)")

    print(f"\n{'=' * 72}\nFINAL TEST  row {cfg.RUN_TAG}  n={len(test_ids)}  (t* = {t_star:.2f} chosen on validation)")
    for t in sorted({0.5, t_star}):
        print_test_row(te[t], t, K, kr)
    c = te[t_star]['curves']
    g_giir = paired_summary(c[:, 0], c[:, kr - 1], f"gain Dice@{kr} - Dice@1 (GIIR, t*)")
    stats_out = {'giir_gain': g_giir}
    if rc is not None:
        cr = rc[t_star]['curves']
        stats_out['random_gain'] = paired_summary(cr[:, 0], cr[:, kr - 1], f"gain Dice@{kr} - Dice@1 (random)")
        stats_out['giir_vs_random'] = paired_summary(cr[:, kr - 1], c[:, kr - 1], f"GIIR - random at click {kr}")

    ceil = {}
    if cfg.RUN_CEILING:
        ceil = reachable_ceiling(model, hq, test_loader, steps=cfg.CEILING_STEPS)
        print(f"\n  ERROR BUDGET (test, Dice)  deployable@{kr} t*: {te[t_star]['dice']:.4f}"
              f"  |  stride-4 basis ceiling: {ceil['stock'].mean():.4f}"
              + (f"  |  lifted basis ceiling: {ceil['lifted'].mean():.4f}" if 'lifted' in ceil else ""))
        print("  (ceilings = best read-out vector per image, fitted against GT: upper bounds, not results)")

    npz = os.path.join(cfg.CHECKPOINT_DIR, f"results_{tag}.npz")
    np.savez(npz, sids=np.array(te['sids']), t_star=t_star, k_report=kr,
             ceiling_stock=ceil.get('stock', np.zeros(0)), ceiling_lifted=ceil.get('lifted', np.zeros(0)),
             curves_giir_t05=te[0.5]['curves'], curves_giir_tstar=te[t_star]['curves'],
             cldice_t05=te[0.5]['cldice_per_image'], cldice_tstar=te[t_star]['cldice_per_image'],
             curves_random_tstar=rc[t_star]['curves'] if rc is not None else np.zeros(0),
             params=json.dumps(ptab), history=json.dumps(history), stats=json.dumps(stats_out, default=str),
             cfg=json.dumps(cfg_snapshot()))
    print(f"  per-image arrays -> {npz}")
    return te, rc


def print_test_row(o, t, K, kr, label=""):
    c = o['curves']
    print(f"  {label}t={t:.2f}: Dice@1 {c[:, 0].mean():.4f} | Dice@{kr} {o['dice']:.4f} "
          f"(sd {c[:, kr - 1].std(ddof=1):.4f}) | Dice@{K} {c[:, K - 1].mean():.4f} | clDice {o['cldice']:.4f} "
          f"| IoU {o['iou']:.4f} | SE {o['se']:.4f} | SP {o['sp']:.4f}")
    n85 = [noc(ci, 0.85, K) for ci in c]; n90 = [noc(ci, 0.90, K) for ci in c]
    print(f"  {' ' * len(label)}        NoC@85 {np.mean(n85):.2f} (fail {np.mean([x == K and ci[-1] < 0.85 for x, ci in zip(n85, c)]):.0%}) "
          f"| NoC@90 {np.mean(n90):.2f} (fail {np.mean([x == K and ci[-1] < 0.90 for x, ci in zip(n90, c)]):.0%})")


def save_overlay(path, sid, pred_bin, gt_bin, dice):
    """TP white, FN blue (#0072B2), FP orange (#D55E00) over the full projection."""
    img = cv2.imread(os.path.join(cfg.OCTA_FULL_DIR, sid), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != gt_bin.shape:
        img = np.zeros(gt_bin.shape, np.uint8)
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.55
    base[pred_bin & gt_bin] = (235, 235, 235)
    base[gt_bin & ~pred_bin] = (178, 114, 0)
    base[pred_bin & ~gt_bin] = (0, 94, 213)
    out = cv2.resize(base.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.putText(out, f"{sid}  Dice {dice:.4f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "white TP  blue FN  orange FP", (6, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, out)


def _rng_state():
    st = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}
    if torch.cuda.is_available():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    torch.set_rng_state(st['torch'])
    np.random.set_state(st['numpy'])
    random.setstate(st['python'])
    if 'cuda' in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st['cuda'])


def save_last(path, epoch, model, hq, cadapter, optimizer, scheduler, ema, best, best_t, history):
    """Everything needed to continue training after a disconnect (written atomically)."""
    st = {'epoch': epoch, 'trainable': trainable_state(model),
          'hq': hq.state_dict() if hq is not None else None,
          'cadapter': cadapter.state_dict() if cadapter is not None else None,
          'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
          'ema': [s.detach().cpu().clone() for s in ema.shadow] if ema is not None else None,
          'ema_steps': ema.steps if ema is not None else 0,
          'best': best, 'best_t': best_t, 'history': history, 'cfg': cfg_snapshot(), 'rng': _rng_state()}
    tmp = path + '.tmp'
    torch.save(st, tmp)
    os.replace(tmp, path)


def load_last(path, model, hq, cadapter, optimizer, scheduler, ema):
    st = safe_torch_load(path, map_location='cpu')
    saved = st.get('cfg', {})
    keys = CFG_MUST_MATCH + ('EPOCHS', 'LR_DECODER', 'LR_ENCODER', 'LR_NEW', 'SCHEDULER', 'USE_EMA', 'K_ROUNDS')
    bad = {k: (saved.get(k), getattr(cfg, k)) for k in keys if k in saved and not _same_setting(saved.get(k), getattr(cfg, k))}
    if bad:
        raise ValueError(f"{path} was saved with different settings {bad}. "
                         "Use the same settings, or set cfg.RESUME = False to start this row again.")
    load_checkpoint({'trainable': st['trainable'], 'hq': st['hq'], 'cadapter': st['cadapter'], 'meta': {}},
                    model, hq, cadapter)
    optimizer.load_state_dict(st['optimizer'])
    scheduler.load_state_dict(st['scheduler'])
    if ema is not None and st['ema'] is not None:
        with torch.no_grad():
            for s_, v in zip(ema.shadow, st['ema']):
                s_.copy_(v.to(s_.device))
        ema.steps = st['ema_steps']
    _set_rng_state(st['rng'])
    return st['epoch'], st['best'], st['best_t'], st['history']


def evaluate_saved(ceiling_steps=None):
    """RUN='eval': re-test the saved checkpoint of the current row, no training.
      1) test on the native grid (the reported protocol) at t=0.5 and the saved t*
      2) the same weights on the old input grid (GT upsampled to IMG_SIZE, as in the
         earlier versions): separates the protocol effect from the method effect
      3) reachable-set ceilings at several fit lengths (convergence check)
      4) overlays of the worst and best test images."""
    set_seed(cfg.SEED)
    tag = f"{cfg.RUN_TAG}_{cfg.TASK}_{cfg.SUBSET}"
    path = os.path.join(cfg.CHECKPOINT_DIR, f"best_{tag}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — set RUN_TAG / TASK / SUBSET to the trained row.")
    print("=" * 72 + f"\n EVAL ONLY | {path}\n" + "=" * 72)
    ck = safe_torch_load(path, map_location='cpu')
    saved = ck['meta'].get('cfg')
    if saved:                                   # check before building: a mismatch would not even load
        bad = {k: (saved.get(k), getattr(cfg, k)) for k in CFG_MUST_MATCH if k in saved and not _same_setting(saved.get(k), getattr(cfg, k))}
        if bad:
            raise ValueError(f"Config differs from the checkpoint (saved, current): {bad}")
    else:
        print("  (checkpoint predates config saving: make sure the current Config matches the training run)")
    _, _, test_ids = make_splits()
    model, hq, cadapter = build_all()
    meta = load_checkpoint(ck, model, hq, cadapter)
    del ck
    t_star = meta['t_star']
    K, kr = cfg.K_EVAL, cfg.K_ROUNDS
    thresholds = tuple(sorted({0.5, t_star}))
    native_orig = cfg.NATIVE_RES
    res = {}
    try:
        for native in (True, False):
            cfg.NATIVE_RES = native
            dl = DataLoader(OCTADataset(test_ids), batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)
            res[native] = evaluate(model, hq, cadapter, dl, thresholds, K, 'giir', k_report=kr,
                                   compute_cldice=True, return_preds=native,
                                   desc="Test native grid" if native else f"Test {cfg.IMG_SIZE} grid")
    finally:
        cfg.NATIVE_RES = native_orig

    print(f"\nTEST  row {cfg.RUN_TAG}  n={len(test_ids)}  epoch {meta.get('epoch')}  t*={t_star:.2f}")
    for native in (True, False):
        for t in thresholds:
            print_test_row(res[native][t], t, K, kr, label=("native grid  " if native else f"{cfg.IMG_SIZE} grid     "))
    a = res[False][0.5]['curves'][:, kr - 1]
    b = res[True][0.5]['curves'][:, kr - 1]
    prot = paired_summary(a, b, f"protocol effect (native - {cfg.IMG_SIZE} grid), Dice@{kr} t=0.50")

    cfg.NATIVE_RES = True
    dl = DataLoader(OCTADataset(test_ids), batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)
    ceil_runs = {}
    try:
        for steps in (ceiling_steps or cfg.EVAL_CEILING_STEPS):
            c = reachable_ceiling(model, hq, dl, steps=steps)
            ceil_runs[steps] = c
            print(f"  ceiling after {steps:5d} fit steps: stride-4 basis {c['stock'].mean():.4f}"
                  + (f" | lifted basis {c['lifted'].mean():.4f}" if 'lifted' in c else "")
                  + f" | deployable@{kr} t=0.50 {b.mean():.4f}")
    finally:
        cfg.NATIVE_RES = native_orig

    o = res[True][0.5]
    d_final = o['curves'][:, kr - 1]
    order = np.argsort(d_final)
    print("  worst test images (Dice@%d, t=0.50): " % kr +
          ", ".join(f"{o_sid} {d_final[i]:.4f}" for i, o_sid in ((i, res[True]['sids'][i]) for i in order[:5])))
    if cfg.SAVE_OVERLAYS:
        out_dir = os.path.join(cfg.CHECKPOINT_DIR, f"overlays_{tag}")
        os.makedirs(out_dir, exist_ok=True)
        picks = [('worst', i) for i in order[:5]] + [('best', i) for i in order[::-1][:3]]
        for rank, (kind, i) in enumerate(picks):
            sid = res[True]['sids'][i]
            save_overlay(os.path.join(out_dir, f"{kind}{rank % 5 + 1}_{os.path.splitext(sid)[0]}_dice{d_final[i]:.3f}.png"),
                         sid, o['preds'][i], o['gts'][i], d_final[i])
        print(f"  overlays -> {out_dir}")

    npz = os.path.join(cfg.CHECKPOINT_DIR, f"eval_{tag}.npz")
    np.savez(npz, sids=np.array(res[True]['sids']), t_star=t_star, k_report=kr,
             curves_native_t05=res[True][0.5]['curves'], curves_native_tstar=res[True][t_star]['curves'],
             curves_input_t05=res[False][0.5]['curves'], curves_input_tstar=res[False][t_star]['curves'],
             cldice_native_t05=res[True][0.5]['cldice_per_image'],
             ceiling_steps=np.array(list(ceil_runs.keys())),
             ceiling_stock=np.stack([c['stock'] for c in ceil_runs.values()]) if ceil_runs else np.zeros(0),
             ceiling_lifted=np.stack([c['lifted'] for c in ceil_runs.values()])
             if ceil_runs and all('lifted' in c for c in ceil_runs.values()) else np.zeros(0),
             protocol=json.dumps(prot, default=str))
    print(f"  per-image arrays -> {npz}")
    return res, ceil_runs


# ============================================================================
# CHECKS ON SAVED CHECKPOINTS (no training)
#   RUN = 'calibre_check' : where does each adapter's gain land? (GT_LargeVessel)
#   RUN = 'vein_check'    : which vessels does each GT_Vein row mark as vein?
#   Both re-run the test of each saved checkpoint (GIIR, K_ROUNDS clicks, t = 0.50) and
#   analyse the masks. The Dice they print must equal Dice@4 (t = 0.50) of the adapter table:
#   same weights, same clicks, same seeds. Rows that are not trained yet are skipped.
# ============================================================================
import time

CAL_ROWS = ['A_decoder_only', 'B1_lora1_ep100', 'B_lora', 'C_convlora', 'AF_adaptformer',
            'H_hq_only_ep100', 'H1_lora1_hq', 'D_convlora_hq', 'B_lora_hq', 'R_rida']
CAL_THIN_R, CAL_THICK_R = 1, 3               # width bins by the largest disk that fits (native px, ~10 um per px):
                                              #   thin  = no disk of radius 1 fits  -> width 1-2 px
                                              #   thick = a disk of radius 3 fits   -> width >= 7 px; mid = between
CAL_PAIRS = [                                 # (from, to, what is added)
    ('A_decoder_only',  'H_hq_only_ep100', 'RBA read-out, no encoder branch'),
    ('B1_lora1_ep100',  'H1_lora1_hq',     'RBA read-out on LoRA r=1 (= RBA)'),
    ('C_convlora',      'D_convlora_hq',   'RBA read-out on Conv-LoRA r=16'),
    ('B_lora',          'B_lora_hq',       'RBA read-out on LoRA r=16'),
    ('A_decoder_only',  'B1_lora1_ep100',  'rank-1 encoder branch, no read-out'),
    ('H_hq_only_ep100', 'H1_lora1_hq',     'rank-1 encoder branch on the read-out'),
]
VEIN_CHECK_ROWS = ['A_decoder_only', 'H_hq_only_ep100', 'B1_lora1_ep100', 'C_convlora', 'D_convlora_hq',
                   'H1_lora1_hq']


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _opening(mask, r):
    """Binary opening with a disk; the image is padded by edge replication so border vessels keep their width."""
    from scipy import ndimage as ndi
    padded = np.pad(mask, r, mode='edge')
    return ndi.binary_opening(padded, structure=_disk(r))[r:-r, r:-r]


def calibre_maps(gt):
    """Width classes (by morphological opening: where does a disk of a given size fit?) and helper masks."""
    from scipy import ndimage as ndi
    gt = np.asarray(gt, bool)
    fits_mid = _opening(gt, CAL_THIN_R)
    fits_thick = _opening(gt, CAL_THICK_R)
    thin = gt & ~fits_mid
    thick = gt & fits_thick
    mid = gt & fits_mid & ~fits_thick
    skel = skeletonize(gt)
    edge = gt & ~ndi.binary_erosion(gt)
    band = ndi.binary_dilation(edge, iterations=1)                 # the edge, one pixel in and one pixel out
    far = ndi.distance_transform_edt(~gt) > 2.0                    # more than 2 px from any vessel pixel
    return thin, mid, thick, skel, band, far


def calibre_stats(preds, gts):
    """Pooled shares over the test images, plus per-image values for paired tests."""
    keys = ('thin', 'mid', 'thick', 'band', 'fp', 'skel')
    tot = {k: 0 for k in keys}
    hit = {k: 0 for k in keys}
    per = {k: [] for k in ('rec_thin', 'edge_err', 'skel_rec', 'far_fp')}
    for p, g in zip(preds, gts):
        p, g = np.asarray(p, bool), np.asarray(g, bool)
        if not g.any():
            for k in per:
                per[k].append(np.nan)
            continue
        thin, mid, thick, skel, band, far = calibre_maps(g)
        fp = p & ~g
        parts = {'thin': (thin, thin & p), 'mid': (mid, mid & p), 'thick': (thick, thick & p),
                 'band': (band, (p ^ g) & band), 'fp': (fp, fp & far), 'skel': (skel, skel & p)}
        for k, (t_, h_) in parts.items():
            tot[k] += int(t_.sum())
            hit[k] += int(h_.sum())
        per['rec_thin'].append((thin & p).sum() / thin.sum() if thin.any() else np.nan)
        per['edge_err'].append(((p ^ g) & band).sum() / max(band.sum(), 1))
        per['skel_rec'].append((skel & p).sum() / max(skel.sum(), 1))
        per['far_fp'].append(float((fp & far).sum()))
    share = lambda k: hit[k] / max(tot[k], 1)
    vessel = tot['thin'] + tot['mid'] + tot['thick']
    return dict(rec_thin=share('thin'), rec_mid=share('mid'), rec_thick=share('thick'), edge_err=share('band'),
                far_fp_share=share('fp'), skel_rec=share('skel'), thin_share_of_gt=tot['thin'] / max(vessel, 1),
                per={k: np.asarray(v, float) for k, v in per.items()})


def _load_row_for_check(row):
    """use_row + the saved checkpoint of this row on the current task; None if absent or different."""
    use_row(row)
    tag = f"{cfg.RUN_TAG}_{cfg.TASK}_{cfg.SUBSET}"
    path = os.path.join(cfg.CHECKPOINT_DIR, f"best_{tag}.pth")
    if not os.path.exists(path):
        print(f"  {row}: no checkpoint {path} - skipped")
        return None
    ck = safe_torch_load(path, map_location='cpu')
    saved = ck['meta'].get('cfg') or {}
    bad = {k: (saved.get(k), getattr(cfg, k)) for k in CFG_MUST_MATCH
           if k in saved and not _same_setting(saved.get(k), getattr(cfg, k))}
    if bad:
        print(f"  {row}: Config differs from the checkpoint (saved, current): {bad} - skipped")
        return None
    set_seed(cfg.SEED)
    model, hq, cadapter = build_all()
    load_checkpoint(ck, model, hq, cadapter)
    return model, hq, cadapter


def _release():
    """Call after deleting the caller's references to the model: frees the GPU before the next row."""
    if READOUT_TAP[0] is not None:                   # drop the hooks and the last reference to the model
        READOUT_TAP[0].remove()
        READOUT_TAP[0] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _predict(model, hq, cadapter, test_ids, desc):
    dl = DataLoader(OCTADataset(test_ids), batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)
    kr = cfg.K_ROUNDS
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = evaluate(model, hq, cadapter, dl, (0.5,), kr, 'giir', k_report=kr, return_preds=True, desc=desc)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sec = (time.time() - t0) / max(len(test_ids), 1)
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else float('nan')
    return out, sec, mem


def _paired(a, b, lower_is_better=False):
    d = np.asarray(b, float) - np.asarray(a, float)
    ok = ~np.isnan(d)
    d = d[ok]
    better = int((d < 0).sum()) if lower_is_better else int((d > 0).sum())
    try:
        from scipy import stats
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception:
        p = float('nan')
    return float(d.mean()) if len(d) else float('nan'), better, len(d), p


def calibre_check():
    print("=" * 72 + f"\n WHERE DOES THE GAIN LAND? | task {cfg.TASK} | {cfg.SUBSET} | no training\n" + "=" * 72)
    _, _, test_ids = make_splits()
    res = {}
    for row in CAL_ROWS:
        got = _load_row_for_check(row)
        if got is None:
            continue
        model, hq, cadapter = got
        del got
        n_adapt = sum(p.numel() for n, p in model.model.named_parameters()
                      if p.requires_grad and n.startswith('backbone.'))
        n_adapt += sum(p.numel() for p in hq.parameters()) if hq is not None else 0
        out, sec, mem = _predict(model, hq, cadapter, test_ids, f"calibre check {row}")
        o = out[0.5]
        st = calibre_stats(o['preds'], o['gts'])
        st.update(dice=float(o['curves'][:, cfg.K_ROUNDS - 1].mean()), sec=sec, mem=mem, adapter=n_adapt)
        res[row] = st
        del model, hq, cadapter, out, o
        _release()
    if not res:
        print("no trained rows found")
        return res
    any_st = next(iter(res.values()))
    print(f"\nWHERE DOES THE GAIN LAND?  {cfg.TASK}  {cfg.SUBSET}  n={len(test_ids)}  (GIIR, Dice@{cfg.K_ROUNDS}, t = 0.50)")
    print(f"  thin = width < {2 * CAL_THIN_R + 1} px ({any_st['thin_share_of_gt']:.0%} of vessel pixels), "
          f"thick = width >= {2 * CAL_THICK_R + 1} px, mid = between")
    print(f"  {'row':16s} {'params':>7s} {'Dice':>7s} {'thin rec':>9s} {'mid rec':>8s} {'thick rec':>10s} "
          f"{'edge err':>9s} {'centre-line':>12s} {'far FP':>7s} {'ms/img':>7s} {'peak MB':>8s}")
    for row, st in res.items():
        print(f"  {row:16s} {st['adapter'] / 1e6:6.2f}M {st['dice']:7.4f} {st['rec_thin']:9.1%} {st['rec_mid']:8.1%} "
              f"{st['rec_thick']:10.1%} {st['edge_err']:9.1%} {st['skel_rec']:12.1%} {st['far_fp_share']:7.1%} "
              f"{1000 * st['sec']:7.0f} {st['mem']:8.0f}")
    print("  edge err    = wrong pixels within 1 px of the vessel edge, as a share of that band (lower is better)")
    print("  centre-line = share of the GT centre-line covered by the mask (thin branches and connectivity)")
    print("  far FP      = share of false positives more than 2 px from any vessel (spurious vessels, lower is better)")
    print("  ms/img      = test time per image for the encoder pass and the clicks (includes loading and click logic)")
    print("\nWHAT EACH PART ADDS  (to - from; per-image paired, two-sided Wilcoxon p; 'better' = images that improved)")
    for a, b, what in CAL_PAIRS:
        if a not in res or b not in res:
            continue
        A, B = res[a]['per'], res[b]['per']
        lines = []
        for k, label, low in (('rec_thin', 'thin recall', False), ('edge_err', 'edge error', True),
                              ('skel_rec', 'centre-line', False)):
            m, better, n, p = _paired(A[k], B[k], lower_is_better=low)
            lines.append(f"{label} {m:+.3f} ({better}/{n} better, p={p:.0e})")
        dthick = res[b]['rec_thick'] - res[a]['rec_thick']
        lines.append(f"thick recall {dthick:+.3f}")
        print(f"  {what:38s} {a} -> {b}")
        print(f"      " + ";  ".join(lines))
    return res


def vein_stats(preds, gts, arts):
    """Pixel counts pooled over the test images -> three shares."""
    tp = fp = fp_art = n_vein = n_art = art_marked = 0
    for p, g, a in zip(preds, gts, arts):
        p, g, a = np.asarray(p, bool), np.asarray(g, bool), np.asarray(a, bool)
        fpm = p & ~g
        tp += int((p & g).sum())
        fp += int(fpm.sum())
        fp_art += int((fpm & a).sum())
        n_vein += int(g.sum())
        n_art += int(a.sum())
        art_marked += int((p & a).sum())
    return dict(vein_found=tp / max(n_vein, 1), artery_marked=art_marked / max(n_art, 1),
                fp_on_artery=fp_art / max(fp, 1))


def vein_check():
    print("=" * 72 + f"\n VEIN-IDENTITY CHECK | task {cfg.TASK} | {cfg.SUBSET} | no training\n" + "=" * 72)
    if cfg.TASK != 'GT_Vein':
        raise ValueError("set TASK = 'GT_Vein' for this check")
    art_dir = os.path.join(cfg.DATA_DIR, cfg.LABEL_ROOT_REL, 'GT_Artery')
    if not os.path.isdir(art_dir):
        raise FileNotFoundError(f"{art_dir} does not exist (the OCTA-500 artery labels are needed)")
    _, _, test_ids = make_splits()
    res = {}
    for row in VEIN_CHECK_ROWS:
        got = _load_row_for_check(row)
        if got is None:
            continue
        model, hq, cadapter = got
        del got
        out, _, _ = _predict(model, hq, cadapter, test_ids, f"vein check {row}")
        o = out[0.5]
        arts = []
        for sid, g in zip(out['sids'], o['gts']):
            a = np.array(Image.open(os.path.join(art_dir, sid)).convert("L")) > 0
            if a.shape != g.shape:
                raise ValueError(f"{sid}: GT_Artery {a.shape} vs GT_Vein {g.shape}")
            arts.append(a)
        st = vein_stats(o['preds'], o['gts'], arts)
        st['dice'] = float(o['curves'][:, cfg.K_ROUNDS - 1].mean())
        res[row] = st
        del model, hq, cadapter, out, o
        _release()
    print(f"\nVEIN-IDENTITY CHECK  {cfg.TASK}  {cfg.SUBSET}  n={len(test_ids)}  (GIIR, Dice@{cfg.K_ROUNDS}, t = 0.50)")
    print(f"  {'row':16s} {'Dice':>7s} {'vein found':>11s} {'artery marked':>14s} {'FP on arteries':>15s}")
    for row, st in res.items():
        print(f"  {row:16s} {st['dice']:7.4f} {st['vein_found']:11.1%} {st['artery_marked']:14.1%} "
              f"{st['fp_on_artery']:15.1%}")
    print("  artery marked  = share of the artery pixels that the model marks as vein")
    print("  FP on arteries = share of the false positives that lie on arteries")
    return res


# ============================================================================
# COMMAND LINE
# ============================================================================
DATASETS = {   # (subset, task) -> (DATA_DIR, SPLIT_FILE); written by prepare_data.py
    'OCTA_3mm': ('octa', 'data_split_paper.pth'),
    'OCTA_6mm': ('octa', 'data_split_paper_6mm.pth'),
    ('ROSE_O', 'GT_RoseVessel'): ('octa_rose', 'data_split_rose_vessel.pth'),
    ('ROSE_O', 'GT_RoseFAZ'): ('octa_rose', 'data_split_rose_faz.pth'),
}


def configure(row=None, task=None, subset=None, run=None, epochs=None, data_dir=None, split_file=None,
              checkpoint_dir=None):
    """Set the Config for one row, target and dataset (used by the command line and evaluate_five_metrics.py)."""
    if row:
        use_row(row)
    if task:
        cfg.TASK = task
    if subset:
        cfg.SUBSET = subset
    d = DATASETS.get((cfg.SUBSET, cfg.TASK)) or DATASETS.get(cfg.SUBSET)
    if d:
        cfg.DATA_DIR, cfg.SPLIT_FILE = d
    if data_dir:
        cfg.DATA_DIR = data_dir
    if split_file:
        cfg.SPLIT_FILE = split_file
    if checkpoint_dir:
        cfg.CHECKPOINT_DIR = checkpoint_dir
    if epochs:
        cfg.EPOCHS = int(epochs)
    if run:
        cfg.RUN = run


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Train or test one row of the RBA paper.")
    ap.add_argument('--row', choices=sorted(ROWS), help="configuration (default: Config.RUN_TAG, = RBA)")
    ap.add_argument('--task', help="GT_LargeVessel | GT_Vein | GT_Artery | GT_FAZ | GT_Capillary | GT_RoseVessel | GT_RoseFAZ")
    ap.add_argument('--subset', choices=['OCTA_3mm', 'OCTA_6mm', 'ROSE_O'])
    ap.add_argument('--run', choices=['train', 'eval', 'calibre_check', 'vein_check'])
    ap.add_argument('--epochs', type=int, help="default 100, as in the paper")
    ap.add_argument('--data-dir')
    ap.add_argument('--split-file')
    ap.add_argument('--checkpoint-dir')
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    configure(a.row or (cfg.RUN_TAG if cfg.RUN_TAG in ROWS else None), a.task, a.subset, a.run, a.epochs,
              a.data_dir, a.split_file, a.checkpoint_dir)
    if cfg.RUN == 'calibre_check':
        calibre_results = calibre_check()
    elif cfg.RUN == 'vein_check':
        vein_results = vein_check()
    elif cfg.RUN == 'eval':
        evaluate_saved()
    else:
        main()
