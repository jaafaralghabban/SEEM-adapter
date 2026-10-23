# ============================================================================
# RBA MODULE, EXPLAINED — the Read-out Basis Adapter for SEEM, with checks of every claim
#   The two classes below are copied unchanged from train_paper1_hq.py:
#     HQBasis    = RBA's read-out branch  (203,328 parameters)
#     LoRALinear = RBA's encoder branch   (rank 1 in 96 FocalNet-L layers, 270,840 parameters)
#   The cell needs only PyTorch and runs on a CPU in about a minute: SEEM itself is replaced by
#   random tensors of SEEM's sizes, and FocalNet-L by its linear layers with their true sizes.
#   It checks: parameter counts, exact start, the basis view, the kept directions (null space),
#   ordered learning; it also prints the cost of one click.
# ============================================================================
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    SEED = 0
    DEVICE = 'cpu'
    IMG_SIZE = 512                      # SEEM input after resizing
    C_MASK = 512                        # channels of SEEM's mask features F (and of w_q)
    C_RES2 = 192                        # channels of FocalNet-L stage 1 (res2)
    N_QUERIES = 101                     # SEEM's decoder queries = candidates
    HQ_RANK = 32                        # number of basis maps in G(x)
    HQ_STRIDE = 2                       # G at 256 x 256 for a 512 input
    LORA_RANK = 1
    LORA_ALPHA = 2                      # alpha / r = 2
    LORA_DROPOUT = 0.0
    FOCALNET_DIMS = (192, 384, 768, 1536)   # FocalNet-L, as in SEEM
    FOCALNET_DEPTHS = (2, 2, 18, 2)         # 24 blocks
    FOCAL_LEVEL = 4


cfg = Config()


# ----------------------------------------------------------------------------- RBA, unchanged from the trainer
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


# ----------------------------------------------------------------------------- the insertion, as in the trainer
def inject_rank1_lora(backbone, r, alpha, dropout):
    """inject_encoder_adapters(kind='lora') of the trainer: wrap f, fc1, fc2 and proj (nn.Linear only)."""
    targets = []
    for mname, module in backbone.named_modules():
        for child_name in ('f', 'fc1', 'fc2', 'proj'):
            child = getattr(module, child_name, None)
            if isinstance(child, nn.Linear):
                targets.append((mname, module, child_name, child))
    for mname, module, child_name, child in targets:      # replace after collecting
        setattr(module, child_name, LoRALinear(child, r, alpha, dropout, conv=False))
    return len(targets)


# ----------------------------------------------------------------------------- FocalNet-L's linear layers, true sizes
class Modulation(nn.Module):
    def __init__(self, dim, focal_level):
        super().__init__()
        self.f = nn.Linear(dim, 2 * dim + focal_level + 1)    # focal modulation input projection
        self.h = nn.Conv2d(dim, dim, 1)                       # a 1x1 conv: not adapted (not nn.Linear)
        self.proj = nn.Linear(dim, dim)


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)


class Block(nn.Module):
    def __init__(self, dim, focal_level):
        super().__init__()
        self.modulation = Modulation(dim, focal_level)
        self.mlp = Mlp(dim)


class Downsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 2, stride=2)       # named 'proj' but a conv: skipped


class FocalNetLLinearLayers(nn.Module):
    def __init__(self, dims, depths, focal_level):
        super().__init__()
        self.layers = nn.ModuleList(nn.ModuleList(Block(d, focal_level) for _ in range(n))
                                    for d, n in zip(dims, depths))
        self.downsample = nn.ModuleList(Downsample(a, b) for a, b in zip(dims[:-1], dims[1:]))


# ----------------------------------------------------------------------------- helpers
def n_params(module):
    return sum(p.numel() for p in module.parameters())


def report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    torch.manual_seed(cfg.SEED)
    S, C, Q, r = cfg.IMG_SIZE, cfg.C_MASK, cfg.N_QUERIES, cfg.HQ_RANK
    oks = []
    print("=" * 96)
    print("RBA MODULE CHECKS  (SEEM replaced by random tensors of its sizes)")
    print("=" * 96)

    # 1. parameter counts ------------------------------------------------------------------
    hq = HQBasis(c_mask=C, c_res2=cfg.C_RES2, rank=r, stride=cfg.HQ_STRIDE).to(cfg.DEVICE)
    parts = {'stem S': hq.stem, 'projection P2 (res2)': hq.p_res2, 'projection PF (F)': hq.p_mask,
             'fuse Psi': hq.fuse, 'U': hq.U}
    print("\n1. Parameter counts")
    for k, m in parts.items():
        print(f"     {k:<22} {n_params(m):>9,}")
    oks.append(report("read-out branch", n_params(hq) == 203_328, f"{n_params(hq):,} (paper: 203,328)"))
    enc = FocalNetLLinearLayers(cfg.FOCALNET_DIMS, cfg.FOCALNET_DEPTHS, cfg.FOCAL_LEVEL)
    n_layers = inject_rank1_lora(enc, cfg.LORA_RANK, cfg.LORA_ALPHA, cfg.LORA_DROPOUT)
    n_lora = sum(p.numel() for n, p in enc.named_parameters() if 'lora_' in n)
    oks.append(report("encoder branch", n_lora == 270_840 and n_layers == 96,
                      f"{n_lora:,} in {n_layers} layers (paper: 270,840 in 96)"))
    print(f"     RBA in total: {n_params(hq) + n_lora:,} parameters")

    # SEEM-sized stand-ins: image, res2, mask features F, query vectors w, SEEM's logits <w, F>
    img = torch.rand(1, 3, S, S) * 255
    res2 = torch.randn(1, cfg.C_RES2, S // 4, S // 4)
    Fm = torch.randn(1, C, S // 4, S // 4)
    w = torch.randn(1, Q, C) / math.sqrt(C)
    stock = torch.einsum('bqc,bchw->bqhw', w, Fm)                       # SEEM's own read-out, 128 x 128

    # 2. exact start ---------------------------------------------------------------------
    print("\n2. Exact start: U = 0, so the lifted masks equal SEEM's masks, upsampled")
    with torch.no_grad():
        G = hq.basis(img, res2, Fm)
        lifted = hq.readout(stock, w, G)
        up = F.interpolate(stock, size=G.shape[-2:], mode='bilinear', align_corners=False)
    d = (lifted - up).abs().max().item()
    oks.append(report("read-out at the start", d == 0.0, f"max |lifted - Up(SEEM)| = {d:g}, G is {tuple(G.shape)}"))
    lin = enc.layers[2][0].mlp.fc1
    x = torch.randn(4, lin.base.in_features)
    d2 = (lin(x) - lin.base(x)).abs().max().item()
    oks.append(report("encoder layer at the start", d2 == 0.0, f"max |LoRA layer - frozen layer| = {d2:g} (B = 0)"))

    # 3. the basis view ----------------------------------------------------------------------
    print("\n3. Basis view: Up(<w, F>) = <w, Up(F)>, so RBA's new basis is Up(F) + U^T G")
    with torch.no_grad():
        Fup = F.interpolate(Fm, size=G.shape[-2:], mode='bilinear', align_corners=False)
        via_basis = torch.einsum('bqc,bchw->bqhw', w, Fup)
    d3 = ((via_basis - up).abs().max() / up.abs().max()).item()
    oks.append(report("upsampling commutes with the read-out", d3 < 1e-5, f"relative difference {d3:.1e} (float32 rounding)"))

    # 4. kept directions -----------------------------------------------------------------------
    print("\n4. Kept directions: after training, any w with U w = 0 still gives SEEM's mask")
    with torch.no_grad():
        hq.U.weight.normal_(0, 0.05)                                    # stands in for a trained U
        Uw = hq.U.weight                                                # [r, C]
        P = torch.eye(C) - Uw.T @ torch.linalg.solve(Uw @ Uw.T, Uw)    # projector onto null(U)
        w0 = w @ P                                                      # P is symmetric
        lifted0 = hq.readout(torch.einsum('bqc,bchw->bqhw', w0, Fm), w0, G)
        up0 = F.interpolate(torch.einsum('bqc,bchw->bqhw', w0, Fm), size=G.shape[-2:], mode='bilinear',
                            align_corners=False)
    d4 = ((lifted0 - up0).abs().max() / up0.abs().max()).item()
    oks.append(report("w in null(U)", d4 < 1e-5, f"relative difference {d4:.1e}; RBA changes at most {r} of {C} directions"))

    # 5. ordered learning ----------------------------------------------------------------------
    print("\n5. Ordered learning: at the start only U (and LoRA's B) receive gradient")
    hq2 = HQBasis(c_mask=C, c_res2=cfg.C_RES2, rank=r, stride=cfg.HQ_STRIDE)
    target = (torch.rand(1, Q, S // 2, S // 2) > 0.9).float()
    loss = F.binary_cross_entropy_with_logits(hq2.readout(stock, w, hq2.basis(img, res2, Fm)), target)
    loss.backward()
    g_u = hq2.U.weight.grad.abs().sum().item()
    g_rest = sum(p.grad.abs().sum().item() for n, p in hq2.named_parameters() if not n.startswith('U.') and p.grad is not None)
    oks.append(report("read-out branch, first step", g_u > 0 and g_rest == 0.0,
                      f"|grad U| = {g_u:.3g}, |grad of the layers that build G| = {g_rest:g}"))
    lin.zero_grad()
    lin(x).pow(2).sum().backward()
    gA, gB = lin.lora_A.weight.grad.abs().sum().item(), lin.lora_B.weight.grad.abs().sum().item()
    oks.append(report("encoder branch, first step", gB > 0 and gA == 0.0, f"|grad B| = {gB:.3g}, |grad A| = {gA:g}"))

    # 6. cost of one click ---------------------------------------------------------------------
    print("\n6. Cost: G is built once per image; a click only re-reads it")
    with torch.no_grad():
        t0 = time.perf_counter(); G = hq.basis(img, res2, Fm); t1 = time.perf_counter()
        sel = stock[:, :1]; w_sel = w[:, :1]                            # the one candidate SEEM chose
        t2 = time.perf_counter(); hq.readout(sel, w_sel, G); t3 = time.perf_counter()
    macs = C * r + r * (S // 2) ** 2
    print(f"     basis(), once per image : {1000 * (t1 - t0):7.1f} ms on this CPU")
    print(f"     readout(), per click    : {1000 * (t3 - t2):7.1f} ms on this CPU")
    print(f"     multiply-adds per click : {C} x {r} (U) + {r} x {S // 2} x {S // 2} (weighted sum) = {macs:,} "
          f"(about 2.1 million)")

    print("\n" + "=" * 96)
    print(f"{sum(oks)}/{len(oks)} checks passed")


if __name__ == "__main__":
    main()