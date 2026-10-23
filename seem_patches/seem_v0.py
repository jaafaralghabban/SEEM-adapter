# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All at Once
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import logging
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init

from .build import register_decoder
from .modules import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from .prototype.attention_data_struct_seemv0 import AttentionDataStruct
from ..utils import rand_sample_plain as rand_sample
from ..utils import prepare_features, configurable
from ..modules import PositionEmbeddingSine
from ..modules.point_features import point_sample


class SpatialPositionalPrompt(nn.Module):
    """Content-independent learned positional prompt for spatial click tokens.

    Maps normalized (y, x) click coordinates -> hidden_dim embedding via fixed
    Fourier features + a small MLP. It depends ONLY on *where* the click is, never
    on the image feature at that location, so it injects a strong, distinctive
    'click-here' signal even on thin / low-contrast vessels where point_sample
    collapses to background. This is the missing positional signal that lets
    accumulating clicks actually move the predicted mask.
    """
    def __init__(self, hidden_dim, num_freqs=10, mlp_ratio=2, out_std=0.1):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = (2.0 ** torch.arange(num_freqs).float()) * 3.141592653589793
        self.register_buffer('freqs', freqs)            # [num_freqs]
        in_dim = 2 * 2 * num_freqs                      # (sin,cos) x 2 coords x num_freqs
        hidden = hidden_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden_dim),
        )
        nn.init.normal_(self.mlp[-1].weight, std=out_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, coords):
        # coords: [..., 2] normalized in [0, 1]
        c = coords.unsqueeze(-1) * self.freqs           # [..., 2, num_freqs]
        feats = torch.cat([torch.sin(c), torch.cos(c)], dim=-1)  # [..., 2, 2*num_freqs]
        feats = feats.flatten(-2)                       # [..., 4*num_freqs]
        return self.mlp(feats)                           # [..., hidden_dim]


class SEEMDecoder(nn.Module):

    @configurable
    def __init__(
        self,
        lang_encoder: nn.Module,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        dim_proj: int,
        num_queries: int,
        contxt_len: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        task_switch: dict,
        enforce_input_project: bool,
        max_spatial_len: int,
        attn_arch: dict,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.task_switch = task_switch
        self.query_index = {}

        # output FFNs
        self.lang_encoder = lang_encoder
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)

        if task_switch['bbox']:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if task_switch['spatial']:
            # spatial query
            self.mask_sptial_embed = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
            trunc_normal_(self.mask_sptial_embed[0], std=.02)
            trunc_normal_(self.mask_sptial_embed[1], std=.02)
            trunc_normal_(self.mask_sptial_embed[2], std=.02)

            self.max_spatial_len = max_spatial_len
            # spatial memory
            num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
            self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
            self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)

            # learnable positive negative indicator
            self.pn_indicator = nn.Embedding(2, hidden_dim)

            # === content-independent learned positional click prompt (ADDED) ===
            self.spatial_pos_prompt = SpatialPositionalPrompt(hidden_dim, num_freqs=10, out_std=0.1)
            self.pos_prompt_scale = nn.Parameter(torch.tensor(1.0))   # scales positional prompt
            self.content_gate = nn.Parameter(torch.tensor(1.0))       # gates the (often-silent) pooled content
            self.use_learned_pos_prompt = True   # set False to recover EXACT original SEEM behavior
            self.pos_prompt_on_tokens = True     # (A) per-point spatial tokens  -> candidate masks
            self.pos_prompt_on_meanq = True      # (B) mean pos/neg query        -> selection key

        # build AttentionDataStruct
        attn_arch['NUM_LAYERS'] = self.num_layers
        self.attention_data = AttentionDataStruct(attn_arch, task_switch)

    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
        ret = {}

        ret["lang_encoder"] = lang_encoder
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
        ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']

        # Transformer parameters:
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert dec_cfg['DEC_LAYERS'] >= 1
        ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
        ret["pre_norm"] = dec_cfg['PRE_NORM']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["task_switch"] = extra['task_switch']
        ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']

        # attn data struct
        ret["attn_arch"] = cfg['ATTENTION_ARCH']

        return ret

    def _masked_mean_pe(self, coords, valid):
        """coords [bs,P,2] normalized; valid [bs,P] bool. Returns [bs,hidden] mean PE over valid points."""
        pe = self.spatial_pos_prompt(coords.clamp(min=0.0))   # [bs,P,C]
        w = valid.float().unsqueeze(-1)                       # [bs,P,1]
        s = (pe * w).sum(dim=1)                               # [bs,C]
        d = w.sum(dim=1).clamp(min=1.0)                       # [bs,1]
        return s / d

    def forward(self, x, mask_features, mask=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels; del mask
        spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg' or 'refimg_tokens' in extra
        grounding_extra_flag = 'grounding_tokens' in extra.keys()
        spatial_memory_flag = 'prev_mask' in extra.keys()
        flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag, "memories_spatial": spatial_memory_flag}
        self.attention_data.reset(flags, task, extra)

        src, pos, size_list = prepare_features(x, self.num_feature_levels, self.pe_layer, self.input_proj, self.level_embed)
        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        self.attention_data.set('queries_object', 'queries', output, query_embed)

        if self.task_switch['spatial'] and spatial_extra_flag:
            if 'refimg_tokens' not in extra:
                # get divisor
                _,h,w = extra['spatial_query_pos_mask'][0].shape
                divisor = torch.tensor([h,w], device=output.device)[None,]

                # Get mean pos spatial query
                non_zero_pos_point = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[-1]).t() for m in extra['spatial_query_pos_mask']]
                non_zero_pos_point = nn.utils.rnn.pad_sequence(non_zero_pos_point, padding_value=-1).permute(1,0,2)
                non_zero_pos_mask = (non_zero_pos_point.sum(dim=-1) < 0)
                spatial_query_pos = point_sample(mask_features, non_zero_pos_point.flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
                spatial_query_pos = torch.stack([x[m].mean(dim=0, keepdim=True) for x, m in zip(spatial_query_pos.transpose(1,2), ~non_zero_pos_mask)]).transpose(0,1).nan_to_num()
                if getattr(self, 'use_learned_pos_prompt', False) and getattr(self, 'pos_prompt_on_meanq', False):
                    pe_pos = self._masked_mean_pe(non_zero_pos_point, ~non_zero_pos_mask)   # [bs,C]
                    spatial_query_pos = spatial_query_pos + self.pos_prompt_scale * pe_pos[None]

                # Get mean neg spatial query
                non_zero_neg_point = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[-1]).t() for m in extra['spatial_query_neg_mask']]
                non_zero_neg_point = nn.utils.rnn.pad_sequence(non_zero_neg_point, padding_value=-1).permute(1,0,2)
                non_zero_neg_mask = (non_zero_neg_point.sum(dim=-1) < 0)
                spatial_query_neg = point_sample(mask_features, non_zero_neg_point.flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
                spatial_query_neg = torch.stack([x[m].mean(dim=0, keepdim=True) for x, m in zip(spatial_query_neg.transpose(1,2), ~non_zero_neg_mask)]).transpose(0,1).nan_to_num()
                if getattr(self, 'use_learned_pos_prompt', False) and getattr(self, 'pos_prompt_on_meanq', False):
                    pe_neg = self._masked_mean_pe(non_zero_neg_point, ~non_zero_neg_mask)   # [bs,C]
                    spatial_query_neg = spatial_query_neg + self.pos_prompt_scale * pe_neg[None]

                # merge positive and negative sample points for self attention
                # pos_neg_points = [x|y for x,y in zip(extra['spatial_query_pos_mask'], extra['spatial_query_neg_mask'])]

                # Get layerwise spatial query
                src_spatial_queries = []
                src_spatial_maskings = []
                for i in range(len(src)):
                    hw,_,dc = src[i].shape
                    src_mask_features = src[i].view(size_list[i][0],size_list[i][1],bs,dc)
                    src_mask_features = src_mask_features @ self.mask_sptial_embed[i]

                    non_zero_query_point_pos = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[i]).t() for m in extra['spatial_query_pos_mask']]
                    non_zero_query_point_neg = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[i]).t() for m in extra['spatial_query_neg_mask']]
                    non_zero_query_point = [torch.cat([x,y], dim=0) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]

                    pos_neg_indicator = [torch.cat([torch.ones(x.shape[0], device=x.device), -torch.ones(y.shape[0], device=y.device)]) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]
                    pos_neg_indicator = nn.utils.rnn.pad_sequence(pos_neg_indicator, padding_value=0)

                    non_zero_query_point = nn.utils.rnn.pad_sequence(non_zero_query_point, padding_value=-1).permute(1,0,2)
                    non_zero_query_mask = (non_zero_query_point.sum(dim=-1) < 0)
                    non_zero_query_point[non_zero_query_mask] = 0

                    spatial_tokens = point_sample(src_mask_features.permute(2,3,0,1), non_zero_query_point.flip(dims=(2,)).type(src_mask_features.dtype), align_corners=True).permute(2,0,1)
                    if getattr(self, 'use_learned_pos_prompt', False) and getattr(self, 'pos_prompt_on_tokens', False):
                        # content-independent positional prompt, one per click point
                        pe_tok = self.spatial_pos_prompt(non_zero_query_point).permute(1,0,2)        # [P,bs,C]
                        pe_tok = pe_tok * (~non_zero_query_mask).permute(1,0).unsqueeze(-1).type_as(pe_tok)
                        spatial_tokens = self.content_gate * spatial_tokens + self.pos_prompt_scale * pe_tok
                    spatial_tokens[pos_neg_indicator==1] += self.pn_indicator.weight[0:1]
                    spatial_tokens[pos_neg_indicator==-1] += self.pn_indicator.weight[1:2]

                    src_spatial_queries += [spatial_tokens]
                    src_spatial_maskings += [non_zero_query_mask]

                if 'refimg' in task:
                    output_refimg = {}
                    output_refimg['spatial_query_pos'] = spatial_query_pos
                    output_refimg['spatial_query_neg'] = spatial_query_neg
                    output_refimg['src_spatial_queries'] = src_spatial_queries
                    output_refimg['src_spatial_maskings'] = src_spatial_maskings
                    return output_refimg
            else:
                spatial_query_pos = extra['refimg_tokens']['spatial_query_pos']
                spatial_query_neg = extra['refimg_tokens']['spatial_query_neg']
                src_spatial_queries = extra['refimg_tokens']['src_spatial_queries']
                src_spatial_maskings = extra['refimg_tokens']['src_spatial_maskings']

            # Get object query for spatial index
            self.attention_data.set('queries_spatial', 'queries')

            # set spatial memory
            spatial_output = self.spatial_featured.weight.unsqueeze(1).repeat(1, bs, 1)
            spatial_embed = self.spatial_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            self.attention_data.set('memories_spatial', 'memories', spatial_output, spatial_embed)

            # if 'queries_spatial' in extra:
            #     self.attention_data.set('queries_spatial', 'queries', var=extra['queries_spatial'])

            # if spatial_memory_flag:
            #     prev_mask = (extra['prev_mask'].sigmoid() > 0.5).detach()
            #     non_zero_query_point = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[-1]).t() for m in prev_mask]
            #     non_zero_query_point = nn.utils.rnn.pad_sequence(non_zero_query_point, padding_value=-1).permute(1,0,2)
            #     non_zero_query_mask = (non_zero_query_point.sum(dim=-1) < 0)
            #     spatial_memory = point_sample(mask_features, non_zero_query_point.flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
            #     spatial_memory = torch.stack([x[m].mean(dim=0, keepdim=True) for x, m in zip(spatial_memory.transpose(1,2), ~non_zero_query_mask)]).transpose(0,1).nan_to_num()

        if self.task_switch['grounding'] and grounding_extra_flag:
            # Get grounding tokens
            grounding_tokens = extra['grounding_tokens']
            _grounding_tokens = grounding_tokens.detach().clone()

            self.attention_data.set('tokens_grounding', 'tokens', grounding_tokens, _grounding_tokens)
            self.attention_data.set('queries_grounding', 'queries')
            self.attention_data.set_maskings('tokens_grounding', extra['grounding_nonzero_mask'])

        output, query_embed = self.attention_data.cross_attn_variables()
        # prediction heads on learnable query features
        results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        results["predictions_pos_spatial"] = spatial_query_pos.transpose(0,1) if spatial_extra_flag else None
        results["predictions_neg_spatial"] = spatial_query_neg.transpose(0,1) if spatial_extra_flag else None
        self.attention_data.set_results(results)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # CROSS ATTENTION
            output, avg_attn = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=self.attention_data.cross_attn_mask(size_list[level_index], self.num_heads),
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            self.attention_data.update_variables(output, 'cross_attn')

            # SELF ATTENTION
            self_attn_mask = torch.zeros((bs, self.num_queries, self.num_queries), device=query_embed.device).bool() # Default False (attend oq)
            if self.task_switch['spatial'] and spatial_extra_flag:
                # get spatial tokens
                spatial_tokens = src_spatial_queries[level_index]
                _spatial_tokens = spatial_tokens.detach().clone()

                self.attention_data.set('tokens_spatial', 'tokens', spatial_tokens, _spatial_tokens)
                self.attention_data.set_maskings('tokens_spatial', src_spatial_maskings[level_index])

            output, query_embed, self_attn_mask = self.attention_data.self_attn(bs, self.num_heads)

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_attn_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed)

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            self.attention_data.update_variables(output, 'self_attn')
            output, query_embed = self.attention_data.cross_attn_variables()
            results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], layer_id=i)
            results["predictions_pos_spatial"] = spatial_query_pos.transpose(0,1) if spatial_extra_flag else None
            results["predictions_neg_spatial"] = spatial_query_neg.transpose(0,1) if spatial_extra_flag else None
            self.attention_data.set_results(results)

        return self.attention_data.organize_output()

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, layer_id=-1):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        class_embed = decoder_output @ self.class_embed
        outputs_class = self.lang_encoder.compute_similarity(class_embed)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        
        outputs_bbox = [None for i in range(len(outputs_mask))]
        if self.task_switch['bbox']:
            outputs_bbox = self.bbox_embed(decoder_output)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        outputs_caption = class_embed

        results = {
            "attn_mask": attn_mask,
            "predictions_class": outputs_class,
            "predictions_mask": outputs_mask,
            "predictions_bbox": outputs_bbox,
            "predictions_caption": outputs_caption,
            "predictions_maskemb": mask_embed,
        }
        return results

@register_decoder
def get_seem_interface(cfg, in_channels, lang_encoder, mask_classification, extra):
    return SEEMDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
