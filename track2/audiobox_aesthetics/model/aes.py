# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import logging
from typing import Dict
from torch import nn
import torch

from .utils import create_mlp_block
from .wavlm import WavLM, WavLMConfig
from .pooling import (
    Temporal_Average_Pooling,
    Temporal_Statistics_Pooling,
    Self_Attentive_Pooling,
    Attentive_Statistics_Pooling,
    General_Self_Attentive_Pooling,
    General_Attentive_Statistics_Pooling,
    Dual_Resolution_Attentive_Pooling,
    Dual_Resolution_Statistics_Pooling,
)
from huggingface_hub import PyTorchModelHubMixin

logging = logging.getLogger(__name__)


DEFAULT_AUDIO_CFG = WavLMConfig(
    {
        "extractor_mode": "default",
        "encoder_layers": 12,
        "encoder_embed_dim": 768,
        "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12,
        "activation_fn": "gelu",
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.05,
        "dropout_input": 0.1,
        "dropout_features": 0.1,
        "layer_norm_first": False,
        "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        "conv_bias": False,
        "feature_grad_mult": 0.1,
        "mask_length": 10,
        "mask_prob": 0.8,
        "mask_selection": "static",
        "mask_other": 0.0,
        "no_mask_overlap": False,
        "mask_min_space": 1,
        "mask_channel_length": 10,
        "mask_channel_prob": 0.0,
        "mask_channel_selection": "static",
        "mask_channel_other": 0.0,
        "no_mask_channel_overlap": False,
        "mask_channel_min_space": 1,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "relative_position_embedding": True,
        "num_buckets": 320,
        "max_distance": 800,
        "gru_rel_pos": True,
        "normalize": False,
    }
)


@dataclass(eq=False)
class Normalize:
    mean: float
    std: float

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


AXES_NAME = ["CE", "CU", "PC", "PQ"]

@dataclass(eq=False)
class AesMultiOutput(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/facebookresearch/audiobox-aesthetics",
    pipeline_tag="audio-classification",
    license="cc-by-4.0",
):
    proj_num_layer: int = 1
    proj_ln: bool = False
    proj_act_fn: str = "gelu"
    proj_dropout: float = 0
    nth_layer: int = 13
    use_weighted_layer_sum: bool = True
    precision: str = "32"
    normalize_embed: bool = True
    output_dim: int = 1
    target_transform: Dict[str, Dict[str, float]] = None
    freeze_encoder: bool = True  # trf encoder freeze true means no weight update
    pooling_type: str = "tap"
    segment_size: int = 1  # new: segment size for general attentive pooling

    def __post_init__(self):
        super().__init__()
        amodel_cfg = DEFAULT_AUDIO_CFG
        self.wavlm_model = WavLM(amodel_cfg)
        wavlm_out_dim = self.wavlm_model.cfg.encoder_embed_dim

        # 選擇 pooling 類型並初始化對應投影層
        if self.pooling_type == 'tap':
            self.pooling = Temporal_Average_Pooling(wavlm_out_dim)
            pool_out_dim = wavlm_out_dim
        elif self.pooling_type == 'tsp':
            self.pooling = Temporal_Statistics_Pooling(wavlm_out_dim)
            pool_out_dim = wavlm_out_dim * 2
        elif self.pooling_type == 'sap':
            self.pooling = Self_Attentive_Pooling(wavlm_out_dim)
            pool_out_dim = wavlm_out_dim
        elif self.pooling_type == 'asp':
            self.pooling = Attentive_Statistics_Pooling(wavlm_out_dim)
            pool_out_dim = wavlm_out_dim * 2
        elif self.pooling_type == 'gsap':
            self.pooling = General_Self_Attentive_Pooling(wavlm_out_dim, segment_size=self.segment_size)
            pool_out_dim = wavlm_out_dim
        elif self.pooling_type == 'gasp':
            self.pooling = General_Attentive_Statistics_Pooling(wavlm_out_dim, segment_size=self.segment_size)
            pool_out_dim = wavlm_out_dim * 2
        elif self.pooling_type == 'drap':
            self.pooling = Dual_Resolution_Attentive_Pooling(wavlm_out_dim, segment_size=self.segment_size)
            pool_out_dim = wavlm_out_dim
        elif self.pooling_type == 'drsp':
            self.pooling = Dual_Resolution_Statistics_Pooling(wavlm_out_dim, segment_size=self.segment_size)
            pool_out_dim = wavlm_out_dim * 2
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        # 統一的 projection 層，將 pool_out_dim 投影回 upstream_feat_dim
        self.pooling_proj = nn.Linear(in_features=pool_out_dim, out_features=wavlm_out_dim)

        self.axes_name = AXES_NAME
        self.proj_layer = nn.ModuleDict(
            {
                x: nn.Sequential(
                    *create_mlp_block(
                        wavlm_out_dim,
                        self.output_dim,
                        self.proj_num_layer,
                        self.proj_act_fn,
                        self.proj_ln,
                        dropout=self.proj_dropout,
                    )
                )
                for x in self.axes_name
            }
        )

        if self.use_weighted_layer_sum:
            self.layer_weights = nn.ParameterDict(
                {
                    x: torch.nn.Parameter(
                        torch.ones(self.nth_layer) / (self.nth_layer)
                    )
                    for x in self.axes_name
                }
            )

        precision_map = {
            "64": torch.float64,
            "32": torch.float32,
            "16": torch.half,
            "bf16": torch.bfloat16,
        }
        self.precision = precision_map[str(self.precision)]
        self.enable_autocast = str(self.precision) in {"16", "bf16"}
        logging.info(
            f"model precision: {self.precision}, enable autocast: {self.enable_autocast}",
        )

    def forward(self, batch):
        # 1. 準備輸入：batch["wav"] shape [B,1,T]；batch["mask"] shape [B,1,T]（False 表示有效）
        assert batch["wav"].ndim == 3

        wav = batch["wav"].squeeze(1)

        if "mask" in batch:
            padding_mask = ~batch["mask"].squeeze(1)
        else:
            padding_mask = torch.zeros_like(wav, dtype=torch.bool)

        # 2. 自動混合精度與是否可微分設定
        with (
            torch.amp.autocast(
                device_type=wav.device.type,
                dtype=self.precision,
                enabled=self.enable_autocast,
            ),
            torch.set_grad_enabled(self.training),
        ):
            if self.wavlm_model.cfg.normalize:
                wav = torch.nn.functional.layer_norm(wav, wav.shape)

            # 3. （訓練時）部分或全部凍結 encoder 
            with torch.set_grad_enabled(
                self.training and not self.freeze_encoder
            ):
                (_, all_outputs), embed_padding_mask = (
                    self.wavlm_model.extract_features(
                        source=wav,
                        padding_mask=padding_mask,
                        output_layer=self.nth_layer,
                        ret_layer_results=True,
                    )
                )
            all_outputs = torch.stack([gg[0] for gg in all_outputs], dim=-1)  # T B C L

            preds = {}
            for name in self.axes_name:
                # 4. 層輸出加權或直接取最後一層
                if self.use_weighted_layer_sum:
                    norm_weights = torch.nn.functional.softmax(
                        self.layer_weights[name], dim=-1
                    )
                    audio_embed = torch.einsum("tbcl,l->btc", all_outputs, norm_weights)
                else:
                    audio_embed = all_outputs[-1][0].transpose(1, 0)

                # 5. 時間維度做 mask 加權平均
                embed_mask = (
                    (~embed_padding_mask).unsqueeze(dim=-1).type_as(audio_embed)
                )

                # 在 loop 中知道 name，假設有個 mapping:
                head_idx = self.axes_name.index(name)  # CE→0, CU→1, PC→2, PQ→3

                audio_embed = self.pooling(audio_embed, embed_mask)
                audio_embed = self.pooling_proj(audio_embed)

                # 6. Embedding 正規化（若需要）
                if self.normalize_embed:
                    audio_embed = torch.nn.functional.normalize(
                        audio_embed, dim=-1
                    )
                # 7. 經 MLP 投射頭得到分數，去除最後的維度
                preds[name] = self.proj_layer[name](audio_embed).squeeze(-1)
        return preds       # 回傳 dict: {"CE": Tensor[B], "CU":…, "PC":…, "PQ":…}
