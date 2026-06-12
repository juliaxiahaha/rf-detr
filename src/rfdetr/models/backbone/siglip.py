import json
import os

import torch
import torch.nn as nn
from transformers import SiglipVisionModel

from rfdetr.utilities.logger import get_logger

logger = get_logger()

size_to_width = {
    "base": 768,
}

size_to_config = {
    "base": "siglip_base.json",
}


def get_config(size: str) -> dict:
    # 读取 SigLIP 结构参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(current_dir, "dinov2_configs")
    config_path = os.path.join(configs_dir, size_to_config[size])
    with open(config_path, "r") as f:
        return json.load(f)


class SigLip(nn.Module):
    def __init__(
        self,
        shape=(640, 640),
        out_feature_indexes=None,
        size="base",
        model_dir=None,
        gradient_checkpointing=False,
        load_siglip_weights=True,
        patch_size=16,
        num_windows=1,
        freeze=False,
    ):
        super().__init__()
        if out_feature_indexes is None:
            out_feature_indexes = [3, 6, 9, 12]

        cfg = get_config(size)
        self.shape = shape
        self.out_feature_indexes = list(out_feature_indexes)
        self._last_feature_index = max(out_feature_indexes)
        self.hidden_size = cfg["hidden_size"]
        self.image_size = cfg["image_size"]
        self.num_windows = 1  # SigLIP 无 window attention

        if patch_size != cfg["patch_size"]:
            logger.warning(
                f"patch_size={patch_size} 与 siglip_base.json 中 {cfg['patch_size']} 不一致，使用构造参数"
            )
        self.patch_size = patch_size

        max_index = max(out_feature_indexes)
        if max_index > cfg["num_hidden_layers"]:
            raise ValueError(
                f"out_feature_indexes 最大为 {max_index}，超过 num_hidden_layers={cfg['num_hidden_layers']}"
            )

        self._out_feature_channels = [self.hidden_size] * len(out_feature_indexes)
        self._export = False

        if model_dir is None:
            raise ValueError("SigLIP 需要 pretrained_encoder 指向本地权重目录")

        if load_siglip_weights:
            # 加载本地 SigLIP 权重（与 JSON 目录分离）
            self.encoder = SiglipVisionModel.from_pretrained(model_dir, local_files_only=True)
        else:
            raise NotImplementedError("SigLIP 暂不支持随机初始化，请提供 pretrained_encoder")

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            # 冻结对比学习头，检测不用
            for p in self.encoder.vision_model.head.parameters():
                p.requires_grad = False

    def export(self):
        if self._export:
            return
        self._export = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, (
            f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        )

        b, _, h, w = x.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        interpolate_pos = h != self.image_size or w != self.image_size

        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos,
        )

        feats: list[torch.Tensor] = []
        for idx in self.out_feature_indexes:
            # 最深层用 last_hidden_state，保留 post_layernorm 梯度
            if idx == self._last_feature_index:
                hs = outputs.last_hidden_state
            else:
                hs = outputs.hidden_states[idx]
            if hs.shape[1] != gh * gw:
                raise RuntimeError(
                    f"SigLIP hidden seq {hs.shape[1]} != {gh}x{gw} for input {h}x{w}"
                )
            # hidden_states -> (B,C,H,W)，与 DinoV2 输出格式一致
            feat = hs.transpose(1, 2).reshape(b, self.hidden_size, gh, gw)
            feats.append(feat)
        return feats


