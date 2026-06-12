import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.siglip import SigLip
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import NestedTensor

logger = get_logger()

__all__ = ["BackboneSIGLIP"]


class BackboneSIGLIP(BackboneBase):
    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,
        drop_path=0.0,
        out_channels=256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple[int, int] = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 16,
        num_windows: int = 1,
        positional_encoding_size: int = 0,
    ):
        super().__init__()
        name_parts = name.split("_")
        assert name_parts[0] == "siglip" and len(name_parts) == 2, "name 应为 siglip_{size}，如 siglip_base"

        if drop_path > 0.0:
            logger.warning("SigLIP backbone 不支持 drop_path，已忽略")

        self.encoder = SigLip(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            model_dir=pretrained_encoder,
            gradient_checkpointing=gradient_checkpointing,
            load_siglip_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=1,
            freeze=freeze_encoder,
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        assert sorted(self.projector_scale) == self.projector_scale, (
            "only support projector scale P3/P4/P5/P6 in ascending order."
        )
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if not hasattr(self.encoder, "merge_and_unload"):
            return

        try:
            from peft import PeftModel
        except ModuleNotFoundError:
            logger.warning("peft is not installed; skipping LoRA weight merging during export.")
            return
        except ImportError as exc:
            logger.warning("Failed to import PeftModel from peft during export: %s", exc)
            raise

        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights")
            self.encoder = self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor):
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_siglip_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_siglip_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_siglip_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name or "patch_embedding" in name:
            layer_id = 0
        elif ".encoder.layers." in name:
            layer_id = int(name.split(".encoder.layers.")[1].split(".")[0]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_siglip_weight_decay_rate(name, weight_decay_rate=1.0):
    if any(k in name for k in ("bias", "norm", "embeddings", "position_embedding")):
        weight_decay_rate = 0.0
    return weight_decay_rate
