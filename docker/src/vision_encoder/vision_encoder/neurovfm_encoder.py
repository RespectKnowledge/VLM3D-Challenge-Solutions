import json
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neurovfm.models import get_vit_backbone


class NeuroVFMEncoder(nn.Module):
    """
    NeuroVFM adapter for MR-RATE.

    Input:
        x: [B, 1, D, H, W]

    Output:
        [B, N_tokens, 768]
    """

    def __init__(
        self,
        checkpoint_path: str,
        freeze_backbone: bool = True,
        remove_background: bool = False,
    ):
        super().__init__()

        checkpoint_path = Path(checkpoint_path)
        config_path = checkpoint_path / "config.json"
        weights_path = checkpoint_path / "pytorch_model.bin"

        with open(config_path, "r") as f:
            config = json.load(f)

        self.model = get_vit_backbone(**config)

        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = self.model.load_state_dict(
            state_dict,
            strict=False,
        )

        self.output_dim = int(config["params"]["embed_dim"])

        patch_cfg = config["params"]["embed_layer_cf"]["params"]
        self.patch_d = int(patch_cfg["patch_d_size"])
        self.patch_hw = int(patch_cfg["patch_hw_size"])
        self.remove_background = remove_background

        # NeuroVFM MRI normalization
        self.register_buffer(
            "mri_mean",
            torch.tensor(0.3141, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "mri_std",
            torch.tensor(0.2623, dtype=torch.float32),
            persistent=False,
        )

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        print(f"[NeuroVFM] checkpoint: {weights_path}")
        print(f"[NeuroVFM] missing keys: {len(missing)}")
        print(f"[NeuroVFM] unexpected keys: {len(unexpected)}")
        print(f"[NeuroVFM] output_dim: {self.output_dim}")
        print(
            f"[NeuroVFM] patch: "
            f"({self.patch_d},{self.patch_hw},{self.patch_hw})"
        )

    def _tokenize(self, x):
        # x: [1,D,H,W]
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(
                f"Expected [1,D,H,W], got {tuple(x.shape)}"
            )

        _, D, H, W = x.shape

        pd = self.patch_d
        ph = self.patch_hw
        pw = self.patch_hw

        if D % pd != 0 or H % ph != 0 or W % pw != 0:
            raise ValueError(
                f"Volume {tuple(x.shape)} not divisible by "
                f"patch ({pd},{ph},{pw})"
            )

        # MR-RATE percentile cache is [-1,1].
        # NeuroVFM MRI input range is [0,1].
        x = ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)

        # NeuroVFM-specific input adaptation.
        # We reuse the exact same MR-RATE cached volumes but convert the
        # spatial grid to a tractable NeuroVFM token grid.
        x = F.interpolate(
            x.unsqueeze(0),
            size=(64, 192, 192),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        _, D, H, W = x.shape

        nd = D // pd
        nh = H // ph
        nw = W // pw

        tokens = rearrange(
            x,
            "c (d pd) (h ph) (w pw) -> (d h w) (c pd ph pw)",
            d=nd,
            h=nh,
            w=nw,
            pd=pd,
            ph=ph,
            pw=pw,
        )

        # NeuroVFM MRI foreground rule
        threshold = torch.tensor(
            np.percentile(x.detach().cpu().numpy(), 10),
            device=x.device,
            dtype=x.dtype,
        )
        foreground = x[0] > threshold

        mask_tokens = rearrange(
            foreground,
            "(d pd) (h ph) (w pw) -> (d h w) (pd ph pw)",
            d=nd,
            h=nh,
            w=nw,
            pd=pd,
            ph=ph,
            pw=pw,
        )

        # NeuroVFM marks patch foreground only if all voxels are foreground
        fg_mask = mask_tokens.all(dim=1)

        cd = torch.arange(nd, device=x.device)
        ch = torch.arange(nh, device=x.device)
        cw = torch.arange(nw, device=x.device)

        grid = torch.meshgrid(cd, ch, cw, indexing="ij")
        coords = torch.stack(grid, dim=-1).reshape(-1, 3).long()

        if self.remove_background:
            tokens = tokens[fg_mask]
            coords = coords[fg_mask]

        if tokens.shape[0] == 0:
            raise RuntimeError(
                "NeuroVFM background filtering produced zero tokens."
            )

        # NeuroVFM MRI token normalization
        tokens = (tokens - self.mri_mean) / self.mri_std

        return tokens, coords

    def _encode_single(self, x):
        tokens, coords = self._tokenize(x)

        n_tokens = tokens.shape[0]

        cu_seqlens = torch.tensor(
            [0, n_tokens],
            dtype=torch.int32,
            device=tokens.device,
        )

        output = self.model(
            tokens,
            coords,
            masks=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=n_tokens,
            use_flash_attn=False,
        )

        if output.ndim != 2:
            raise RuntimeError(
                f"Expected [N,D], got {tuple(output.shape)}"
            )

        return output

    def forward(self, x, return_encoded_tokens=True):
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B,1,D,H,W], got {tuple(x.shape)}"
            )

        outputs = [
            self._encode_single(x[b])
            for b in range(x.shape[0])
        ]

        if len(outputs) == 1:
            return outputs[0].unsqueeze(0)

        lengths = [o.shape[0] for o in outputs]

        if len(set(lengths)) != 1:
            raise RuntimeError(
                f"Variable token counts within batch: {lengths}. "
                "MR-RATE baseline normally uses batch_size=1."
            )

        return torch.stack(outputs, dim=0)
