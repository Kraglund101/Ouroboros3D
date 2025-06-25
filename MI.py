"""
MI.py – Mutual-information predictor + noise helper
Compatible with Python ≥ 3.8  (uses typing.Union instead of PEP-604 A | B)
"""

from typing import Optional, Union
import torch
import torch.nn as nn
from diffusers.models import UNet2DModel


# ──────────────────────────────  Constants  ──────────────────────────────── #
EPS: float = 1e-4               # ### NEW – safe margin for clamping


# ──────────────────────────────  Noise generator  ────────────────────────── #
@torch.no_grad()
def generate_noise(
    shape,
    noise_mode: str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    p: Union[torch.Tensor, float, None] = None,
):
    """
    shape      : (B, M, C, H, W)
    noise_mode : 'independent' | 'shared' | 'interpolated'
    p          : mixing coefficient in (0,1) for 'interpolated' mode
    """
    B, M, C, H, W = shape
    device = device or "cpu"

    # allow passing a single float
    if isinstance(p, float):
        p = torch.full((B, M), p, device=device, dtype=dtype)

    if noise_mode == "independent":
        return torch.randn(B, M, C, H, W, device=device, dtype=dtype)

    if noise_mode == "shared":
        shared = torch.randn(C, H, W, device=device, dtype=dtype)
        return shared.unsqueeze(0).unsqueeze(0).expand(B, M, C, H, W)

    if noise_mode == "interpolated":
        if p is None:
            raise ValueError("`p` must be provided for interpolated mode")

        # --- safety ------------------------------------------------------- #
        p = p.detach()                                     # ### NEW
        p = torch.clamp(p, EPS, 1.0 - EPS)                 # ### NEW
        # ------------------------------------------------------------------ #

        shared = torch.randn(C, H, W, device=device, dtype=dtype)
        shared = shared.unsqueeze(0).unsqueeze(0).expand(B, M, C, H, W)
        iid    = torch.randn(B, M, C, H, W, device=device, dtype=dtype)

        while p.dim() < 5:                                 # make p broadcastable
            p = p.unsqueeze(-1)

        return p.sqrt() * shared + (1 - p).sqrt() * iid

    raise ValueError(f"Unknown noise_mode: {noise_mode}")


# ─────────────────────  Mutual-information predictor  ────────────────────── #
class MI_predictor(nn.Module):
    """
    Streams leave-one-out views so no (B·M)×(M-1) tensor is ever allocated.
    Returns an MI score p_k in (0,1) for every view.
    """

    def __init__(
        self,
        backbone: UNet2DModel,
        avg_MI_val: float,
        *,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        # load pretrained backbone weights
        self.MI_model = backbone.to(dtype=dtype)
        ckpt = torch.load(
            "pretrain/MI/final_best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.MI_model.load_state_dict(ckpt["model_state_dict"])
        self.MI_model.eval().requires_grad_(False)

        self.avg_MI_val = avg_MI_val
        self.dtype = dtype

        # 0-byte buffer follows the module to whatever device Lightning chooses
        self.register_buffer("_dummy", torch.empty(0))

    # --------------------------------------------------------------------- #
    @staticmethod
    def _nmi(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8):
        """Normalised MSE: 1 – residual_variance / target_variance."""
        res_var = ((target - pred) - (target - pred).mean((1, 2, 3), True)).pow(2).mean((1, 2, 3))
        tar_var = (target - target.mean((1, 2, 3), True)).pow(2).mean((1, 2, 3))
        return 1.0 - res_var / (tar_var + eps)

    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents : (B, M, C, H, W)  – any dtype/device
        returns : (B, M)           – MI per view, detached & clamped
        """
        B, M, C, H, W = latents.shape
        device = latents.device                   # current runtime device

        latents = latents.to(device=device, dtype=self.dtype, non_blocking=True)
        t0 = torch.zeros(B, device=device, dtype=torch.long)

        mi_per_view = []
        with torch.cuda.amp.autocast(dtype=self.dtype):
            for tgt in range(M):
                context = latents[:, torch.arange(M, device=device) != tgt]   # (B, M-1, C, H, W)
                inp = context.reshape(B, (M - 1) * C, H, W)                  # (B, (M-1)·C, H, W)
                pred = self.MI_model(inp, t0).sample                         # (B, C, H, W)
                mi   = self._nmi(latents[:, tgt], pred)                      # (B,)
                mi_per_view.append(mi.detach())
                del context, inp, pred, mi                                   # free Python refs

        p = torch.stack(mi_per_view, dim=1)                                  # (B, M)

        # --- safety ------------------------------------------------------- #
        p = torch.clamp(p, EPS, 1.0 - EPS)            # ### NEW
        # ------------------------------------------------------------------ #
        return p
