import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class FACLLoss(nn.Module):
    """Fourier Amplitude and Correlation Loss for rain-field reconstruction.

    Expected input shapes are [B, T, H, W] or [B, T, C, H, W].
    FFT is applied only over the final spatial dimensions.
    """

    def __init__(
        self,
        total_steps: int,
        alpha: float = 0.1,
        eps: float = 1e-8,
        reduction: str = "official",
        scale_by_sqrt_hw: bool = True,
        value_scale: float = 1.0,
        normalize_by_scale: bool = True,
    ) -> None:
        super().__init__()
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1.")
        if reduction not in {"official", "framewise"}:
            raise ValueError("reduction must be 'official' or 'framewise'.")
        if value_scale <= 0:
            raise ValueError("value_scale must be positive.")

        self.total_steps = int(total_steps)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.reduction = reduction
        self.scale_by_sqrt_hw = bool(scale_by_sqrt_hw)
        self.value_scale = float(value_scale)
        self.normalize_by_scale = bool(normalize_by_scale)

        self.fal_only_steps = int(self.total_steps * self.alpha)
        self.decay_steps = max(self.total_steps - self.fal_only_steps, 1)

    def probability_fcl(self, global_step: int) -> float:
        step = max(int(global_step), 0)
        if step < self.fal_only_steps:
            return 0.0
        if self.decay_steps <= 1:
            return 0.0
        step = step - self.fal_only_steps
        if step >= self.decay_steps - 1:
            return 0.0
        return 1.0 - step / (self.decay_steps - 1)

    def _prepare_field(self, tensor: Tensor) -> Tensor:
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)
        elif tensor.ndim != 5:
            raise ValueError(
                "FACL expects [B, T, H, W] or [B, T, C, H, W]. "
                "Received {}.".format(tuple(tensor.shape))
            )

        tensor = tensor.float()
        if self.normalize_by_scale:
            tensor = torch.clamp(tensor / self.value_scale, 0.0, 1.0)
        return tensor

    @staticmethod
    def fourier_amplitude_loss(fft_pred: Tensor, fft_target: Tensor) -> Tensor:
        return torch.mean((torch.abs(fft_pred) - torch.abs(fft_target)) ** 2)

    def fourier_correlation_loss(self, fft_pred: Tensor, fft_target: Tensor) -> Tensor:
        if self.reduction == "official":
            numerator = (torch.conj(fft_pred) * fft_target).sum().real
            pred_energy = (torch.abs(fft_pred) ** 2).sum()
            target_energy = (torch.abs(fft_target) ** 2).sum()
            denominator = torch.sqrt((pred_energy + self.eps) * (target_energy + self.eps))
            fcl = 1.0 - numerator / denominator
            both_zero = (pred_energy < self.eps) & (target_energy < self.eps)
            return torch.where(both_zero, torch.zeros_like(fcl), fcl)

        numerator = (torch.conj(fft_pred) * fft_target).sum(dim=(-2, -1)).real
        pred_energy = (torch.abs(fft_pred) ** 2).sum(dim=(-2, -1))
        target_energy = (torch.abs(fft_target) ** 2).sum(dim=(-2, -1))
        denominator = torch.sqrt((pred_energy + self.eps) * (target_energy + self.eps))
        fcl = 1.0 - numerator / denominator
        both_zero = (pred_energy < self.eps) & (target_energy < self.eps)
        fcl = torch.where(both_zero, torch.zeros_like(fcl), fcl)
        return fcl.mean()

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        global_step: int = 0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        if pred.shape != target.shape:
            raise ValueError(
                "Shape mismatch: pred={} target={}".format(
                    tuple(pred.shape),
                    tuple(target.shape),
                )
            )

        pred_field = self._prepare_field(pred)
        target_field = self._prepare_field(target)

        fft_pred = torch.fft.fft2(pred_field, dim=(-2, -1), norm="ortho")
        fft_target = torch.fft.fft2(target_field, dim=(-2, -1), norm="ortho")

        fal = self.fourier_amplitude_loss(fft_pred, fft_target)
        fcl = self.fourier_correlation_loss(fft_pred, fft_target)

        p_fcl = self.probability_fcl(global_step)
        use_fal = bool((torch.rand((), device=pred.device) > p_fcl).item())
        loss = fal if use_fal else fcl

        height, width = pred_field.shape[-2:]
        if self.scale_by_sqrt_hw:
            loss = loss * math.sqrt(height * width)

        diagnostics = {
            "loss": loss.detach().item(),
            "fal": fal.detach().item(),
            "fcl": fcl.detach().item(),
            "p_fcl": p_fcl,
            "p_fal": 1.0 - p_fcl,
            "selected_fal": float(use_fal),
            "selected_fcl": float(not use_fal),
        }
        return loss, diagnostics


def add_facl_args(parser):
    parser.add_argument(
        "--forecast_loss",
        choices=["weighted_l1", "facl"],
        default="weighted_l1",
        help="Loss for the final predicted rain field.",
    )
    parser.add_argument("--facl_alpha", type=float, default=0.1)
    parser.add_argument("--facl_eps", type=float, default=1e-8)
    parser.add_argument(
        "--facl_reduction",
        choices=["official", "framewise"],
        default="official",
    )
    parser.add_argument("--no_facl_sqrt_hw", action="store_true")
    parser.add_argument("--no_facl_normalize", action="store_true")
    return parser


def build_facl_loss(args, total_steps: int) -> Optional[FACLLoss]:
    if getattr(args, "forecast_loss", "weighted_l1") != "facl":
        return None
    return FACLLoss(
        total_steps=total_steps,
        alpha=getattr(args, "facl_alpha", 0.1),
        eps=getattr(args, "facl_eps", 1e-8),
        reduction=getattr(args, "facl_reduction", "official"),
        scale_by_sqrt_hw=not getattr(args, "no_facl_sqrt_hw", False),
        value_scale=max(float(getattr(args, "intensity_scale", 1.0)), 1e-6),
        normalize_by_scale=not getattr(args, "no_facl_normalize", False),
    )


def compute_forecast_reconstruction_loss(
    pred: Tensor,
    target: Tensor,
    args,
    weighted_l1_fn,
    facl_criterion: Optional[FACLLoss] = None,
    global_step: int = 0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    if getattr(args, "forecast_loss", "weighted_l1") != "facl":
        return weighted_l1_fn(pred, target, args.intensity_scale), {}

    if facl_criterion is None:
        raise ValueError("facl_criterion is required when --forecast_loss facl.")

    facl_loss, facl_log = facl_criterion(pred, target, global_step)
    logs = {
        "forecast_weighted_l1": weighted_l1_fn(pred, target, args.intensity_scale).detach(),
    }
    for key, value in facl_log.items():
        logs["facl_{}".format(key)] = pred.new_tensor(value)
    return facl_loss, logs
