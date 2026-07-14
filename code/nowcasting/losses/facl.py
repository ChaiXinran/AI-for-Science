import torch
import torch.nn.functional as F


def _fft2(x):
    return torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")


def fourier_amplitude_loss(pred, target):
    pred_fft = _fft2(pred)
    target_fft = _fft2(target)
    return F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))


def fourier_correlation_loss(pred, target, eps=1e-8):
    pred_fft = _fft2(pred).flatten(start_dim=2)
    target_fft = _fft2(target).flatten(start_dim=2)
    numerator = torch.real((target_fft * torch.conj(pred_fft)).sum(dim=-1))
    pred_energy = (torch.abs(pred_fft) ** 2).sum(dim=-1)
    target_energy = (torch.abs(target_fft) ** 2).sum(dim=-1)
    correlation = numerator / torch.sqrt(pred_energy * target_energy + eps)
    return (1.0 - correlation).mean()


def fourier_amplitude_and_correlation_loss(pred, target, fal_probability=0.5):
    if torch.rand((), device=pred.device) < fal_probability:
        return fourier_amplitude_loss(pred, target)
    return fourier_correlation_loss(pred, target)
