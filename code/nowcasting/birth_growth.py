import torch
import torch.nn.functional as F


def apply_pwv_control(pwv, mode, input_length=None):
    """Apply a PWV control without exposing forecast-period PWV to the model.

    Controls that manipulate time or space operate only on the observed input
    prefix.  The unused suffix is left untouched so diagnostics cannot leak
    future PWV into the first ``input_length`` frames consumed by the model.
    """
    if mode == "real":
        return pwv
    if mode == "zero":
        return torch.zeros_like(pwv)
    observed_length = pwv.size(1) if input_length is None else int(input_length)
    if observed_length <= 0 or observed_length > pwv.size(1):
        raise ValueError("input_length must be in [1, {}]".format(pwv.size(1)))
    observed = pwv[:, :observed_length]
    controlled = pwv.clone()
    if mode == "temporal_reverse":
        controlled[:, :observed_length] = torch.flip(observed, dims=[1])
        return controlled
    if mode == "level_only":
        observed_mean = observed.mean(dim=1, keepdim=True)
        controlled[:, :observed_length] = observed_mean.expand_as(observed)
        return controlled
    if mode == "spatial_shift":
        # A fixed half-domain cyclic displacement preserves each sample's PWV
        # distribution, spatial texture, and temporal evolution while breaking
        # radar/PWV geographical co-location without interpolation artifacts.
        shifts = (pwv.size(-2) // 2, pwv.size(-1) // 2)
        controlled[:, :observed_length] = torch.roll(
            observed, shifts=shifts, dims=(-2, -1)
        )
        return controlled
    raise ValueError("Unknown PWV control mode: {}".format(mode))


def birth_growth_targets(radar_evolution, target, low_threshold, high_threshold, growth_delta):
    """Build explicit targets relative to the radar-only evolution baseline."""
    baseline = radar_evolution.detach()
    residual = target - baseline
    positive_source = F.relu(residual)
    birth = ((baseline < low_threshold) & (target >= high_threshold)).float()
    growth = ((baseline >= low_threshold) & (residual > growth_delta)).float()
    return {
        "birth": birth,
        "growth": growth,
        "positive_source": positive_source,
    }


def focal_binary_probability_loss(
    probability, target, alpha=0.75, gamma=2.0, normalization="class_balanced"
):
    probability = probability.clamp(1e-5, 1.0 - 1e-5)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    focal = -(1.0 - pt).pow(gamma) * torch.log(pt)
    if normalization == "pixel_mean":
        alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
        return (alpha_t * focal).mean()
    if normalization != "class_balanced":
        raise ValueError("Unknown focal normalization: {}".format(normalization))
    positive = target > 0.5
    negative = ~positive
    zero = focal.sum() * 0.0
    positive_loss = focal[positive].mean() if positive.any() else zero
    negative_loss = focal[negative].mean() if negative.any() else zero
    return alpha * positive_loss + (1.0 - alpha) * negative_loss


def birth_growth_losses(aux, target, args):
    labels = birth_growth_targets(
        aux["radar_evolution"],
        target,
        args.birth_low_threshold,
        args.birth_high_threshold,
        args.growth_delta,
    )
    birth_probability = aux["birth_probability"][:, :, 0]
    growth_probability = aux["growth_probability"][:, :, 0]
    contribution = aux["pwv_contribution"][:, :, 0]
    birth_loss = focal_binary_probability_loss(
        birth_probability, labels["birth"], args.birth_focal_alpha, args.birth_focal_gamma,
        getattr(args, "birth_loss_normalization", "class_balanced"),
    )
    growth_loss = focal_binary_probability_loss(
        growth_probability, labels["growth"], args.birth_focal_alpha, args.birth_focal_gamma,
        getattr(args, "birth_loss_normalization", "class_balanced"),
    )
    active = torch.clamp(labels["birth"] + labels["growth"], 0.0, 1.0)
    active_mask = active > 0.5
    inactive_mask = ~active_mask
    source_error = (contribution - labels["positive_source"]).abs()
    zero = source_error.sum() * 0.0
    source_active_loss = source_error[active_mask].mean() if active_mask.any() else zero
    source_inactive_loss = source_error[inactive_mask].mean() if inactive_mask.any() else zero
    source_loss = source_active_loss + args.source_inactive_weight * source_inactive_loss
    sparse_loss = (contribution * (1.0 - active)).mean()
    total = (
        args.lambda_birth * birth_loss
        + args.lambda_growth * growth_loss
        + args.lambda_positive_source * source_loss
        + args.lambda_source_sparse * sparse_loss
    )
    return total, {
        "birth": birth_loss.detach(),
        "growth": growth_loss.detach(),
        "positive_source": source_loss.detach(),
        "positive_source_active": source_active_loss.detach(),
        "positive_source_inactive": source_inactive_loss.detach(),
        "source_sparse": sparse_loss.detach(),
        "birth_rate": labels["birth"].mean().detach(),
        "growth_rate": labels["growth"].mean().detach(),
    }


class BinaryHistogramAccumulator:
    def __init__(self, bins=200, decision_threshold=0.5):
        self.bins = int(bins)
        self.decision_threshold = float(decision_threshold)
        self.pos_hist = torch.zeros(self.bins, dtype=torch.float64)
        self.neg_hist = torch.zeros(self.bins, dtype=torch.float64)
        self.tp = self.fp = self.fn = self.tn = 0
        self.brier_sum = 0.0
        self.count = 0

    def update(self, probability, target):
        probability = probability.detach().float().cpu().clamp(0.0, 1.0).reshape(-1)
        target = target.detach().bool().cpu().reshape(-1)
        bucket = torch.clamp((probability * self.bins).long(), 0, self.bins - 1)
        self.pos_hist += torch.bincount(bucket[target], minlength=self.bins).double()
        self.neg_hist += torch.bincount(bucket[~target], minlength=self.bins).double()
        prediction = probability >= self.decision_threshold
        self.tp += int((prediction & target).sum())
        self.fp += int((prediction & ~target).sum())
        self.fn += int((~prediction & target).sum())
        self.tn += int((~prediction & ~target).sum())
        self.brier_sum += float(((probability - target.float()) ** 2).sum())
        self.count += target.numel()

    @staticmethod
    def _safe(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else None

    def finalize(self):
        pos_cum = torch.flip(torch.cumsum(torch.flip(self.pos_hist, dims=[0]), dim=0), dims=[0])
        neg_cum = torch.flip(torch.cumsum(torch.flip(self.neg_hist, dims=[0]), dim=0), dims=[0])
        total_pos = float(self.pos_hist.sum())
        precision = pos_cum / (pos_cum + neg_cum).clamp_min(1.0)
        recall = pos_cum / max(total_pos, 1.0)
        recall_asc = torch.cat([torch.zeros(1, dtype=torch.float64), torch.flip(recall, dims=[0])])
        precision_asc = torch.cat([torch.ones(1, dtype=torch.float64), torch.flip(precision, dims=[0])])
        pr_auc = float(torch.trapz(precision_asc, recall_asc)) if total_pos else None
        return {
            "threshold": self.decision_threshold,
            "positives": int(total_pos),
            "count": self.count,
            "positive_rate": self._safe(total_pos, self.count),
            "precision": self._safe(self.tp, self.tp + self.fp),
            "recall_pod": self._safe(self.tp, self.tp + self.fn),
            "false_alarm_ratio": self._safe(self.fp, self.tp + self.fp),
            "csi": self._safe(self.tp, self.tp + self.fp + self.fn),
            "f1": self._safe(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "brier": self._safe(self.brier_sum, self.count),
            "pr_auc_histogram": pr_auc,
            "histogram_bins": self.bins,
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
        }


class BirthGrowthAccumulator:
    def __init__(self, decision_threshold=0.5, bins=200):
        self.birth = BinaryHistogramAccumulator(bins, decision_threshold)
        self.growth = BinaryHistogramAccumulator(bins, decision_threshold)
        self.source_abs_error = 0.0
        self.source_active_abs_error = 0.0
        self.source_inactive_abs_error = 0.0
        self.source_count = 0
        self.source_active_count = 0
        self.source_inactive_count = 0

    def update(self, aux, target, args):
        labels = birth_growth_targets(
            aux["radar_evolution"], target, args.birth_low_threshold,
            args.birth_high_threshold, args.growth_delta
        )
        self.birth.update(aux["birth_probability"][:, :, 0], labels["birth"])
        self.growth.update(aux["growth_probability"][:, :, 0], labels["growth"])
        contribution = aux["pwv_contribution"][:, :, 0].detach()
        error = (contribution - labels["positive_source"]).abs()
        active = (labels["birth"] + labels["growth"]) > 0
        inactive = ~active
        self.source_abs_error += float(error.sum())
        self.source_active_abs_error += float(error[active].sum())
        self.source_inactive_abs_error += float(error[inactive].sum())
        self.source_count += error.numel()
        self.source_active_count += int(active.sum())
        self.source_inactive_count += int(inactive.sum())

    def finalize(self):
        safe = BinaryHistogramAccumulator._safe
        return {
            "birth": self.birth.finalize(),
            "growth": self.growth.finalize(),
            "positive_source_mae": safe(self.source_abs_error, self.source_count),
            "positive_source_mae_active": safe(self.source_active_abs_error, self.source_active_count),
            "positive_source_mae_inactive": safe(self.source_inactive_abs_error, self.source_inactive_count),
        }
