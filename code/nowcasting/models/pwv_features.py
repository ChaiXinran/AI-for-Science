import torch
import torch.nn.functional as F


def parse_tendency_windows(value):
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [float(part) for part in parts if part]
    return [float(part) for part in value]


def tendency_modes(mode):
    mode = str(mode or "slope").strip().lower()
    if mode in ("", "none", "off"):
        return []
    if mode == "both":
        return ["diff", "slope"]
    if mode in ("diff", "slope"):
        return [mode]
    raise ValueError("Unknown pwv_tendency_mode: {}".format(mode))


def pwv_feature_group_count(configs):
    windows = parse_tendency_windows(getattr(configs, "pwv_tendency_windows", ""))
    return 4 + len(windows) * len(tendency_modes(getattr(configs, "pwv_tendency_mode", "slope")))


def _window_steps(window_minutes, frame_minutes):
    frame_minutes = max(float(frame_minutes), 1e-6)
    return max(1, int(round(float(window_minutes) / frame_minutes)))


def window_diff_rate(pwv_input, window_minutes, frame_minutes):
    batch, length, height, width = pwv_input.shape
    steps = _window_steps(window_minutes, frame_minutes)
    frame_hours = max(float(frame_minutes) / 60.0, 1e-6)
    rates = []
    zero = pwv_input.new_zeros(batch, height, width)
    for end in range(length):
        start = max(0, end - steps)
        if start == end:
            rates.append(zero)
            continue
        elapsed_hours = max((end - start) * frame_hours, frame_hours)
        rates.append((pwv_input[:, end] - pwv_input[:, start]) / elapsed_hours)
    return torch.stack(rates, dim=1)


def window_linear_slope(pwv_input, window_minutes, frame_minutes):
    batch, length, height, width = pwv_input.shape
    steps = _window_steps(window_minutes, frame_minutes)
    frame_hours = max(float(frame_minutes) / 60.0, 1e-6)
    slopes = []
    zero = pwv_input.new_zeros(batch, height, width)
    for end in range(length):
        start = max(0, end - steps)
        segment = pwv_input[:, start : end + 1]
        seg_len = segment.shape[1]
        if seg_len <= 1:
            slopes.append(zero)
            continue
        tau = torch.arange(seg_len, device=pwv_input.device, dtype=pwv_input.dtype) * frame_hours
        tau = tau - tau.mean()
        denom = tau.pow(2).sum().clamp_min(1e-6)
        centered = segment - segment.mean(dim=1, keepdim=True)
        slopes.append((centered * tau.view(1, seg_len, 1, 1)).sum(dim=1) / denom)
    return torch.stack(slopes, dim=1)


def pwv_tendency_maps(pwv_input, frame_minutes=6.0, windows=None, mode="slope"):
    windows = parse_tendency_windows(windows)
    modes = tendency_modes(mode)
    maps = []
    for window in windows:
        if "diff" in modes:
            maps.append(window_diff_rate(pwv_input, window, frame_minutes))
        if "slope" in modes:
            maps.append(window_linear_slope(pwv_input, window, frame_minutes))
    return maps


def base_pwv_features(pwv_input):
    mean = pwv_input.mean(dim=(1, 2, 3), keepdim=True)
    std = pwv_input.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    value = (pwv_input - mean) / std
    anomaly = pwv_input - pwv_input.mean(dim=1, keepdim=True)
    delta = torch.zeros_like(pwv_input)
    delta[:, 1:] = pwv_input[:, 1:] - pwv_input[:, :-1]
    dx = F.pad(pwv_input[..., :, 1:] - pwv_input[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(pwv_input[..., 1:, :] - pwv_input[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx * dx + dy * dy + 1e-6)
    return [value, anomaly, delta, gradient]


def build_base_pwv_features(pwv_input):
    return torch.clamp(torch.cat(base_pwv_features(pwv_input), dim=1), -5.0, 5.0)


def build_pwv_features(pwv_input, configs):
    frame_minutes = getattr(configs, "frame_minutes", 6.0)
    windows = getattr(configs, "pwv_tendency_windows", "")
    mode = getattr(configs, "pwv_tendency_mode", "slope")
    features = base_pwv_features(pwv_input)
    features.extend(pwv_tendency_maps(pwv_input, frame_minutes, windows, mode))
    return torch.clamp(torch.cat(features, dim=1), -5.0, 5.0)


def positive_growth_signal(pwv_input, configs):
    frame_minutes = getattr(configs, "frame_minutes", 6.0)
    windows = parse_tendency_windows(getattr(configs, "pwv_tendency_windows", ""))
    if windows:
        maps = pwv_tendency_maps(
            pwv_input,
            frame_minutes,
            windows,
            getattr(configs, "pwv_tendency_mode", "slope"),
        )
        if maps:
            return torch.stack([F.relu(item[:, -1]) for item in maps], dim=1).amax(dim=1)
    return F.relu(pwv_input[:, -1] - pwv_input[:, 0])
