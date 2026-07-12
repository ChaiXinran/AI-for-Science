import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class TemporalDiscriminator(nn.Module):
    """Temporal discriminator for future radar sequences.

    The first stage follows the NowcastNet paper at a practical level: several
    3D convolution branches observe different temporal extents, then their
    features are concatenated and scored by a shared 3D convolutional head.
    """

    def __init__(self, pred_length, base_channels=32, temporal_kernels=None):
        super().__init__()
        self.pred_length = pred_length
        if temporal_kernels is None:
            temporal_kernels = [4, 8, 12, pred_length]
        temporal_kernels = sorted({max(1, min(pred_length, int(k))) for k in temporal_kernels})

        self.branches = nn.ModuleList()
        for kernel_t in temporal_kernels:
            self.branches.append(
                nn.Sequential(
                    spectral_norm(
                        nn.Conv3d(
                            1,
                            base_channels,
                            kernel_size=(kernel_t, 3, 3),
                            stride=(1, 2, 2),
                            padding=(0, 1, 1),
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        merged_channels = base_channels * len(self.branches)
        self.head = nn.Sequential(
            spectral_norm(nn.Conv3d(merged_channels, base_channels * 2, 3, stride=(1, 2, 2), padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(base_channels * 2, base_channels * 4, 3, stride=(1, 2, 2), padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(base_channels * 4, base_channels * 4, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = spectral_norm(nn.Linear(base_channels * 4, 1))

    def forward(self, sequence):
        if sequence.dim() == 4:
            sequence = sequence.unsqueeze(1)
        elif sequence.dim() == 5 and sequence.shape[1] != 1:
            sequence = sequence.permute(0, 4, 1, 2, 3)

        features = []
        for branch in self.branches:
            feat = branch(sequence)
            feat = F.adaptive_avg_pool3d(feat, output_size=(1, feat.shape[-2], feat.shape[-1]))
            features.append(feat)

        x = torch.cat(features, dim=1)
        x = self.head(x)
        x = F.adaptive_avg_pool3d(x, output_size=1).flatten(1)
        return self.classifier(x).squeeze(1)
