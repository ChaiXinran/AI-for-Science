import torch
import torch.nn as nn


class LeadTimeConditioner(nn.Module):
    """Lightweight lead-time adaptive scale/bias modulation.

    The module maps normalized lead time indices to per-lead affine parameters.
    All output heads are zero-initialized, so enabled models start from the same
    behavior as the unconditioned network and can learn lead-specific changes.
    """

    def __init__(self, pred_length, embed_dim=0, targets=None):
        super(LeadTimeConditioner, self).__init__()
        self.pred_length = int(pred_length)
        self.embed_dim = int(embed_dim or 0)
        if targets is None:
            targets = {"source": 1, "gate": 1, "motion": 2}
        self.targets = dict(targets)

        if self.embed_dim <= 0:
            self.enabled = False
            return

        self.enabled = True
        if self.pred_length > 1:
            positions = torch.linspace(0.0, 1.0, self.pred_length).view(self.pred_length, 1)
        else:
            positions = torch.zeros(self.pred_length, 1)
        self.register_buffer("lead_positions", positions)
        self.encoder = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleDict()
        for name, channels in self.targets.items():
            head = nn.Linear(self.embed_dim, int(channels) * 2)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.heads[name] = head

    def _params(self, name, dtype):
        emb = self.encoder(self.lead_positions.to(dtype=dtype))
        params = self.heads[name](emb)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta

    def forward(self, x, name):
        if not getattr(self, "enabled", False):
            return x
        if name not in self.heads:
            raise KeyError("Unknown lead-time conditioning target: {}".format(name))

        gamma, beta = self._params(name, x.dtype)
        if x.dim() == 4:
            gamma = gamma.view(1, self.pred_length, 1, 1)
            beta = beta.view(1, self.pred_length, 1, 1)
        elif x.dim() == 5:
            channels = x.shape[2]
            gamma = gamma.view(1, self.pred_length, channels, 1, 1)
            beta = beta.view(1, self.pred_length, channels, 1, 1)
        else:
            raise ValueError("Expected 4D or 5D tensor, got shape {}".format(tuple(x.shape)))
        return x * (1.0 + gamma) + beta

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if getattr(self, "enabled", False):
            for key, value in self.state_dict().items():
                full_key = prefix + key
                if full_key not in state_dict:
                    state_dict[full_key] = value.detach().clone()
        super(LeadTimeConditioner, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
