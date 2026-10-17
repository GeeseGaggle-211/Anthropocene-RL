"""model9: typed entity-set encoder, service-level actor heads and a factored two-hot critic.

Actor and critic do not share parameters (the critic may later receive privileged inputs that must never reach
the actor). Inputs are normalized as symlog(x) followed by a fixed TRAIN-fitted affine; missing rows are excluded
from pooling, so missing and zero never look alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import contracts

LOGIT_EPS = 1e-3
MASKED = -30.0  # finite logit for actions the native mask rules out
BINS = 255
BIN_LOW, BIN_HIGH = -2.0, 9.0  # symlog(u): u in roughly [-6, 8100] welfare-years per resident


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


@dataclass
class Normalizer:
    core_mean: np.ndarray
    core_std: np.ndarray
    set_mean: dict[str, np.ndarray] = field(default_factory=dict)
    set_std: dict[str, np.ndarray] = field(default_factory=dict)

    @staticmethod
    def fit(batch: dict[str, np.ndarray]) -> "Normalizer":
        def moments(x: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x = np.sign(x) * np.log1p(np.abs(x))
            w = present.astype(np.float64)
            n = np.maximum(1.0, w.sum(axis=0))
            mean = (x * w).sum(axis=0) / n
            var = (((x - mean) ** 2) * w).sum(axis=0) / n
            return mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-3).astype(np.float32)

        core_mean, core_std = moments(batch["core"], batch["core_present"])
        normalizer = Normalizer(core_mean, core_std)
        for name, rows, width in contracts.SETS:
            x = batch[name].reshape(-1, width)
            present = np.repeat(batch[name + "_present"].reshape(-1, 1), width, axis=1)
            normalizer.set_mean[name], normalizer.set_std[name] = moments(x, present)
        return normalizer

    def state_dict(self) -> dict[str, list[float]]:
        out = {"core_mean": self.core_mean.tolist(), "core_std": self.core_std.tolist()}
        for name in self.set_mean:
            out[name + "_mean"] = self.set_mean[name].tolist()
            out[name + "_std"] = self.set_std[name].tolist()
        return out

    @staticmethod
    def from_state(state: dict[str, list[float]]) -> "Normalizer":
        n = Normalizer(np.asarray(state["core_mean"], np.float32), np.asarray(state["core_std"], np.float32))
        for name, _, _ in contracts.SETS:
            n.set_mean[name] = np.asarray(state[name + "_mean"], np.float32)
            n.set_std[name] = np.asarray(state[name + "_std"], np.float32)
        return n


def to_tensors(batch: dict[str, np.ndarray], normalizer: Normalizer, device: str = "cpu") -> dict[str, torch.Tensor]:
    def norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return ((np.sign(x) * np.log1p(np.abs(x))) - mean) / std

    present = batch["core_present"].astype(np.float32)
    core = norm(batch["core"], normalizer.core_mean, normalizer.core_std) * present
    out = {"core": torch.as_tensor(np.concatenate([core, present], axis=1), dtype=torch.float32, device=device)}
    for name, _, _ in contracts.SETS:
        x = norm(batch[name], normalizer.set_mean[name], normalizer.set_std[name])
        mask = batch[name + "_present"]
        x = x * mask[..., None]
        out[name] = torch.as_tensor(x, dtype=torch.float32, device=device)
        out[name + "_present"] = torch.as_tensor(mask, dtype=torch.bool, device=device)
    for key in ("movement_mask", "clothing_route_mask", "shelter_route_mask"):
        out[key] = torch.as_tensor(batch[key], dtype=torch.bool, device=device)
    return out


class SetEncoder(nn.Module):
    """One small MLP applied to every row, then masked sum, mean and max pooling."""

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        self.row = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.hidden = hidden

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.row(x)
        mask = present.unsqueeze(-1).float()
        rows = rows * mask
        count = mask.sum(dim=1).clamp(min=1.0)
        total = rows.sum(dim=1)
        mean = total / count
        maximum = rows.masked_fill(~present.unsqueeze(-1), -1e4).max(dim=1).values
        maximum = torch.where(present.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        any_row = present.any(dim=1, keepdim=True).float()
        return torch.cat([total / 8.0, mean, maximum, any_row], dim=1), rows


class Encoder(nn.Module):
    def __init__(self, core_width: int, width: int = 256, set_hidden: int = 64) -> None:
        super().__init__()
        self.sets = nn.ModuleDict({name: SetEncoder(w, set_hidden) for name, _, w in contracts.SETS})
        self.core = nn.Sequential(nn.Linear(core_width * 2, width), nn.SiLU(), nn.LayerNorm(width))
        pooled = len(contracts.SETS) * (3 * set_hidden + 1)
        self.trunk = nn.Sequential(
            nn.Linear(width + pooled, width), nn.SiLU(), nn.LayerNorm(width),
            nn.Linear(width, width), nn.SiLU(), nn.LayerNorm(width),
        )
        self.width, self.set_hidden = width, set_hidden

    def forward(self, t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled, rows = [], {}
        for name, _, _ in contracts.SETS:
            p, r = self.sets[name](t[name], t[name + "_present"])
            pooled.append(p)
            rows[name] = r
        h = self.trunk(torch.cat([self.core(t["core"])] + pooled, dim=1))
        return h, rows


class Actor(nn.Module):
    """Logistic-normal service levels and strategic controls, masked categorical routes and fission policy, and a
    pointer over site rows for movement (stay is slot 0)."""

    def __init__(self, core_width: int, width: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder(core_width, width)
        self.strategic = nn.Linear(width, 2 * len(contracts.STRATEGIC))
        self.service = nn.Linear(width, 2 * len(contracts.SERVICES))
        self.clothing = nn.Linear(width, 2)
        self.shelter = nn.Linear(width, 2)
        self.fission = nn.Linear(width, contracts.FISSION_POLICIES)
        self.stay = nn.Linear(width, 1)
        self.site_query = nn.Linear(width, self.encoder.set_hidden)

    def forward(self, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h, rows = self.encoder(t)
        strategic = self.strategic(h).view(-1, len(contracts.STRATEGIC), 2)
        service = self.service(h).view(-1, len(contracts.SERVICES), 2)
        site_scores = torch.einsum("bh,brh->br", self.site_query(h), rows["sites"]) / np.sqrt(self.encoder.set_hidden)
        movement = torch.cat([self.stay(h), site_scores], dim=1).masked_fill(~t["movement_mask"], MASKED)
        return {
            "strategic_mean": strategic[..., 0], "strategic_log_std": strategic[..., 1].clamp(-5, 1),
            "service_mean": service[..., 0], "service_log_std": service[..., 1].clamp(-5, 1),
            "clothing": self.clothing(h).masked_fill(~t["clothing_route_mask"], MASKED),
            "shelter": self.shelter(h).masked_fill(~t["shelter_route_mask"], MASKED),
            "fission": self.fission(h),
            "movement": movement,
        }


class Critic(nn.Module):
    """Predicts u = return per opening resident as a two-hot distribution over symlog-spaced bins."""

    def __init__(self, core_width: int, width: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder(core_width, width)
        self.head = nn.Linear(width, BINS)
        self.register_buffer("bins", torch.linspace(BIN_LOW, BIN_HIGH, BINS))

    def forward(self, t: dict[str, torch.Tensor]) -> torch.Tensor:
        h, _ = self.encoder(t)
        return self.head(h)

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        # Expectation in raw units avoids the downward bias of inverting a mean in symlog space.
        return (torch.softmax(logits, dim=-1) * symexp(self.bins)).sum(dim=-1)

    def two_hot(self, u: torch.Tensor) -> torch.Tensor:
        y = symlog(u).clamp(BIN_LOW, BIN_HIGH)
        step = (BIN_HIGH - BIN_LOW) / (BINS - 1)
        position = (y - BIN_LOW) / step
        low = position.floor().long().clamp(0, BINS - 2)
        upper_weight = (position - low.float()).clamp(0, 1)
        target = torch.zeros(u.shape[0], BINS, device=u.device)
        target.scatter_(1, low.unsqueeze(1), (1 - upper_weight).unsqueeze(1))
        target.scatter_add_(1, (low + 1).unsqueeze(1), upper_weight.unsqueeze(1))
        return target


def logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(LOGIT_EPS, 1 - LOGIT_EPS)
    return torch.log(p) - torch.log1p(-p)


def logistic_normal_nll(mean: torch.Tensor, log_std: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    z = logit(label)
    return 0.5 * ((z - mean) / log_std.exp()) ** 2 + log_std


def act_deterministic(out: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """The mode used for closed-loop evaluation of the cloned policy (levels snap to the unit-interval ends)."""
    def level(mean: torch.Tensor) -> np.ndarray:
        p = torch.sigmoid(mean).detach().cpu().numpy()
        p[p < 0.01] = 0.0
        p[p > 0.99] = 1.0
        return p

    return {
        "strategic": level(out["strategic_mean"]),
        "service": level(out["service_mean"]),
        "clothing": out["clothing"].argmax(dim=1).cpu().numpy(),
        "shelter": out["shelter"].argmax(dim=1).cpu().numpy(),
        "fission": out["fission"].argmax(dim=1).cpu().numpy(),
        "movement": out["movement"].argmax(dim=1).cpu().numpy() - 1,
    }
