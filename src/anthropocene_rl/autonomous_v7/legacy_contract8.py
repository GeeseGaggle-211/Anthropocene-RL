"""Pinned contract-8 actor inference for Gate-1 historical diagnostics.

This is an explicit transcription of the actor/input/decoder surface at commit
``2f4cc895e6bcd897200d340851d69be98fd18e00``. It deliberately does not call
``upgrade`` and does not create
post-contract-8 tensors.  The current bridge may expose appended observation
features/sets; this adapter selects only the reviewed contract-8 prefix and five
named entity sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

PINNED_SOURCE_COMMIT = "2f4cc895e6bcd897200d340851d69be98fd18e00"
MASKED = -30.0
SETS = (
    ("sites", 7, 14),
    ("fishing", 3, 11),
    ("herds", 8, 8),
    ("commodities", 11, 8),
    ("contacts", 8, 9),
)
UNIT_STRATEGIC = 3
ECONOMY_MIX = 3
SERVICES = 6
FISSION_POLICIES = 3


@dataclass
class Contract8Normalizer:
    core_mean: np.ndarray
    core_std: np.ndarray
    set_mean: dict[str, np.ndarray]
    set_std: dict[str, np.ndarray]

    @classmethod
    def from_state(cls, state: dict[str, Any], core_width: int) -> Contract8Normalizer:
        required = {"core_mean", "core_std"} | {
            f"{name}_{suffix}" for name, _, _ in SETS for suffix in ("mean", "std")
        }
        optional = {"privileged_mean", "privileged_std"}
        if set(state) not in (required, required | optional):
            raise ValueError("contract-8 normalizer keys differ from the pinned schema")

        def moment(name: str, *, positive: bool) -> np.ndarray:
            raw = np.asarray(state[name])
            if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
                raise ValueError(f"contract-8 normalizer {name} is not numeric")
            value = np.asarray(state[name], dtype=np.float32)
            if not np.isfinite(value).all() or (positive and not np.all(value > 0)):
                raise ValueError(f"contract-8 normalizer {name} has invalid moments")
            return value

        core_mean = moment("core_mean", positive=False)
        core_std = moment("core_std", positive=True)
        if core_mean.shape != (core_width,) or core_std.shape != (core_width,):
            raise ValueError("contract-8 normalizer core shape differs from the raw checkpoint")
        means = {}
        stds = {}
        for name, _, width in SETS:
            means[name] = moment(name + "_mean", positive=False)
            stds[name] = moment(name + "_std", positive=True)
            if means[name].shape != (width,) or stds[name].shape != (width,):
                raise ValueError(
                    f"contract-8 normalizer {name} shape differs from the pinned schema"
                )
        return cls(core_mean, core_std, means, stds)


class SetEncoder(nn.Module):
    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        self.row = nn.Sequential(
            nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.hidden = hidden

    def forward(
        self, values: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.row(values)
        mask = present.unsqueeze(-1).float()
        rows = rows * mask
        total = rows.sum(dim=1)
        mean = total / mask.sum(dim=1).clamp(min=1.0)
        maximum = rows.masked_fill(~present.unsqueeze(-1), -1e4).max(dim=1).values
        maximum = torch.where(present.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return torch.cat(
            [total / 8.0, mean, maximum, present.any(dim=1, keepdim=True).float()], dim=1
        ), rows


class Encoder(nn.Module):
    def __init__(self, core_width: int, width: int = 256, set_hidden: int = 64) -> None:
        super().__init__()
        self.sets = nn.ModuleDict({name: SetEncoder(size, set_hidden) for name, _, size in SETS})
        self.core = nn.Sequential(nn.Linear(core_width * 2, width), nn.SiLU(), nn.LayerNorm(width))
        pooled = len(SETS) * (3 * set_hidden + 1)
        self.trunk = nn.Sequential(
            nn.Linear(width + pooled, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.set_hidden = set_hidden

    def forward(
        self, tensors: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled = []
        rows = {}
        for name, _, _ in SETS:
            value, row = self.sets[name](tensors[name], tensors[name + "_present"])
            pooled.append(value)
            rows[name] = row
        return self.trunk(torch.cat([self.core(tensors["core"]), *pooled], dim=1)), rows


class Contract8Actor(nn.Module):
    def __init__(self, core_width: int, width: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder(core_width, width)
        self.strategic = nn.Linear(width, 2 * UNIT_STRATEGIC)
        self.mix = nn.Linear(width, 2 * ECONOMY_MIX)
        self.service = nn.Linear(width, 2 * SERVICES)
        self.clothing = nn.Linear(width, 2)
        self.shelter = nn.Linear(width, 2)
        self.fission = nn.Linear(width, FISSION_POLICIES)
        self.stay = nn.Linear(width, 1)
        self.site_query = nn.Linear(width, self.encoder.set_hidden)
        self.commitments = False
        self.region_movement = False
        self.learned_fission = False
        self.learned_trade = False
        self.region_head = "flat"
        self.grouped_region = False
        self.region_selection = None

    def forward(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden, rows = self.encoder(tensors)
        strategic = self.strategic(hidden).view(-1, UNIT_STRATEGIC, 2)
        mix = self.mix(hidden).view(-1, ECONOMY_MIX, 2)
        service = self.service(hidden).view(-1, SERVICES, 2)
        sites = torch.einsum("bh,brh->br", self.site_query(hidden), rows["sites"]) / np.sqrt(
            self.encoder.set_hidden
        )
        movement = torch.cat([self.stay(hidden), sites], dim=1).masked_fill(
            ~tensors["movement_mask"], MASKED
        )
        return {
            "strategic_mean": strategic[..., 0],
            "strategic_log_std": strategic[..., 1].clamp(-5, 1),
            "mix_logits": mix[..., 0].masked_fill(~tensors["economy_mix_mask"], MASKED),
            "mix_log_std": mix[..., 1].clamp(-5, 1),
            "service_mean": service[..., 0],
            "service_log_std": service[..., 1].clamp(-5, 1),
            "clothing": self.clothing(hidden).masked_fill(~tensors["clothing_route_mask"], MASKED),
            "shelter": self.shelter(hidden).masked_fill(~tensors["shelter_route_mask"], MASKED),
            "fission": self.fission(hidden),
            "movement": movement,
        }


def to_tensors(
    batch: dict[str, np.ndarray], normalizer: Contract8Normalizer
) -> dict[str, torch.Tensor]:
    def normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (np.sign(values) * np.log1p(np.abs(values)) - mean) / std

    width = len(normalizer.core_mean)
    core_values = np.asarray(batch["core"], dtype=np.float32)[:, :width]
    core_present = np.asarray(batch["core_present"], dtype=np.float32)[:, :width]
    output = {
        "core": torch.as_tensor(
            np.concatenate(
                [
                    normalize(core_values, normalizer.core_mean, normalizer.core_std)
                    * core_present,
                    core_present,
                ],
                axis=1,
            ),
            dtype=torch.float32,
        )
    }
    for name, _, field_width in SETS:
        values = np.asarray(batch[name], dtype=np.float32)
        if values.shape[1:] != (next(rows for key, rows, _ in SETS if key == name), field_width):
            raise ValueError(f"live {name} observation differs from pinned contract-8 shape")
        present = np.asarray(batch[name + "_present"], dtype=bool)
        normalized = (
            normalize(values, normalizer.set_mean[name], normalizer.set_std[name])
            * present[..., None]
        )
        output[name] = torch.as_tensor(normalized, dtype=torch.float32)
        output[name + "_present"] = torch.as_tensor(present, dtype=torch.bool)
    for key in ("movement_mask", "clothing_route_mask", "shelter_route_mask"):
        output[key] = torch.as_tensor(batch[key], dtype=torch.bool)
    mask = batch.get("economy_mix_mask", np.ones((len(core_values), ECONOMY_MIX), bool))
    output["economy_mix_mask"] = torch.as_tensor(mask, dtype=torch.bool)
    return output


def load(state: dict[str, Any]) -> tuple[Contract8Actor, Contract8Normalizer]:
    if int(state.get("contract", 8)) != 8:
        raise ValueError("pinned adapter accepts only raw contract-8 checkpoints")
    core_width = state.get("core_width")
    if isinstance(core_width, bool) or not isinstance(core_width, int) or core_width <= 0:
        raise ValueError("contract-8 checkpoint lacks an exact core width")
    actor_state = state.get("actor")
    if not isinstance(actor_state, dict):
        raise ValueError("contract-8 checkpoint lacks an actor state dict")
    actor = Contract8Actor(core_width)
    expected = set(actor.state_dict())
    actual = set(actor_state)
    if expected != actual:
        raise ValueError(
            f"contract-8 raw actor keys differ (missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)})"
        )
    expected_state = actor.state_dict()
    for name in sorted(expected):
        value, reference = actor_state[name], expected_state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != reference.dtype
            or value.shape != reference.shape
            or value.layout != torch.strided
        ):
            raise ValueError(
                f"contract-8 raw actor tensor {name} differs in dtype/device/shape/layout"
            )
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    return actor, Contract8Normalizer.from_state(state["normalizer"], core_width)


def deterministic_mode(output: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    def level(mean: torch.Tensor) -> np.ndarray:
        value = torch.sigmoid(mean).detach().cpu().numpy()
        value[value < 0.01] = 0.0
        value[value > 0.99] = 1.0
        return value

    mix = torch.softmax(output["mix_logits"], dim=1).detach().cpu().numpy()
    mix[mix < 0.01] = 0.0
    mix /= np.maximum(mix.sum(axis=1, keepdims=True), 1e-12)
    rest = mix[:, 0] + mix[:, 1]
    field = np.where(rest > 1e-9, mix[:, 0] / np.maximum(rest, 1e-12), 0.5)
    return {
        "strategic": np.concatenate(
            [level(output["strategic_mean"]), field[:, None], mix[:, 2, None]], axis=1
        ),
        "mix": mix,
        "service": level(output["service_mean"]),
        "clothing": output["clothing"].argmax(dim=1).cpu().numpy(),
        "shelter": output["shelter"].argmax(dim=1).cpu().numpy(),
        "fission": output["fission"].argmax(dim=1).cpu().numpy(),
        "movement": output["movement"].argmax(dim=1).cpu().numpy() - 1,
    }
