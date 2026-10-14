"""Paired native interventions and explicit continuous-head reachability bounds."""

import copy

import numpy as np

ALPHA_MIN = float(np.logaddexp(0, -12) + 0.5)
ALPHA_MAX = float(np.logaddexp(0, 12) + 0.5)
CONTROL_MIN = ALPHA_MIN / (ALPHA_MIN + ALPHA_MAX)
CONTROL_MAX = 1 - CONTROL_MIN


def shares(mask, selected=None, *, representable=False):
    mask = np.asarray(mask, dtype=bool)
    result = mask.astype(float)
    if selected is not None:
        if not mask[selected]:
            raise ValueError("Intervention selects an unavailable coordinate")
        result[:] = 0
        if representable:
            result[mask] = ALPHA_MIN
            result[selected] = ALPHA_MAX
        else:
            result[selected] = 1
    if result.sum():
        result /= result.sum()
    return result


def base_action(obs, role=None, *, representable=False):
    role = (1 if obs["WorkforceMask"][1] else 0) if role is None else role
    controls = np.zeros(3)
    if representable:
        controls[obs["ControlMask"]] = CONTROL_MIN
    if obs["ControlMask"][1]:
        controls[1] = CONTROL_MAX if representable else 1
    return dict(
        workforce=shares(obs["WorkforceMask"], role, representable=representable),
        activities=np.stack([shares(m) for m in obs["ActivityMask"]]),
        controls=controls,
        movement=0,
    )


def market_peer(obs, *, ratio=0.5):
    action = base_action(obs)
    partners = np.flatnonzero(obs["CounterpartyMask"])
    if len(partners) and obs["GoodMask"][1] and obs["GoodMask"][0] and obs["Stocks"][1] >= 1e-6:
        q = max(1e-6 / min(1, ratio), min(20, obs["Stocks"][1]))
        action["quote"] = dict(
            Counterparty=int(partners[0]), Give=1, Want=0, GiveQuantity=q, WantQuantity=q * ratio
        )
    return action


def paired_actions(obs, group, index, schema):
    mean = index >= 4
    action = base_action(obs, representable=mean)
    other = copy.deepcopy(action)
    if group in ("workforce", "fishing"):
        r1, r2 = (1, 0) if group == "workforce" else (3, 0)
        action["workforce"] = shares(obs["WorkforceMask"], r1, representable=mean)
        other["workforce"] = shares(obs["WorkforceMask"], r2, representable=mean)
    elif group == "gathering_recipes":
        role = 0 if index % 4 < 2 else 4
        if role == 0:
            first, second = 0, 3
        else:
            first = schema["activities"].index("Pots")
            second = schema["activities"].index("Cloth")
        action = base_action(obs, role, representable=mean)
        other = copy.deepcopy(action)
        action["activities"][role] = shares(obs["ActivityMask"][role], first, representable=mean)
        other["activities"][role] = shares(obs["ActivityMask"][role], second, representable=mean)
    elif group in ("cultivation", "breeding", "retention"):
        role = 1 if group == "cultivation" else 2
        action = base_action(obs, role, representable=mean)
        other = copy.deepcopy(action)
        if group == "retention":
            action["controls"][1] = CONTROL_MAX if mean else 1
            other["controls"][1] = 0.5
        else:
            k = 0 if group == "cultivation" else 2
            action["controls"][k] = CONTROL_MIN if mean else 0
            other["controls"][k] = CONTROL_MAX if mean else 1
    elif group == "movement":
        legal = np.flatnonzero(obs["MovementMask"][1:]) + 1
        if not len(legal):
            raise ValueError("Frozen movement stratum has no legal destination")
        other["movement"] = int(legal[0])
    elif group == "quotes":
        partner = schema["counterparty_ordinal"]
        wood, clay = schema["goods"].index("Wood"), schema["goods"].index("Clay")
        quantity = max(1e-6, min(obs["Stocks"][wood], 5e5) * 0.2) if mean else 10
        other["quote"] = dict(
            Counterparty=partner,
            Give=wood,
            Want=clay,
            GiveQuantity=quantity,
            WantQuantity=quantity * 2,
        )
    else:
        raise ValueError("Unknown intervention group")
    return action, other
