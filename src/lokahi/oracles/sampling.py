"""Reference sampler: mlx_lm filter semantics with Philox4x32-10 Gumbel-max draws.

Mirrors ``native/engine/src/sampling.h`` independently, in float64:

* top-p, min-p and top-k filter the untempered distribution, each keeping
  every token at or above its threshold logit (boundary ties are kept);
* the token is ``argmax`` over kept ``i`` of ``logit_i / T + G(seed, step, i)``
  where ``G`` is Gumbel noise from Philox4x32-10 keyed by the seed, with the
  counter ``(i // 4, step, STREAM, 0)`` and output lane ``i % 4``.

``sample`` also reports how close the decision was, so tests compare exact
tokens only where float32 rounding in the engine cannot flip the result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_M0, _M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_MASK = np.uint64(0xFFFFFFFF)
STREAM = 0x6C6F6B61


def philox4x32_10(counters: np.ndarray, key: tuple[int, int]) -> np.ndarray:
    """Philox4x32-10 over ``[n, 4]`` uint32 counters with one 64-bit key."""

    c = np.asarray(counters, dtype=np.uint64).reshape(-1, 4)
    k0, k1 = key[0] & 0xFFFFFFFF, key[1] & 0xFFFFFFFF
    for round_index in range(10):
        if round_index:
            k0 = (k0 + _W0) & 0xFFFFFFFF
            k1 = (k1 + _W1) & 0xFFFFFFFF
        p0 = _M0 * c[:, 0]
        p1 = _M1 * c[:, 2]
        c = np.stack(
            [(p1 >> np.uint64(32)) ^ c[:, 1] ^ np.uint64(k0), p1 & _MASK,
             (p0 >> np.uint64(32)) ^ c[:, 3] ^ np.uint64(k1), p0 & _MASK],
            axis=1,
        )
    return c.astype(np.uint32)


def gumbel(seed: int, step: int, vocab: int) -> np.ndarray:
    """Gumbel noise for every token id at one step, float64."""

    seed &= (1 << 64) - 1
    blocks = (vocab + 3) // 4
    counters = np.zeros((blocks, 4), dtype=np.uint32)
    counters[:, 0] = np.arange(blocks, dtype=np.uint32)
    counters[:, 1] = step
    counters[:, 2] = STREAM
    bits = philox4x32_10(counters, (seed & 0xFFFFFFFF, seed >> 32)).reshape(-1)[:vocab]
    u = ((bits >> 9).astype(np.float64) + 0.5) * 2.0**-23
    return -np.log(-np.log(u))


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int = 0

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0 or self.top_k == 1

    def cli_args(self) -> list[str]:
        return ["--temperature", repr(self.temperature), "--top-p", repr(self.top_p), "--top-k",
                str(self.top_k), "--min-p", repr(self.min_p), "--seed", str(self.seed)]


@dataclass(frozen=True)
class Draw:
    token: int
    kept: np.ndarray       # boolean mask of the support
    score_margin: float    # best minus runner-up perturbed score (inf if one token is kept)
    filters_stable: bool   # the token survives the filter limits moving by the given deltas


def kept_mask(logits: np.ndarray, params: SamplingParams, mass_shift: float = 0.0,
              logit_shift: float = 0.0) -> np.ndarray:
    """The support after filtering. The shifts move the top-p limit (mass) and
    the min-p threshold (logits) to test how close a decision is to flipping;
    top-k compares input values only, so rounding cannot move it."""

    values = np.asarray(logits, dtype=np.float64)
    top = values.max()
    keep = np.ones(values.shape[0], dtype=bool)
    if params.min_p > 0.0:
        keep &= values >= top + np.log(params.min_p) + logit_shift
    if 0 < params.top_k < values.shape[0]:
        keep &= values >= np.sort(values)[::-1][params.top_k - 1]
    if params.top_p < 1.0:
        probs = np.exp(values - top)
        probs /= probs.sum()
        distinct, inverse = np.unique(-values, return_inverse=True)  # descending groups
        group_mass = np.bincount(inverse, weights=probs, minlength=distinct.shape[0])
        before = np.concatenate([[0.0], np.cumsum(group_mass)[:-1]])  # mass strictly above each group
        keep &= before[inverse] < params.top_p + mass_shift
    keep[np.argmax(values)] = True  # every filter keeps the most likely token
    return keep


def sample(logits: np.ndarray, params: SamplingParams, step: int, mass_delta: float = 1e-6,
           logit_delta: float = 1e-5) -> Draw:
    values = np.asarray(logits, dtype=np.float32)
    if params.greedy:
        order = np.argsort(-values.astype(np.float64), kind="stable")
        margin = float(values[order[0]] - values[order[1]]) if values.shape[0] > 1 else np.inf
        return Draw(int(order[0]), np.eye(values.shape[0], dtype=bool)[order[0]], margin, True)
    noise = values.astype(np.float64) / params.temperature + gumbel(params.seed, step, values.shape[0])

    def draw(keep: np.ndarray) -> tuple[int, float]:
        scores = np.where(keep, noise, -np.inf)
        order = np.argsort(-scores, kind="stable")
        margin = float(scores[order[0]] - scores[order[1]]) if keep.sum() > 1 else np.inf
        return int(order[0]), margin

    keep = kept_mask(values, params)
    token, margin = draw(keep)
    shifted = [(mass_delta, 0.0), (-mass_delta, 0.0), (0.0, logit_delta), (0.0, -logit_delta)]
    stable = all(draw(kept_mask(values, params, m, l))[0] == token for m, l in shifted)
    return Draw(token, keep, margin, stable)


def expected_distribution(logits: np.ndarray, params: SamplingParams) -> np.ndarray:
    """Exact probabilities the sampler draws from."""

    keep = kept_mask(logits, params)
    values = np.asarray(logits, dtype=np.float64) / params.temperature
    weights = np.where(keep, np.exp(values - values[keep].max()), 0.0)
    return weights / weights.sum()
