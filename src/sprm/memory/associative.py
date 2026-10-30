"""Form A: per-layer associative memory with closed-form delta-rule writes.

State is two matrices ``M_k, M_v in R^{B x d}``. A read is
``a = softmax(τ · k̂ M_kᵀ); v̂ = a M_v``. A write moves each slot towards the residual
``v - M_v a`` with gain ``η g``. Everything is fp32 and gradient-free.

Design corrections relative to DESIGN.md v0.2, recorded in docs/ADR-001 and ADR-002:

1. Keys are stored *unrotated* (RoPE removed) and rotated per slot at injection time.
   Mixing post-RoPE keys from different absolute positions is not a valid key.
2. Keys are L2-normalized and the softmax uses an explicit temperature ``τ``. With unit
   keys and ``d=512``, the uncorrected ``k M_kᵀ / sqrt(d)`` is nearly uniform and cannot
   retrieve anything.
3. The trust region has a norm floor, otherwise the first write to a zero-initialized
   ``M_v`` is scaled to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sprm.contracts.types import MemoryStats, WriteReport
from sprm.errors import MemoryInvariantError


@dataclass(frozen=True)
class AssociativeMemoryConfig:
    name: str
    num_slots: int
    key_dim: int
    value_dim: int
    read_temperature: float
    write_gain: float
    gate: float
    forgetting: float
    trust_region_eps: float
    key_l2_normalize: bool
    kmeans_iters: int = 8
    refine_passes: int = 2
    trust_region_floor: float = 1.0
    eps: float = 1e-8


class AssociativeMemory:
    """A single layer's B-slot key/value memory."""

    def __init__(self, config: AssociativeMemoryConfig, *, device: torch.device | str, seed: int):
        self.config = config
        if config.num_slots < 1 or config.key_dim < 1 or config.value_dim < 1:
            raise MemoryInvariantError("num_slots/key_dim/value_dim must be positive")
        if config.read_temperature <= 0:
            raise MemoryInvariantError("read_temperature must be positive")
        self.device = torch.device(device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        keys = torch.randn(config.num_slots, config.key_dim, generator=generator, dtype=torch.float32)
        keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(config.eps)
        self.M_k = keys.to(self.device)
        self.M_v = torch.zeros(config.num_slots, config.value_dim, device=self.device, dtype=torch.float32)
        self.slot_positions = torch.full((config.num_slots,), -1, device=self.device, dtype=torch.long)
        self.slot_block_ids = torch.full((config.num_slots,), -1, device=self.device, dtype=torch.long)
        self._stats = MemoryStats(
            name=config.name,
            num_slots=config.num_slots,
            key_dim=config.key_dim,
            value_dim=config.value_dim,
        )
        self._cond_cache: float | None = None
        self._initialized_slots = 0

    # ------------------------------------------------------------------ helpers
    @property
    def stats(self) -> MemoryStats:
        return self._stats

    @property
    def num_slots(self) -> int:
        return self.config.num_slots

    def _normalize(self, keys: torch.Tensor) -> torch.Tensor:
        if not self.config.key_l2_normalize:
            return keys
        normalized: torch.Tensor = keys / keys.norm(dim=-1, keepdim=True).clamp_min(self.config.eps)
        return normalized

    def _softmax_weights(self, keys: torch.Tensor) -> torch.Tensor:
        logits = self.config.read_temperature * (keys @ self.M_k.transpose(0, 1))
        return torch.softmax(logits.float(), dim=-1)

    def _invalidate(self) -> None:
        self._cond_cache = None

    # ------------------------------------------------------------------ reads
    def read(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``keys`` ``[T,d]`` unrotated -> ``(values_hat [T,d], attn [T,B])``."""
        if keys.ndim != 2 or keys.shape[-1] != self.config.key_dim:
            raise MemoryInvariantError(
                f"read keys must be [T,{self.config.key_dim}], got {tuple(keys.shape)}"
            )
        k = self._normalize(keys.float())
        a = self._softmax_weights(k)
        return a @ self.M_v, a

    def recon_error(self, keys: torch.Tensor, values: torch.Tensor) -> float:
        if keys.shape[-1] != self.config.key_dim or values.shape[-1] != self.config.value_dim:
            raise MemoryInvariantError("recon_error shape mismatch")
        if keys.shape[0] != values.shape[0]:
            raise MemoryInvariantError("recon_error needs one key per value")
        if keys.shape[0] == 0:
            return float("nan")
        v_hat, _ = self.read(keys)
        numerator = (values.float() - v_hat).square().sum()
        denominator = values.float().square().sum().clamp_min(self.config.eps)
        value = float((numerator / denominator).item())
        self._stats.last_recon_error = value
        if not torch.isfinite(torch.tensor(value)):
            raise MemoryInvariantError(f"non-finite reconstruction error: {value}")
        alpha = 0.1
        if self._stats.recon_error_ema != self._stats.recon_error_ema:  # NaN
            self._stats.recon_error_ema = value
        else:
            self._stats.recon_error_ema = (1 - alpha) * self._stats.recon_error_ema + alpha * value
        return value

    def condition(self) -> float:
        if self._cond_cache is not None:
            return self._cond_cache
        # SVD on CPU: MPS has no stable svdvals for every dtype, and this is diagnostic.
        matrix = self.M_k.detach().float().cpu()
        try:
            singular = torch.linalg.svdvals(matrix)
        except RuntimeError:
            self._cond_cache = float("inf")
            return self._cond_cache
        smax = float(singular[0].item()) if singular.numel() else 0.0
        smin = float(singular[-1].item()) if singular.numel() else 0.0
        cond = float("inf") if smin <= self.config.eps else smax / smin
        self._cond_cache = cond
        self._stats.condition_number = cond
        return cond

    # ------------------------------------------------------------------ writes
    def _delta_update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        gain: float,
        slot_slice: slice,
        count: bool = True,
    ) -> WriteReport:
        if keys.shape[0] == 0:
            raise MemoryInvariantError("empty write")
        k = self._normalize(keys.float())
        v = values.float()
        mk = self.M_k[slot_slice]
        mv = self.M_v[slot_slice]
        logits = self.config.read_temperature * (k @ mk.transpose(0, 1))
        a = torch.softmax(logits, dim=-1)
        v_hat = a @ mv
        residual = v - v_hat
        n = k.shape[0]
        delta_v = (a.transpose(0, 1) @ residual) / n
        delta_k = (a.transpose(0, 1) @ (k - a @ mk)) / n
        delta_v = gain * delta_v
        delta_k = gain * delta_k

        state_norm = float(torch.linalg.matrix_norm(mv, ord="fro").item())
        delta_norm = float(torch.linalg.matrix_norm(delta_v, ord="fro").item())
        cap = self.config.trust_region_eps * max(state_norm, self.config.trust_region_floor)
        trust_scaled = False
        if delta_norm > cap > 0:
            scale = cap / delta_norm
            delta_v = delta_v * scale
            delta_k = delta_k * scale
            trust_scaled = True
            delta_norm = cap
            self._stats.trust_scaled_count += 1

        with torch.no_grad():
            mv.add_(delta_v)
            mk.add_(delta_k)
            if self.config.key_l2_normalize:
                mk.copy_(self._normalize(mk))
        self._invalidate()
        if count:
            self._stats.writes += 1
            self._stats.tokens_written += int(n)
        touched = int(torch.unique(a.argmax(dim=-1)).numel())
        return WriteReport(
            tokens=int(n),
            slots_touched=touched,
            delta_norm=delta_norm,
            state_norm=state_norm,
            trust_scaled=trust_scaled,
            gain=gain,
            mean_abs_error=float(residual.abs().mean().item()),
        )

    def write(self, keys: torch.Tensor, values: torch.Tensor, *, gain: float | None = None) -> WriteReport:
        """W1: correct the whole memory from a batch of exact tokens."""
        if keys.ndim != 2 or values.ndim != 2:
            raise MemoryInvariantError("write expects [T,d] tensors")
        if keys.shape[0] != values.shape[0]:
            raise MemoryInvariantError("write needs one value per key")
        effective_gain = self.config.write_gain if gain is None else gain
        if effective_gain < 0:
            raise MemoryInvariantError("gain must be >= 0")
        return self._delta_update(keys, values, gain=effective_gain, slot_slice=slice(0, self.num_slots))

    def write_slots(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        slot_start: int,
        slot_count: int,
        positions: torch.Tensor,
        block_id: int,
        gain: float | None = None,
    ) -> WriteReport:
        """W2: initialize and refine one block's dedicated slots.

        Initialization is a cosine k-means; the delta refinement then runs only over this
        block's slot range, so a later block cannot wash out an earlier one during build.
        """
        if slot_start < 0 or slot_count < 1 or slot_start + slot_count > self.num_slots:
            raise MemoryInvariantError(
                f"invalid slot range [{slot_start},{slot_start + slot_count}) for B={self.num_slots}"
            )
        if keys.shape[0] < slot_count:
            raise MemoryInvariantError(
                f"block has {keys.shape[0]} tokens but requests {slot_count} slots"
            )
        if positions.numel() != keys.shape[0]:
            raise MemoryInvariantError("positions must match the block token count")
        k = self._normalize(keys.float())
        v = values.float()
        assignments, centers, value_centers, center_positions = _cosine_kmeans(
            k, v, positions, slot_count, iters=self.config.kmeans_iters, seed=self._seed_for(block_id)
        )
        sl = slice(slot_start, slot_start + slot_count)
        with torch.no_grad():
            self.M_k[sl] = centers
            self.M_v[sl] = value_centers
            self.slot_positions[sl] = center_positions
            self.slot_block_ids[sl] = block_id
        self._invalidate()
        self._initialized_slots = min(self.num_slots, self._initialized_slots + slot_count)
        report = WriteReport(
            tokens=int(k.shape[0]),
            slots_touched=int(slot_count),
            delta_norm=0.0,
            state_norm=float(torch.linalg.matrix_norm(self.M_v[sl], ord="fro").item()),
            trust_scaled=False,
            gain=0.0,
            mean_abs_error=float((v - value_centers[assignments]).abs().mean().item()),
        )
        self._stats.writes += 1
        self._stats.tokens_written += int(k.shape[0])
        gain = self.config.write_gain if gain is None else gain
        for _ in range(self.config.refine_passes):
            report = self._delta_update(k, v, gain=gain, slot_slice=sl, count=False)
        return report

    def _seed_for(self, block_id: int) -> int:
        import hashlib

        base = int.from_bytes(hashlib.sha256(self.config.name.encode("utf-8")).digest()[:4], "little")
        return base ^ (block_id * 2654435761)

    def apply_forgetting(self, factor: float | None = None) -> None:
        """Decay stored *values*.

        Unit-normalized keys are invariant to a global scalar, so decaying ``M_k`` would be a
        no-op after renormalization. Conditioning is handled by the trust region and the
        condition monitor; forgetting controls the value memory's effective horizon.
        """
        factor = self.config.forgetting if factor is None else factor
        if not 0.0 <= factor <= 1.0:
            raise MemoryInvariantError("forgetting factor must be in [0,1]")
        with torch.no_grad():
            self.M_v.mul_(factor)
        self._invalidate()
        self._stats.forgetting_events += 1

    # ------------------------------------------------------------------ housekeeping
    def memory_bytes(self) -> int:
        return int(
            self.M_k.numel() * self.M_k.element_size()
            + self.M_v.numel() * self.M_v.element_size()
            + self.slot_positions.numel() * self.slot_positions.element_size()
            + self.slot_block_ids.numel() * self.slot_block_ids.element_size()
        )

    def slot_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "M_k": self.M_k.detach().float().cpu(),
            "M_v": self.M_v.detach().float().cpu(),
            "slot_positions": self.slot_positions.detach().cpu(),
            "slot_block_ids": self.slot_block_ids.detach().cpu(),
        }

    def clone(self) -> AssociativeMemory:
        """Deep copy of the state, used so two decode arms start from one build."""
        other = object.__new__(AssociativeMemory)
        other.config = self.config
        other.device = self.device
        other.M_k = self.M_k.clone()
        other.M_v = self.M_v.clone()
        other.slot_positions = self.slot_positions.clone()
        other.slot_block_ids = self.slot_block_ids.clone()
        other._stats = MemoryStats(**vars(self._stats))
        other._stats.extra = dict(self._stats.extra)
        other._cond_cache = self._cond_cache
        other._initialized_slots = self._initialized_slots
        return other

    def reset(self) -> None:
        AssociativeMemory.__init__(self, self.config, device=self.device, seed=0)


# ---------------------------------------------------------------------- k-means
def _cosine_kmeans(
    keys: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    k: int,
    *,
    iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic wrapper: cluster on CPU, return tensors on the input device.

    MPS reductions (`index_add_`) are not bit-reproducible, and a small drift in cluster
    centers changes the decode logits. The block is at most a few hundred tokens, so the CPU
    cost is negligible and the result is stable across runs.
    """
    device = keys.device
    assignments, centers, value_centers, center_positions = _cosine_kmeans_cpu(
        keys.detach().float().cpu(),
        values.detach().float().cpu(),
        positions.detach().cpu(),
        k,
        iters=iters,
        seed=seed,
    )
    return (
        assignments.to(device),
        centers.to(device),
        value_centers.to(device=device, dtype=values.dtype),
        center_positions.to(device),
    )


def _cosine_kmeans_cpu(
    keys: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    k: int,
    *,
    iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic cosine k-means with farthest-point initialization.

    Returns ``(assignments, key_centers, value_centers, center_positions)``.
    """
    n, _dim = keys.shape
    if n < k:
        raise MemoryInvariantError(f"cannot cluster {n} tokens into {k} slots")
    if k == 1:
        assignments = torch.zeros(n, dtype=torch.long, device=keys.device)
        return (
            assignments,
            keys.mean(dim=0, keepdim=True),
            values.mean(dim=0, keepdim=True),
            positions.float().mean().round().long().view(1),
        )

    # Farthest-point init: deterministic and spreads centers over the block.
    first = int(torch.argmax(keys.norm(dim=-1)).item())
    chosen = [first]
    min_dist = 1.0 - keys @ keys[first]
    for _ in range(1, k):
        idx = int(torch.argmax(min_dist).item())
        if idx in chosen:
            # All remaining points coincide; pick the first unchosen index.
            idx = next(i for i in range(n) if i not in chosen)
        chosen.append(idx)
        dist = 1.0 - keys @ keys[idx]
        min_dist = torch.minimum(min_dist, dist)
    centers = keys[torch.tensor(chosen, device=keys.device)].clone()

    assignments = torch.zeros(n, dtype=torch.long, device=keys.device)
    for _ in range(iters):
        sim = keys @ centers.transpose(0, 1)
        assignments = sim.argmax(dim=-1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=keys.device, dtype=torch.float32)
        new_centers.index_add_(0, assignments, keys)
        counts.index_add_(0, assignments, torch.ones(n, device=keys.device))
        empty = counts == 0
        if bool(empty.any()):
            # Re-seed empty clusters with the currently worst-fit points.
            worst = torch.topk(1.0 - sim.max(dim=-1).values, int(empty.sum())).indices
            new_centers[empty] = keys[worst]
            counts[empty] = 1.0
        centers = new_centers / counts.clamp_min(1.0).unsqueeze(-1)
        centers = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    sim = keys @ centers.transpose(0, 1)
    assignments = sim.argmax(dim=-1)
    value_centers = torch.zeros(k, values.shape[-1], device=values.device, dtype=values.dtype)
    counts = torch.zeros(k, device=values.device, dtype=torch.float32)
    value_centers.index_add_(0, assignments, values)
    counts.index_add_(0, assignments, torch.ones(n, device=values.device))
    value_centers = value_centers / counts.clamp_min(1.0).unsqueeze(-1)
    pos_sum = torch.zeros(k, device=positions.device, dtype=torch.float32)
    pos_sum.index_add_(0, assignments, positions.float())
    center_positions = (pos_sum / counts.clamp_min(1.0)).round().long()
    return assignments, centers, value_centers, center_positions
