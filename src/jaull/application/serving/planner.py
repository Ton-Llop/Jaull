"""Choose slots and context for a llama-server shared by N users.

With one user the server gets exactly the requested context. With more, the KV
cache is one shared pool (``--kv-unified``): any single user may reach the full
per-user context, and the pool is as large as fits, up to everyone at full
context. The bytes come from the estimator, called through ``estimate_for`` with
``concurrent_users = N`` and each sequence's share of the pool; that is the case
validated on the B60 (docs/TFG/concurrencia-b60.md). What "fits" means for a
server is the policy in :mod:`jaull.application.serving.policies`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import gcd

from jaull.application.serving import policies
from jaull.domain.estimation import HardwareFitMode, HardwareFitResult, MemoryEstimate
from jaull.domain.inference import InferenceConfiguration, TargetDevice

EstimateFor = Callable[[InferenceConfiguration], MemoryEstimate]


class ServerSizingError(ValueError):
    """The requested users and context do not fit for serving on this machine."""

    def __init__(self, message: str, *, largest_context_per_user: int | None) -> None:
        super().__init__(message)
        self.largest_context_per_user = largest_context_per_user


@dataclass(frozen=True)
class ContextLimit:
    """The largest context one person can have on this machine, for ``users`` at once."""

    users: int
    #: Rounded down to the planning step. None when not even one step fits.
    largest_per_user: int | None
    #: What the model declares it was trained for, when known. Never exceeded.
    native: int | None


@dataclass(frozen=True)
class ServerSizing:
    parallel: int
    context_per_user: int
    total_context: int
    estimate: MemoryEstimate
    gpu_budget_bytes: int | None

    @property
    def unified_kv(self) -> bool:
        return self.parallel > 1


def size_server(
    *,
    users: int,
    context_per_user: int,
    base: InferenceConfiguration,
    estimate_for: EstimateFor,
    shared_context: bool = False,
    native_limit: int | None = None,
) -> ServerSizing:
    """``shared_context`` keeps the pool at ``context_per_user`` instead of growing it.

    Everyone then draws from one pool of that size: a single person may use all of
    it, and several people split it as they use it. Without the option the pool
    grows, as far as it fits, towards everyone having the full context at once.
    """
    if users < 1 or context_per_user < 1:
        raise ValueError("Users and context per user must be positive.")
    if native_limit is not None and context_per_user > native_limit:
        limit = context_limit(users=users, base=base, estimate_for=estimate_for,
                              native_limit=native_limit)
        raise ServerSizingError(
            f"The model was trained for {native_limit} tokens of context, less than the "
            f"{context_per_user} asked for. The largest this machine can serve per person is "
            f"{limit.largest_per_user or 0} tokens.",
            largest_context_per_user=limit.largest_per_user,
        )
    step = _context_step(native_limit)
    full = _round_up(context_per_user, step)
    if native_limit is not None:
        full = min(full, native_limit)
    cache: dict[int, MemoryEstimate] = {}

    def estimate(share: int) -> MemoryEstimate:
        if share not in cache:
            cache[share] = estimate_for(base.model_copy(update={
                "context_length": share, "concurrent_users": users,
            }))
        return cache[share]

    def fits(share: int) -> bool:
        return _fits_for_serving(estimate(share))

    smallest = _smallest_share(full, users, step)
    if not fits(smallest):
        raise _does_not_fit(users, full, step, fits)
    if shared_context:
        share = smallest
    else:
        share = _largest(smallest // step, full // step, lambda k: fits(k * step)) * step
    chosen = estimate(share)
    return ServerSizing(
        parallel=users,
        context_per_user=full,
        # A shared pool is exactly the asked-for context. Its estimate used the
        # rounded-up share, so it errs on the side of more memory.
        total_context=full if shared_context else share * users,
        estimate=chosen,
        gpu_budget_bytes=_gpu_budget(chosen.hardware_fit),
    )


def context_limit(
    *,
    users: int,
    base: InferenceConfiguration,
    estimate_for: EstimateFor,
    native_limit: int | None = None,
) -> ContextLimit:
    """Largest context per person that fits for serving, searched with the estimator.

    The pool is the smallest that lets one person reach that context, so the answer
    holds both with and without ``shared_context``. It never exceeds what the model
    declares; without a declaration the search stops at a policy ceiling.
    """
    if users < 1:
        raise ValueError("Users must be positive.")
    step = _context_step(native_limit)
    ceiling = native_limit or policies.SERVE_CONTEXT_SEARCH_CEILING_TOKENS
    cache: dict[int, bool] = {}

    def fits_at(k: int) -> bool:
        share = _smallest_share(k * step, users, step)
        if share not in cache:
            cache[share] = _fits_for_serving(estimate_for(base.model_copy(update={
                "context_length": share, "concurrent_users": users,
            })))
        return cache[share]

    top = max(1, ceiling // step)
    if not fits_at(1):
        return ContextLimit(users=users, largest_per_user=None, native=native_limit)
    return ContextLimit(users=users, largest_per_user=_largest(1, top, fits_at) * step,
                        native=native_limit)


def _smallest_share(context: int, users: int, step: int) -> int:
    """Per-sequence share of the smallest pool that still lets one user reach ``context``."""
    return _round_up(-(-context // users), step)


def _context_step(native_limit: int | None) -> int:
    if native_limit is not None and native_limit < 1:
        raise ValueError("Native context must be positive.")
    # Keep both the search grid and its rounded estimates within the native limit.
    return (gcd(policies.SERVE_CONTEXT_STEP_TOKENS, native_limit)
            if native_limit is not None else policies.SERVE_CONTEXT_STEP_TOKENS)


def _does_not_fit(
    users: int, full: int, step: int, fits: Callable[[int], bool],
) -> ServerSizingError:
    who = "1 user" if users == 1 else f"{users} users"
    if not fits(step):
        return ServerSizingError(
            f"The model does not fit for serving {who} on this machine, even with "
            f"{step} tokens of context. Try a smaller quantization or model.",
            largest_context_per_user=None,
        )
    largest = _largest(1, full // step, lambda k: fits(_smallest_share(k * step, users, step)))
    context = largest * step
    return ServerSizingError(
        f"{full} tokens per user does not fit for serving {who} with the memory margin. "
        f"The largest that fits is {context} tokens. Choose fewer users or a shorter context.",
        largest_context_per_user=context,
    )


def _largest(low: int, high: int, ok: Callable[[int], bool]) -> int:
    """Largest k in [low, high] with ok(k), given ok(low) and ok monotone decreasing."""
    while low < high:
        middle = (low + high + 1) // 2
        if ok(middle):
            low = middle
        else:
            high = middle - 1
    return low


def _fits_for_serving(estimate: MemoryEstimate) -> bool:
    fit = estimate.hardware_fit
    if fit is None:
        # An explicit CPU target runs no placement; its assessment is the answer.
        return (
            estimate.inference_configuration.target_device is TargetDevice.CPU
            and estimate.assessment.status in policies.SERVE_CPU_ACCEPTED_STATUSES
        )
    if fit.mode is HardwareFitMode.CPU_RAM:
        return True
    if fit.mode is not HardwareFitMode.GPU_RESIDENT:
        return False
    physical = fit.gpu_physical_bytes
    budget = _gpu_budget(fit)
    return physical is not None and budget is not None and physical <= budget


def _gpu_budget(fit: HardwareFitResult | None) -> int | None:
    if fit is None or fit.available_vram_bytes is None or not fit.places_weights_on_gpu:
        return None
    return int(fit.available_vram_bytes * policies.SERVE_GPU_MAX_USAGE_FRACTION)


def _round_up(value: int, step: int) -> int:
    return -(-value // step) * step


__all__ = [
    "ContextLimit",
    "EstimateFor",
    "ServerSizing",
    "ServerSizingError",
    "context_limit",
    "size_server",
]
