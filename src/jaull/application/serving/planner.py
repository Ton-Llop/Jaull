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
) -> ServerSizing:
    if users < 1 or context_per_user < 1:
        raise ValueError("Users and context per user must be positive.")
    step = policies.SERVE_CONTEXT_STEP_TOKENS
    full = _round_up(context_per_user, step)
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
    share = _largest(smallest // step, full // step, lambda k: fits(k * step)) * step
    chosen = estimate(share)
    return ServerSizing(
        parallel=users,
        context_per_user=full,
        total_context=share * users,
        estimate=chosen,
        gpu_budget_bytes=_gpu_budget(chosen.hardware_fit),
    )


def _smallest_share(context: int, users: int, step: int) -> int:
    """Per-sequence share of the smallest pool that still lets one user reach ``context``."""
    return _round_up(-(-context // users), step)


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
        f"The largest that fits is {context} tokens; choose fewer users or a shorter context.",
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


__all__ = ["EstimateFor", "ServerSizing", "ServerSizingError", "size_server"]
