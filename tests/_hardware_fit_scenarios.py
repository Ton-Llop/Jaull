"""Deterministic placement scenarios for the hardware fit analyzer.

One catalogue, two consumers: ``tests/test_hardware_fit_scenarios.py`` asserts
the expected mode and the conservation invariants, and
``scripts/hardware_fit_matrix.py`` renders the same rows as an observation
table (and feeds ``scripts/hardware_fit_vs_llmfit.py``). Keeping them on one
list is the point — a scenario that only the report knows about is a scenario
nothing checks.

Every input is an exact power-of-two multiple of a MiB, chosen so the expected
mode follows from arithmetic a reader can redo by hand rather than from a
recorded run. The analyzer consumes memory components it did not compute, so
the numbers here are placement inputs, not estimator output: how a real
``MemoryEstimate`` arrives at them is exercised in the estimator's own tests.

The machine knob is *available* VRAM, not installed VRAM, because that is what
``analyze_components`` reads. ``vram_total_bytes`` is set equal to it so the
fixtures state one number per pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jaull.domain.estimation import (
    HardwareFitMode,
    HardwareFitPlacementMethod,
    HardwareFitResult,
)
from jaull.domain.hardware import (
    AcceleratorProfile,
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.estimator.hardware_fit import analyze_components

MIB = 1024**2
GIB = 1024**3

# Every property the catalogue promises to exercise.
# ``test_catalogue_covers_every_required_case`` fails if a scenario stops
# carrying one of these, so coverage cannot quietly regress when rows are
# edited.
REQUIRED_COVERAGE = frozenset(
    {
        "gpu_resident",
        "gpu_offload",
        "cpu_ram",
        "too_large",
        "exact_boundary",
        "kv_sensitivity",
        "concurrency_sensitivity",
        "transformer_block_placement",
        "byte_fallback",
        "false_pool_sum",
        "no_gpu_topology",
        "unified_reserve",
    }
)

# The fields the analyzer is being observed on, in report order.
OBSERVED_FIELDS: tuple[str, ...] = (
    "mode",
    "memory_topology",
    "placement_method",
    "available_vram_bytes",
    "available_ram_bytes",
    "gpu_required_bytes",
    "ram_required_bytes",
    "gpu_weight_bytes",
    "ram_weight_bytes",
    "gpu_kv_cache_bytes",
    "ram_kv_cache_bytes",
    "gpu_overhead_bytes",
    "ram_overhead_bytes",
    "gpu_safety_margin_bytes",
    "ram_safety_margin_bytes",
    "device_reserve_bytes",
    "gpu_transformer_blocks",
    "total_transformer_blocks",
    "offload_diagnostics",
)


@dataclass(frozen=True)
class Machine:
    """A machine described only by the pools the analyzer actually reads."""

    name: str
    ram_available_bytes: int
    ram_total_bytes: int
    vram_available_bytes: int | None = None
    unified_memory: bool = False

    def profile(self) -> HardwareProfile:
        vram = self.vram_available_bytes
        gpus = (
            [
                GpuInfo(
                    name=f"{self.name} GPU",
                    vram_total_bytes=vram,
                    vram_available_bytes=vram,
                    driver_version="000.00",
                    cuda_version="12.0",
                )
            ]
            if vram is not None and not self.unified_memory
            else []
        )
        accelerators = (
            [
                AcceleratorProfile(
                    name=f"{self.name} unified accelerator",
                    shared_memory=True,
                    available_memory_bytes=self.ram_available_bytes,
                )
            ]
            if self.unified_memory
            else []
        )
        return HardwareProfile(
            os="Linux",
            arch="x86_64",
            cpu=CpuInfo(model="Scenario CPU", physical_cores=8, logical_cores=16),
            memory=MemoryInfo(
                total_bytes=self.ram_total_bytes,
                available_bytes=self.ram_available_bytes,
            ),
            storage=[],
            gpus=gpus,
            accelerators=accelerators,
            warnings=[],
        )


DISCRETE_8GB_24GB = Machine(
    name="8 GiB VRAM / 24 GiB free RAM",
    ram_available_bytes=24 * GIB,
    ram_total_bytes=32 * GIB,
    vram_available_bytes=8 * GIB,
)
DISCRETE_8GB_16GB = Machine(
    name="8 GiB VRAM / 16 GiB free RAM",
    ram_available_bytes=16 * GIB,
    ram_total_bytes=16 * GIB,
    vram_available_bytes=8 * GIB,
)
DISCRETE_6GB_10GB = Machine(
    name="6 GiB VRAM / 10 GiB free RAM",
    ram_available_bytes=10 * GIB,
    ram_total_bytes=16 * GIB,
    vram_available_bytes=6 * GIB,
)
DISCRETE_4GB_24GB = Machine(
    name="4 GiB VRAM / 24 GiB free RAM",
    ram_available_bytes=24 * GIB,
    ram_total_bytes=32 * GIB,
    vram_available_bytes=4 * GIB,
)
NO_GPU_12GB = Machine(
    name="no GPU / 12 GiB free RAM",
    ram_available_bytes=12 * GIB,
    ram_total_bytes=16 * GIB,
)
UNIFIED_16GB = Machine(
    name="unified memory / 16 GiB free",
    ram_available_bytes=16 * GIB,
    ram_total_bytes=24 * GIB,
    unified_memory=True,
)


@dataclass(frozen=True)
class Scenario:
    """One placement question, its inputs, and the answer it must produce."""

    name: str
    question: str
    machine: Machine
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    expected_mode: HardwareFitMode
    expected_placement: HardwareFitPlacementMethod
    covers: frozenset[str]
    device_reserve_bytes: int = 0
    safety_margin_bytes: int = 0
    total_transformer_blocks: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def analyze(self) -> HardwareFitResult:
        return analyze_components(
            weights_bytes=self.weights_bytes,
            kv_cache_bytes=self.kv_cache_bytes,
            overhead_bytes=self.overhead_bytes,
            hardware=self.machine.profile(),
            device_reserve_bytes=self.device_reserve_bytes,
            safety_margin_bytes=self.safety_margin_bytes,
            total_transformer_blocks=self.total_transformer_blocks,
        )

    def inputs(self) -> dict[str, object]:
        return {
            "machine": self.machine.name,
            "weights_bytes": self.weights_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "overhead_bytes": self.overhead_bytes,
            "device_reserve_bytes": self.device_reserve_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "total_transformer_blocks": self.total_transformer_blocks,
        }

    def observe(self) -> dict[str, object]:
        """The analyzer's answer, flattened to the fields under observation."""
        result = self.analyze()
        observed: dict[str, object] = {}
        for name in OBSERVED_FIELDS:
            value = getattr(result, name)
            if hasattr(value, "model_dump"):
                observed[name] = value.model_dump(mode="json")
            else:
                observed[name] = value.value if hasattr(value, "value") else value
        return observed


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="gpu_resident_comfortable",
        question="Everything fits in VRAM with headroom to spare.",
        machine=DISCRETE_8GB_24GB,
        weights_bytes=5 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=512 * MIB,
        safety_margin_bytes=256 * MIB,
        total_transformer_blocks=32,
        expected_mode=HardwareFitMode.GPU_RESIDENT,
        expected_placement=HardwareFitPlacementMethod.TRANSFORMER_BLOCKS,
        covers=frozenset({"gpu_resident"}),
        notes=("7.75 GiB required against 8 GiB of VRAM.",),
    ),
    Scenario(
        name="gpu_offload_transformer_block_split",
        question="Weights exceed VRAM, but a transformer-block split fits both pools.",
        machine=DISCRETE_8GB_24GB,
        weights_bytes=10 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=512 * MIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=40,
        expected_mode=HardwareFitMode.GPU_OFFLOAD,
        expected_placement=HardwareFitPlacementMethod.TRANSFORMER_BLOCKS,
        covers=frozenset({"gpu_offload", "transformer_block_placement"}),
        notes=("13 GiB resident requirement; 6.5 GiB of VRAM left for weights.",),
    ),
    Scenario(
        name="cpu_ram_when_no_gpu_placement_is_viable",
        question="One transformer block exceeds the whole card, so only RAM is left.",
        machine=DISCRETE_4GB_24GB,
        weights_bytes=16 * GIB,
        kv_cache_bytes=4 * GIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=512 * MIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=4,
        expected_mode=HardwareFitMode.CPU_RAM,
        expected_placement=HardwareFitPlacementMethod.NONE,
        covers=frozenset({"cpu_ram"}),
        notes=(
            "The smallest placeable unit is one block, 4 GiB, which already "
            "fills the 4 GiB card before its KV share or the reserve.",
            "This case used to be reached by charging the whole KV cache to "
            "VRAM. Once the cache follows the block placement, a single-block "
            "offload is cheap, so CPU_RAM on discrete hardware now means the "
            "block itself does not fit — not that the cache did not.",
        ),
    ),
    Scenario(
        name="too_large_for_both_pools",
        question="Neither a GPU split nor a RAM-only placement fits.",
        machine=DISCRETE_8GB_16GB,
        weights_bytes=40 * GIB,
        kv_cache_bytes=4 * GIB,
        overhead_bytes=4 * GIB,
        device_reserve_bytes=512 * MIB,
        safety_margin_bytes=4 * GIB,
        total_transformer_blocks=80,
        expected_mode=HardwareFitMode.TOO_LARGE,
        expected_placement=HardwareFitPlacementMethod.NONE,
        covers=frozenset({"too_large"}),
        notes=("52 GiB of CPU requirement against 16 GiB of free RAM.",),
    ),
    Scenario(
        name="exact_vram_boundary_is_a_fit",
        question="required == available on the VRAM side.",
        machine=DISCRETE_8GB_24GB,
        weights_bytes=5 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=512 * MIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=32,
        expected_mode=HardwareFitMode.GPU_RESIDENT,
        expected_placement=HardwareFitPlacementMethod.TRANSFORMER_BLOCKS,
        covers=frozenset({"gpu_resident", "exact_boundary"}),
        notes=("gpu_required_bytes is exactly 8 GiB; the comparison is <=.",),
    ),
    Scenario(
        name="exact_ram_boundary_is_a_fit",
        question="required == available on the RAM side, with no GPU present.",
        machine=NO_GPU_12GB,
        weights_bytes=8 * GIB,
        kv_cache_bytes=2 * GIB,
        overhead_bytes=1 * GIB,
        safety_margin_bytes=1 * GIB,
        total_transformer_blocks=32,
        expected_mode=HardwareFitMode.CPU_RAM,
        expected_placement=HardwareFitPlacementMethod.NONE,
        covers=frozenset({"cpu_ram", "exact_boundary", "no_gpu_topology"}),
        notes=(
            "ram_required_bytes is exactly 12 GiB. On discrete hardware a device "
            "reserve is VRAM, so a CPU_RAM placement does not pay for it. "
            "gpu_transformer_blocks is None, not 0: there is no GPU for a "
            "transformer-block count to describe.",
        ),
    ),
    Scenario(
        name="no_gpu_too_large",
        question=(
            "No GPU and not enough RAM — does gpu_transformer_blocks stay None?"
        ),
        machine=NO_GPU_12GB,
        weights_bytes=20 * GIB,
        kv_cache_bytes=2 * GIB,
        overhead_bytes=2 * GIB,
        safety_margin_bytes=2 * GIB,
        total_transformer_blocks=48,
        expected_mode=HardwareFitMode.TOO_LARGE,
        expected_placement=HardwareFitPlacementMethod.NONE,
        covers=frozenset({"too_large", "no_gpu_topology"}),
        notes=(
            "Pairs with exact_ram_boundary_is_a_fit: both report "
            "gpu_transformer_blocks None because the machine has no GPU, "
            "regardless of the mode reached.",
        ),
    ),
    Scenario(
        name="kv_small_keeps_the_model_on_the_gpu",
        question="Short context: the same weights stay VRAM-resident.",
        machine=DISCRETE_8GB_24GB,
        weights_bytes=5 * GIB,
        kv_cache_bytes=512 * MIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=512 * MIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=32,
        expected_mode=HardwareFitMode.GPU_RESIDENT,
        expected_placement=HardwareFitPlacementMethod.TRANSFORMER_BLOCKS,
        covers=frozenset({"gpu_resident", "kv_sensitivity"}),
        notes=("Pairs with kv_large_pushes_the_model_off_the_gpu.",),
    ),
    Scenario(
        name="kv_large_pushes_the_model_off_the_gpu",
        question="Long context: only the KV cache changed, and the fit moved.",
        machine=DISCRETE_8GB_24GB,
        weights_bytes=5 * GIB,
        kv_cache_bytes=3 * GIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=512 * MIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=32,
        expected_mode=HardwareFitMode.GPU_OFFLOAD,
        expected_placement=HardwareFitPlacementMethod.TRANSFORMER_BLOCKS,
        covers=frozenset({"gpu_offload", "kv_sensitivity"}),
        notes=("Pairs with kv_small_keeps_the_model_on_the_gpu.",),
    ),
    Scenario(
        name="single_user_kv_fits_on_the_gpu",
        question="One concurrent user: 1 x 1 GiB of KV cache.",
        machine=DISCRETE_8GB_24GB,
        weights_bytes=4 * GIB,
        kv_cache_bytes=1 * (1 * GIB),
        overhead_bytes=1 * GIB,
        device_reserve_bytes=512 * MIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=28,
        expected_mode=HardwareFitMode.GPU_RESIDENT,
        expected_placement=HardwareFitPlacementMethod.TRANSFORMER_BLOCKS,
        covers=frozenset({"gpu_resident", "concurrency_sensitivity"}),
        notes=("Pairs with four_concurrent_users_force_offload.",),
    ),
    Scenario(
        name="four_concurrent_users_force_offload",
        question="Four concurrent users: 4 x 1 GiB of KV cache, same weights.",
        machine=DISCRETE_8GB_24GB,
        weights_bytes=4 * GIB,
        kv_cache_bytes=4 * (1 * GIB),
        overhead_bytes=1 * GIB,
        device_reserve_bytes=512 * MIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=28,
        expected_mode=HardwareFitMode.GPU_OFFLOAD,
        expected_placement=HardwareFitPlacementMethod.TRANSFORMER_BLOCKS,
        covers=frozenset({"gpu_offload", "concurrency_sensitivity"}),
        notes=("Pairs with single_user_kv_fits_on_the_gpu.",),
    ),
    Scenario(
        name="transformer_block_placement_with_metadata",
        question=(
            "With a transformer-block count, the split is expressed in "
            "transformer blocks."
        ),
        machine=DISCRETE_8GB_24GB,
        weights_bytes=9 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=256 * MIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=36,
        expected_mode=HardwareFitMode.GPU_OFFLOAD,
        expected_placement=HardwareFitPlacementMethod.TRANSFORMER_BLOCKS,
        covers=frozenset({"gpu_offload", "transformer_block_placement"}),
        notes=("Pairs with byte_fallback_without_transformer_block_metadata.",),
    ),
    Scenario(
        name="byte_fallback_without_transformer_block_metadata",
        question="Without a transformer-block count, the split is estimated in bytes.",
        machine=DISCRETE_8GB_24GB,
        weights_bytes=9 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=256 * MIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=None,
        expected_mode=HardwareFitMode.GPU_OFFLOAD,
        expected_placement=HardwareFitPlacementMethod.ESTIMATED_BYTES,
        covers=frozenset({"gpu_offload", "byte_fallback"}),
        notes=("Pairs with transformer_block_placement_with_metadata.",),
    ),
    Scenario(
        name="vram_plus_ram_is_not_a_single_pool",
        question="Sum of pools is exactly enough, yet no placement is viable.",
        machine=DISCRETE_6GB_10GB,
        weights_bytes=10 * GIB,
        kv_cache_bytes=5 * GIB,
        overhead_bytes=1 * GIB,
        total_transformer_blocks=10,
        expected_mode=HardwareFitMode.TOO_LARGE,
        expected_placement=HardwareFitPlacementMethod.NONE,
        covers=frozenset({"too_large", "false_pool_sum"}),
        notes=(
            "weights + kv + overhead == 16 GiB == VRAM + RAM. A tool that adds "
            "the pools reports a fit here; a placement-aware one cannot, because "
            "the KV cache pins 5 of the 6 GiB of VRAM and the 9 GiB of weights "
            "left over do not fit in 10 GiB of RAM once overhead is split.",
        ),
    ),
    Scenario(
        name="unified_memory_uses_one_shared_pool",
        question="Unified memory: one pool, checked once, never summed.",
        machine=UNIFIED_16GB,
        weights_bytes=6 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=32,
        expected_mode=HardwareFitMode.CPU_RAM,
        expected_placement=HardwareFitPlacementMethod.NONE,
        covers=frozenset({"cpu_ram"}),
        notes=("available_vram_bytes stays None: there is no separate VRAM pool.",),
    ),
    Scenario(
        name="unified_memory_too_large",
        question="Unified memory: the shared pool is simply too small.",
        machine=UNIFIED_16GB,
        weights_bytes=30 * GIB,
        kv_cache_bytes=4 * GIB,
        overhead_bytes=3 * GIB,
        safety_margin_bytes=3 * GIB,
        total_transformer_blocks=64,
        expected_mode=HardwareFitMode.TOO_LARGE,
        expected_placement=HardwareFitPlacementMethod.NONE,
        covers=frozenset({"too_large"}),
    ),
    Scenario(
        name="unified_memory_charges_device_reserve_to_the_pool",
        question="Unified memory with a device reserve: who pays for it?",
        machine=UNIFIED_16GB,
        weights_bytes=6 * GIB,
        kv_cache_bytes=1 * GIB,
        overhead_bytes=1 * GIB,
        device_reserve_bytes=1 * GIB,
        safety_margin_bytes=512 * MIB,
        total_transformer_blocks=32,
        expected_mode=HardwareFitMode.CPU_RAM,
        expected_placement=HardwareFitPlacementMethod.NONE,
        covers=frozenset({"cpu_ram", "unified_reserve"}),
        notes=(
            "The shared pool does. Identical to unified_memory_uses_one_shared_"
            "pool except for the 1 GiB reserve, and ram_required_bytes is 1 GiB "
            "higher as a result. On discrete hardware the reserve is VRAM and a "
            "CPU_RAM placement never pays it; here the accelerator draws from "
            "the same pool as the CPU, so it does.",
        ),
    ),
)

# Scenario pairs that isolate one variable. The tests assert both the mode
# change and that the pair really differs in that single field, so a fixture
# edit cannot turn a controlled comparison into a coincidence.
CONTROLLED_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "kv_small_keeps_the_model_on_the_gpu",
        "kv_large_pushes_the_model_off_the_gpu",
        "kv_cache_bytes",
    ),
    (
        "single_user_kv_fits_on_the_gpu",
        "four_concurrent_users_force_offload",
        "kv_cache_bytes",
    ),
    (
        "transformer_block_placement_with_metadata",
        "byte_fallback_without_transformer_block_metadata",
        "total_transformer_blocks",
    ),
    (
        "unified_memory_uses_one_shared_pool",
        "unified_memory_charges_device_reserve_to_the_pool",
        "device_reserve_bytes",
    ),
)

BY_NAME: dict[str, Scenario] = {item.name: item for item in SCENARIOS}


def scenario(name: str) -> Scenario:
    return BY_NAME[name]


__all__ = [
    "BY_NAME",
    "CONTROLLED_PAIRS",
    "GIB",
    "MIB",
    "OBSERVED_FIELDS",
    "REQUIRED_COVERAGE",
    "SCENARIOS",
    "Machine",
    "Scenario",
    "scenario",
]
