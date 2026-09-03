from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
JAULL = SRC / "jaull"


@dataclass(frozen=True)
class DependencyRule:
    source_prefix: str
    forbidden_import_prefixes: tuple[str, ...]


@dataclass(frozen=True)
class ImportEdge:
    source: str
    imported: str


RULES = (
    DependencyRule(
        source_prefix="src/jaull/domain/",
        forbidden_import_prefixes=(
            "jaull.application",
            "jaull.adapters",
            "jaull.workflow",
            "jaull.tui",
            "jaull.cli",
            "jaull.presentation",
            "jaull.advisor",
            "jaull.runtime",
            "jaull.discovery",
            "textual",
            "huggingface_hub",
            "torch",
            "subprocess",
        ),
    ),
    DependencyRule(
        source_prefix="src/jaull/recommendation/",
        forbidden_import_prefixes=(
            "jaull.tui",
            "jaull.cli",
            "jaull.advisor",
            "jaull.workflow",
        ),
    ),
    DependencyRule(
        source_prefix="src/jaull/application/",
        forbidden_import_prefixes=(
            "jaull.adapters",
            "jaull.tui",
            "jaull.cli",
            "jaull.presentation",
            "jaull.advisor",
            "jaull.workflow",
            "jaull.huggingface",
            "huggingface_hub",
            "textual",
        ),
    ),
    DependencyRule(
        source_prefix="src/jaull/ports/",
        forbidden_import_prefixes=(
            "jaull.adapters",
            "jaull.tui",
            "jaull.cli",
            "jaull.presentation",
            "jaull.advisor",
            "jaull.workflow",
            "jaull.huggingface",
            "huggingface_hub",
            "textual",
        ),
    ),
    DependencyRule(
        source_prefix="src/jaull/execution_plans/",
        forbidden_import_prefixes=(
            "jaull.workflow",
            "jaull.recommendation",
        ),
    ),
    DependencyRule(
        source_prefix="src/jaull/presentation/",
        forbidden_import_prefixes=(
            "jaull.adapters",
            "jaull.huggingface",
            "textual",
            "huggingface_hub",
        ),
    ),
    DependencyRule(
        source_prefix="src/jaull/tui/",
        forbidden_import_prefixes=("jaull.huggingface", "huggingface_hub"),
    ),
)


LEGACY_IMPORT_ALLOWLIST: set[tuple[str, str]] = set()


def test_architecture_dependency_rules_do_not_gain_new_violations() -> None:
    forbidden_edges = _forbidden_edges(_python_import_edges())
    allowed_edges = {
        edge
        for edge in forbidden_edges
        if (edge.source, edge.imported) in LEGACY_IMPORT_ALLOWLIST
    }
    new_violations = sorted(forbidden_edges - allowed_edges, key=_edge_key)
    stale_allowlist = sorted(
        LEGACY_IMPORT_ALLOWLIST
        - {(edge.source, edge.imported) for edge in forbidden_edges}
    )

    assert not new_violations and not stale_allowlist, _failure_message(
        new_violations,
        stale_allowlist,
    )


def test_runtime_launch_policy_does_not_consume_hardware_fit_placement() -> None:
    """Runtime flags must not treat HFA transformer blocks as runtime layers."""

    violations = sorted(_hardware_fit_runtime_policy_violations())

    assert not violations, (
        "Runtime/execution-plan modules must not derive backend layer flags "
        "from HardwareFitResult or estimator.hardware_fit:\n"
        + "\n".join(f"- {violation}" for violation in violations)
    )


def test_hardware_fit_analyzer_does_not_depend_on_any_runtime() -> None:
    """The HardwareFitAnalyzer must not import or name any runtime backend concept.

    An AST check, not a text scan: a legitimate comment mentioning "llama.cpp" or
    "ngl" is fine; an import of ``jaull.runtime.*`` or a reference to
    ``RuntimeRecommendation`` / ``RuntimeFlag`` / ``RuntimeName`` is not.
    """
    path = JAULL / "estimator" / "hardware_fit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_names = {
        "RuntimeRecommendation",
        "RuntimeFlag",
        "RuntimeFlagSource",
        "RuntimeName",
        "pick_gpu_layers",
        "pick_device_map",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations += [
                f"import {alias.name}"
                for alias in node.names
                if _matches_prefix(alias.name, "jaull.runtime")
            ]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if _matches_prefix(node.module, "jaull.runtime"):
                violations.append(f"from {node.module} import ...")
            if node.module == "jaull.domain.runtime":
                violations += [
                    f"from jaull.domain.runtime import {alias.name}"
                    for alias in node.names
                ]
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            violations.append(f"name {node.id!r} at line {node.lineno}")

    assert not violations, "hardware_fit.py must stay runtime-agnostic:\n" + "\n".join(
        f"- {violation}" for violation in sorted(set(violations))
    )


def test_execution_frontends_do_not_construct_their_own_runtime_recommendation() -> None:
    """CLI run + the TUI execution screen must get the launch plan from the planner.

    Bare RuntimeRecommendations for ranking/display elsewhere (``engine_v2``,
    ``advisor._runtime_for_variant``) are out of scope and allowed; the two files
    that actually hand a runtime to a runner are not.
    """
    targets = [
        JAULL / "cli" / "run.py",
        JAULL / "tui" / "screens" / "recommendation_execution.py",
    ]
    violations: list[str] = []
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "RuntimeRecommendation"
            ):
                violations.append(f"{path.name}:{node.lineno}")

    assert not violations, (
        "execution frontends must not construct RuntimeRecommendation directly; "
        "route through AdvisorService.plan_execution: " + ", ".join(violations)
    )


def _python_import_edges() -> set[ImportEdge]:
    edges: set[ImportEdge] = set()
    for path in JAULL.rglob("*.py"):
        source = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                edges.update(ImportEdge(source, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                edges.update(_from_import_edges(source, node))
    return edges


def _hardware_fit_runtime_policy_violations() -> set[str]:
    violations: set[str] = set()
    for root in (JAULL / "runtime", JAULL / "execution_plans"):
        for path in root.rglob("*.py"):
            source = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=source)
            for node in ast.walk(tree):
                violation = _hardware_fit_runtime_policy_violation(source, node)
                if violation is not None:
                    violations.add(violation)
    return violations


def _hardware_fit_runtime_policy_violation(
    source: str,
    node: ast.AST,
) -> str | None:
    line = getattr(node, "lineno", "?")
    if isinstance(node, ast.Import):
        for alias in node.names:
            if _matches_prefix(alias.name, "jaull.estimator.hardware_fit"):
                return f"{source}:{line} imports {alias.name}"
        return None
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if _matches_prefix(module, "jaull.estimator.hardware_fit"):
            return f"{source}:{line} imports from {module}"
        if module == "jaull.estimator" and any(
            alias.name == "hardware_fit" for alias in node.names
        ):
            return f"{source}:{line} imports jaull.estimator.hardware_fit"
        if module == "jaull.domain.estimation" and any(
            alias.name
            in {
                "HardwareFitResult",
                "HardwareFitOffloadCandidate",
                "HardwareFitOffloadDiagnostics",
            }
            for alias in node.names
        ):
            return f"{source}:{line} imports HardwareFit placement diagnostics"
        return None
    if isinstance(node, ast.Attribute) and node.attr in {
        "hardware_fit",
        "gpu_transformer_blocks",
        "total_transformer_blocks",
        "offload_diagnostics",
    }:
        return f"{source}:{line} reads {node.attr}"
    return None


def _from_import_edges(source: str, node: ast.ImportFrom) -> set[ImportEdge]:
    assert node.module is not None
    edges: set[ImportEdge] = set()
    for alias in node.names:
        if alias.name == "*":
            edges.add(ImportEdge(source, node.module))
            continue
        imported = _resolve_imported_module(node.module, alias.name)
        edges.add(ImportEdge(source, imported))
    return edges


def _resolve_imported_module(module: str, name: str) -> str:
    candidate = f"{module}.{name}"
    return candidate if _module_exists(candidate) else module


def _module_exists(module: str) -> bool:
    if not module.startswith("jaull."):
        return False
    relative = Path(*module.split("."))
    return (SRC / relative).with_suffix(".py").exists() or (SRC / relative / "__init__.py").exists()


def _forbidden_edges(edges: set[ImportEdge]) -> set[ImportEdge]:
    forbidden: set[ImportEdge] = set()
    for edge in edges:
        for rule in RULES:
            if not edge.source.startswith(rule.source_prefix):
                continue
            if any(
                _matches_prefix(edge.imported, prefix)
                for prefix in rule.forbidden_import_prefixes
            ):
                forbidden.add(edge)
    return forbidden


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _edge_key(edge: ImportEdge) -> tuple[str, str]:
    return edge.source, edge.imported


def _failure_message(
    new_violations: list[ImportEdge],
    stale_allowlist: list[tuple[str, str]],
) -> str:
    sections: list[str] = []
    if new_violations:
        sections.append(
            "New architecture dependency violations:\n"
            + "\n".join(
                f"- {edge.source} imports {edge.imported}"
                for edge in new_violations
            )
        )
    if stale_allowlist:
        sections.append(
            "Stale architecture allowlist entries:\n"
            + "\n".join(
                f"- {source} imports {imported}"
                for source, imported in stale_allowlist
            )
        )
    return "\n\n".join(sections)
