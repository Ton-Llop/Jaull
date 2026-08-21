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
            "jaull.workflow",
            "jaull.tui",
            "jaull.cli",
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
        source_prefix="src/jaull/execution_plans/",
        forbidden_import_prefixes=(
            "jaull.workflow",
            "jaull.recommendation",
        ),
    ),
    DependencyRule(
        source_prefix="src/jaull/tui/",
        forbidden_import_prefixes=("jaull.huggingface",),
    ),
)


LEGACY_IMPORT_ALLOWLIST = {
    (
        "src/jaull/execution_plans/service.py",
        "jaull.workflow.policies",
    ),
    (
        "src/jaull/execution_plans/service.py",
        "jaull.workflow.model_analysis_cache",
    ),
    (
        "src/jaull/execution_plans/service.py",
        "jaull.workflow.telemetry",
    ),
    (
        "src/jaull/execution_plans/service.py",
        "jaull.recommendation.models",
    ),
    (
        "src/jaull/tui/screens/estimate.py",
        "jaull.huggingface.url_parser",
    ),
    (
        "src/jaull/tui/screens/inspect.py",
        "jaull.huggingface.url_parser",
    ),
}


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
