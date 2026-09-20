"""The layering rules, enforced.

These are not style tests. Each one prevents a specific failure we measured in
the previous system:

- ``test_layer_dependencies`` prevents the 738-module import cascade: importing
  the momentum strategy from ``regime-trader`` dragged in hmmlearn, scipy and
  sklearn because the strategy lived in the same file as an HMM allocator. That
  is not a performance problem, it is the signal that layers are fused and can no
  longer be replaced independently.
- ``test_no_module_level_mutable_state`` prevents implicit context. A module-level
  dict or list is shared by every caller in the process, which breaks the moment
  two portfolios, two strategies or two runs exist at once.
- ``test_contracts_are_frozen`` prevents accidental mutation of values that cross
  layer boundaries.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Every internal package, and exactly which other internal packages it may import.
#: Read it as "this layer is allowed to know about these layers, and nothing else".
ALLOWED: dict[str, set[str]] = {
    # The centre of the onion. Depends on nothing, so everything can depend on it.
    "contracts": set(),
    # Seams: each implements a port and knows only the port's vocabulary.
    "access": {"contracts"},
    "data": {"contracts"},
    "execution": {"contracts"},
    # Strategies are the whole point of the layering: they must stay swappable,
    # so they see the contracts and nothing else. No engine, no risk, no peers.
    "strategies": {"contracts"},
    # Machinery. Knows the contracts and the layers it orchestrates.
    "engine": {"contracts", "data"},
    "risk": {"contracts"},
    "validation": {"contracts", "engine"},
    "reports": {"contracts", "validation"},
    # The composition root. Everything may be imported BY it, it is imported by
    # nothing.
    "runtime": {
        "contracts", "access", "data", "strategies",
        "engine", "risk", "validation", "execution", "reports",
    },
}

PACKAGES = sorted(ALLOWED)


def python_files() -> list[Path]:
    """Every source file in an internal package."""
    files: list[Path] = []
    for package in PACKAGES:
        files.extend(sorted((ROOT / package).rglob("*.py")))
    return files


def internal_imports(path: Path) -> set[str]:
    """Internal top-level packages that ``path`` imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                if head in ALLOWED:
                    found.add(head)
        # level > 0 is a relative import: same package, always fine.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            head = node.module.split(".")[0]
            if head in ALLOWED:
                found.add(head)
    return found


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_layer_dependencies(path: Path) -> None:
    """No module imports a layer its own layer is not allowed to know about."""
    package = path.relative_to(ROOT).parts[0]
    permitted = ALLOWED[package] | {package}
    violations = internal_imports(path) - permitted
    assert not violations, (
        f"{path.relative_to(ROOT)} (layer '{package}') imports {sorted(violations)}, "
        f"which is outside its allowance {sorted(ALLOWED[package])}. "
        f"Either the import is wrong or ALLOWED needs a deliberate change."
    )


def test_nothing_imports_the_composition_root() -> None:
    """``runtime`` wires everything together, so nothing may depend on it."""
    offenders = [
        str(path.relative_to(ROOT))
        for path in python_files()
        if path.relative_to(ROOT).parts[0] != "runtime" and "runtime" in internal_imports(path)
    ]
    assert not offenders, f"these modules import runtime/: {offenders}"


MUTABLE_CALLS = ("list", "dict", "set", "defaultdict", "Counter", "deque")
MUTATING_METHODS = (
    "append", "extend", "insert", "remove", "pop", "clear",
    "update", "setdefault", "add", "discard", "sort", "popitem",
)


def _is_mutable_literal(node: ast.expr) -> bool:
    if isinstance(node, (ast.List, ast.Dict, ast.Set)):
        return True
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in MUTABLE_CALLS
    )


def _is_frozen_table(node: ast.expr) -> bool:
    """A non-empty literal container: a lookup table, not an accumulator.

    Emptiness is the tell. ``STATUSES = {"a": 1, "b": 2}`` is data the module was
    born with; ``REGISTRY = {}`` is a hole something fills later, and naming it in
    capitals does not change that.
    """
    if isinstance(node, (ast.List, ast.Set)):
        return bool(node.elts)
    if isinstance(node, ast.Dict):
        return bool(node.keys)
    return False


def _mutated_names(tree: ast.Module) -> set[str]:
    """Names written through anywhere in the file, at any nesting depth."""
    mutated: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in MUTATING_METHODS
                and isinstance(func.value, ast.Name)
            ):
                mutated.add(func.value.id)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    mutated.add(target.value.id)
                elif isinstance(target, ast.AugAssign):  # pragma: no cover - defensive
                    pass
    return mutated


def mutable_module_state(path: Path) -> list[str]:
    """Names bound to mutable values at module scope, exemptions applied."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mutated = _mutated_names(tree)
    offenders: list[str] = []
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None or not _is_mutable_literal(value):
            continue
        for target in targets:
            name = getattr(target, "id", "")
            if not name:
                continue
            exempt = name.isupper() and _is_frozen_table(value) and name not in mutated
            if not exempt:
                offenders.append(name)
    return offenders


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_module_level_mutable_state(path: Path) -> None:
    """No mutable value at module scope.

    Frozen constants (tuples, frozensets, strings, numbers) are fine — they cannot
    carry state between callers. A module-level list or dict can, and that is how a
    system quietly acquires a single implicit user, a single active portfolio or a
    single 'current' anything.

    An ``UPPER_CASE`` name is exempt only when it is a non-empty literal table that
    nothing in the module writes to. Capitals alone are not an exemption: an empty
    ``REGISTRY = {}`` is a singleton waiting to happen, whatever it is called.
    """
    offenders = mutable_module_state(path)
    assert not offenders, (
        f"{path.relative_to(ROOT)} has mutable module-level state: {offenders}. "
        f"Pass it explicitly instead, or make it a frozen UPPER_CASE table "
        f"(a non-empty literal nothing writes to)."
    )


def test_every_package_is_declared() -> None:
    """A new top-level package must be added to ALLOWED deliberately."""
    on_disk = {
        entry.name
        for entry in ROOT.iterdir()
        if entry.is_dir() and (entry / "__init__.py").exists() and not entry.name.startswith(".")
    }
    undeclared = on_disk - set(ALLOWED) - {"tests"}
    assert not undeclared, (
        f"packages on disk but not in ALLOWED: {sorted(undeclared)}. "
        f"Adding a layer is an architectural decision; declare its allowance."
    )


# ---------------------------------------------------------------------------
# The guard, guarded.
#
# Every test above passes today because the codebase is clean. That is also what
# a broken detector looks like. These tests feed known violations to the same
# functions the tests above use and require them to be caught, so the suite tells
# us whether the rule still bites rather than only that nothing has tripped it.
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_the_import_rule_catches_a_crossed_layer(tmp_path: Path) -> None:
    probe = _write(tmp_path, "from data.bitemporal import BitemporalStore\n")
    assert internal_imports(probe) == {"data"}
    # strategies/ is the layer we most need to keep isolated.
    assert "data" not in ALLOWED["strategies"]


def test_the_import_rule_sees_plain_and_relative_imports(tmp_path: Path) -> None:
    assert internal_imports(_write(tmp_path, "import engine.backtest\n")) == {"engine"}
    assert internal_imports(_write(tmp_path, "import pandas as pd\n")) == set()
    # A relative import cannot cross a package boundary, so it is not a violation.
    assert internal_imports(_write(tmp_path, "from .sibling import thing\n")) == set()


@pytest.mark.parametrize(
    "source, caught",
    [
        ("registry = {}\n", ["registry"]),
        ("cache: dict[str, int] = {}\n", ["cache"]),
        ("seen = set()\n", ["seen"]),
        ("buffer = list()\n", ["buffer"]),
        # Capitals are not a loophole: an empty table exists to be filled.
        ("REGISTRY = {}\n", ["REGISTRY"]),
        ("CACHE: dict[str, int] = dict()\n", ["CACHE"]),
        # Nor is a table that the module itself writes to.
        ('TABLE = {"a": 1}\n\n\ndef add(k):\n    TABLE[k] = 2\n', ["TABLE"]),
        ('TABLE = {"a": 1}\n\n\ndef add(k):\n    TABLE.update(k)\n', ["TABLE"]),
        # What stays allowed: a genuine frozen lookup table, and immutables.
        ('STATUSES = {"new": 1, "done": 2}\n', []),
        ("NAMES = (\"a\", \"b\")\n", []),
        ("LIMIT = 10\n", []),
        ("KEYS = frozenset({\"a\"})\n", []),
    ],
)
def test_the_state_rule_catches_mutable_module_scope(
    tmp_path: Path, source: str, caught: list[str]
) -> None:
    assert mutable_module_state(_write(tmp_path, source)) == caught
