"""General-purpose data ingestion for the course packs: one loader family over
.csv/.tsv/.json/.npz/.npy plus the engine-dump convenience (.ply -> axial profile).

Contract: every failure raises :class:`TabularError` whose message states what WAS
found (columns, keys, shapes) — toolset modules pass ``str(exc)`` straight into
``ToolError.hint``. Arrays returned are float64 and C-contiguous. Size guards are
MCP-friendly module constants (tests monkeypatch them small).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_FILE_BYTES: int = 64 * 2**20  # pre-parse stat() guard — genfromtxt on huge CSV blows MCP timeouts
MAX_CELLS: int = 5_000_000
SUPPORTED_SUFFIXES = (".csv", ".tsv", ".json", ".npz", ".npy", ".ply")

# Reserved .npz member carrying the sample spacing of a series (meters or seconds).
DT_KEY = "dt"


class TabularError(ValueError):
    """Single exception type all loaders raise; message is hint-ready."""


@dataclass
class SeriesLoad:
    """One 1-D series plus the metadata packs need for honest reporting."""

    y: np.ndarray  # (n,) float64
    dt: float | None  # sample spacing when the format carries one (.npz 'dt', .ply profile)
    n_dropped: int  # non-finite values removed (0 when dropna=False)
    column: str  # resolved column/key name
    source: str  # str(path)
    extras: dict[str, np.ndarray] = field(default_factory=dict)  # e.g. .ply t-axis


def _check_size(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise TabularError(f"cannot stat '{path}': {exc}") from exc
    if size > MAX_FILE_BYTES:
        raise TabularError(
            f"file is {size / 2**20:.0f} MB — limit {MAX_FILE_BYTES / 2**20:.0f} MB; "
            "downsample or split the file"
        )


def _check_cells(rows: int, cols: int) -> None:
    if rows * cols > MAX_CELLS:
        raise TabularError(
            f"table is {rows} x {cols} = {rows * cols:.2e} cells — limit {MAX_CELLS:.0e}; "
            "pass columns= to select fewer, or downsample the file"
        )


def _suffix(path: Path) -> str:
    sfx = path.suffix.lower()
    if sfx not in SUPPORTED_SUFFIXES:
        raise TabularError(
            f"unsupported table format '{sfx or path.name}' — supported: " + ", ".join(SUPPORTED_SUFFIXES)
        )
    return sfx


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _load_csv(path: Path, delimiter: str) -> dict[str, np.ndarray]:
    _check_size(path)
    with path.open(encoding="utf-8", errors="replace") as fh:
        first = ""
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first = stripped
                break
    if not first:
        raise TabularError(f"no numeric rows parsed from {path}")
    tokens = [t.strip() for t in first.split(delimiter)]
    has_header = not all(_is_float(t) for t in tokens if t != "")
    kwargs: dict = {"delimiter": delimiter, "dtype": np.float64, "encoding": "utf-8"}
    if has_header:
        arr = np.genfromtxt(path, names=True, deletechars="", **kwargs)
        names = list(arr.dtype.names or [])
        if not names:
            raise TabularError(f"no numeric rows parsed from {path}")
        rows = int(arr.shape[0]) if arr.ndim else 1
        _check_cells(rows, len(names))
        atleast = np.atleast_1d(arr)
        return {name: np.ascontiguousarray(atleast[name], dtype=np.float64) for name in names}
    arr = np.genfromtxt(path, **kwargs)
    arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    if arr.size == 0:
        raise TabularError(f"no numeric rows parsed from {path}")
    if arr.shape[0] == 1 and arr.shape[1] > 1 and first.count(delimiter) == 0:
        arr = arr.T  # single column written one value per line
    _check_cells(arr.shape[0], arr.shape[1])
    return {f"col{i}": np.ascontiguousarray(arr[:, i]) for i in range(arr.shape[1])}


def _coerce_numeric_list(values: list, colname: str) -> np.ndarray | None:
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            out[i] = np.nan
        elif isinstance(v, bool):
            out[i] = float(v)
        elif isinstance(v, (int, float)):
            out[i] = float(v)
        else:
            return None  # non-numeric column
    return out


def _load_json(path: Path) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """Returns (numeric table, raw non-numeric columns kept for load_labels)."""
    _check_size(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TabularError(f"invalid JSON in {path}: {exc}") from exc

    numeric: dict[str, np.ndarray] = {}
    raw: dict[str, list] = {}
    if (
        isinstance(payload, list)
        and payload
        and all(isinstance(v, (int, float, bool)) or v is None for v in payload)
    ):
        arr = _coerce_numeric_list(payload, "value")
        assert arr is not None
        numeric["value"] = arr
    elif isinstance(payload, list) and payload and all(isinstance(v, list) for v in payload):
        # list-of-lists -> matrix with synthetic column names (engmath matrices)
        widths = {len(row) for row in payload}
        if len(widths) != 1:
            raise TabularError(f"matrix rows have unequal lengths: {sorted(widths)}")
        width = widths.pop()
        _check_cells(len(payload), width)
        try:
            mat = np.asarray(payload, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise TabularError(f"matrix has a non-numeric cell: {exc}") from exc
        numeric = {f"col{i}": np.ascontiguousarray(mat[:, i]) for i in range(width)}
    elif isinstance(payload, list) and payload and all(isinstance(v, dict) for v in payload):
        keys: list[str] = []
        for rec in payload:
            for k in rec:
                if k not in keys:
                    keys.append(k)
        _check_cells(len(payload), max(len(keys), 1))
        for k in keys:
            values = [rec.get(k) for rec in payload]
            arr = _coerce_numeric_list(values, k)
            if arr is not None:
                numeric[k] = arr
            else:
                raw[k] = values
    elif isinstance(payload, dict) and payload and all(isinstance(v, list) for v in payload.values()):
        lengths = {k: len(v) for k, v in payload.items()}
        if len(set(lengths.values())) != 1:
            raise TabularError(f"column lists have unequal lengths: {lengths}")
        _check_cells(next(iter(lengths.values())), len(payload))
        for k, values in payload.items():
            arr = _coerce_numeric_list(values, k)
            if arr is not None:
                numeric[k] = arr
            else:
                raw[k] = values
    else:
        raise TabularError(
            "JSON must be a list of numbers, a list of equal-length numeric lists, a list of "
            f"flat objects, or an object of column lists — got {type(payload).__name__}"
        )
    if not numeric and not raw:
        raise TabularError(f"no numeric rows parsed from {path}")
    return numeric, raw


def _npz_members(path: Path) -> np.lib.npyio.NpzFile:
    _check_size(path)
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TabularError(f"cannot read npz '{path}': {exc}") from exc


def _npz_pick(z: np.lib.npyio.NpzFile, key: str | None) -> tuple[str, np.ndarray]:
    keys = [k for k in z.files if k != DT_KEY]
    if key is not None:
        if key not in z.files:
            raise TabularError(f"key '{key}' not in archive — keys present: {z.files}")
        picked = key
    elif len(keys) == 1:
        picked = keys[0]
    else:
        raise TabularError(f"archive has {len(keys)} arrays — pass key= one of {keys}")
    arr = z[picked]
    if arr.size > MAX_CELLS:
        raise TabularError(f"array '{picked}' has {arr.size:.2e} elements — limit {MAX_CELLS:.0e}")
    return picked, np.asarray(arr, dtype=np.float64)


def _npz_dt(z: np.lib.npyio.NpzFile) -> float | None:
    if DT_KEY in z.files:
        try:
            return float(np.asarray(z[DT_KEY]).reshape(-1)[0])
        except (ValueError, IndexError):
            return None
    return None


def _npz_named_matrix(z: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str]] | None:
    """Feature-matrix convention: members 'X' + 'feature_names'."""
    if "X" in z.files and "feature_names" in z.files:
        x = np.asarray(z["X"], dtype=np.float64)
        names = [str(n) for n in z["feature_names"]]
        if x.ndim == 2 and len(names) == x.shape[1]:
            _check_cells(x.shape[0], x.shape[1])
            return np.ascontiguousarray(x), names
    return None


def _parse_ply_column(column: str | int | None) -> tuple[str, str]:
    joint, channel = "armLowerL", "posed"
    if column is not None and str(column):
        parts = str(column).split(":", 1)
        if parts[0]:
            joint = parts[0]
        if len(parts) == 2 and parts[1]:
            channel = parts[1]
    if channel not in ("posed", "rest"):
        raise TabularError(f"ply channel must be posed|rest, got '{channel}'")
    return joint, channel


def _load_ply_series(path: Path, column: str | int | None, cache_dir: Path | None) -> SeriesLoad:
    from dsp_server.engine import ply, transforms  # lazy: keep tabular numpy-only otherwise

    joint, channel = _parse_ply_column(column)
    dump = ply.load_dump_cached(path, cache_dir)
    bone_map = ply.bone_map_for(path)
    try:
        forearm = transforms.extract_forearm_signal(
            dump, joint=joint, bone_map=bone_map or None, channel=channel
        )
    except (ValueError, KeyError) as exc:
        raise TabularError(str(exc)) from exc
    sig = forearm.signal
    t_axis = sig.t0 + np.arange(sig.y.size) * sig.dt
    return SeriesLoad(
        y=np.ascontiguousarray(sig.y, dtype=np.float64),
        dt=float(sig.dt),
        n_dropped=0,
        column=f"{joint}:{channel}",
        source=str(path),
        extras={"t": t_axis},
    )


def _resolve_column(table: dict[str, np.ndarray], column: str | int | None, context: str) -> str:
    names = list(table.keys())
    if column is None:
        if len(names) == 1:
            return names[0]
        raise TabularError(
            f"table has {len(names)} columns — pass column= one of {names} "
            f"(name or index 0..{len(names) - 1})"
        )
    if isinstance(column, str) and column in table:
        return column
    if isinstance(column, str):
        ci = [n for n in names if n.lower() == column.lower()]
        if len(ci) == 1:
            return ci[0]
    if isinstance(column, int) or (isinstance(column, str) and column.isdigit()):
        idx = int(column)
        if 0 <= idx < len(names):
            return names[idx]
        raise TabularError(
            f"column index {idx} out of range — columns present: {names} (0..{len(names) - 1})"
        )
    raise TabularError(
        f"column '{column}' not found — {context} columns present: {names} "
        f"(pass a name or an integer index 0..{len(names) - 1})"
    )


def load_table(
    path: str | Path,
    columns: list[str] | None = None,
    key: str | None = None,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Insertion-ordered dict of equal-length float64 columns (NaN preserved —
    use :func:`drop_nan_rows` and report the count)."""
    path = Path(path)
    sfx = _suffix(path)
    if sfx in (".csv", ".tsv"):
        table = _load_csv(path, "," if sfx == ".csv" else "\t")
    elif sfx == ".json":
        table, _raw = _load_json(path)
        if not table:
            raise TabularError(
                f"no numeric columns in {path} — non-numeric columns: {list(_raw)} "
                "(use load_labels for string columns)"
            )
    elif sfx == ".npz":
        with _npz_members(path) as z:
            named = _npz_named_matrix(z)
            if named is not None:
                x, names = named
                table = {n: np.ascontiguousarray(x[:, i]) for i, n in enumerate(names)}
            else:
                picked, arr = _npz_pick(z, key)
                if arr.ndim == 1:
                    table = {picked: np.ascontiguousarray(arr)}
                elif arr.ndim == 2:
                    _check_cells(arr.shape[0], arr.shape[1])
                    table = {f"col{i}": np.ascontiguousarray(arr[:, i]) for i in range(arr.shape[1])}
                else:
                    raise TabularError(f"array '{picked}' has shape {arr.shape} — need 1-D or 2-D")
    elif sfx == ".npy":
        if key is not None:
            raise TabularError("'.npy' holds one unnamed array — key= is only for .npz")
        _check_size(path)
        arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if arr.ndim == 1:
            table = {"value": np.ascontiguousarray(arr)}
        elif arr.ndim == 2:
            _check_cells(arr.shape[0], arr.shape[1])
            table = {f"col{i}": np.ascontiguousarray(arr[:, i]) for i in range(arr.shape[1])}
        else:
            raise TabularError(f"array has shape {arr.shape} — need 1-D or 2-D")
    elif sfx == ".ply":
        series = _load_ply_series(path, columns[0] if columns else None, cache_dir)
        table = {"t": series.extras["t"], "r": series.y}
        return table
    else:  # pragma: no cover — _suffix already guards
        raise TabularError(f"unsupported table format '{sfx}'")

    if columns is not None:
        out: dict[str, np.ndarray] = {}
        for name in columns:
            resolved = _resolve_column(table, name, "requested")
            out[resolved] = table[resolved]
        return out
    return table


def load_series(
    path: str | Path,
    column: str | int | None = None,
    key: str | None = None,
    dropna: bool = True,
    cache_dir: Path | None = None,
) -> SeriesLoad:
    """One numeric series from any supported file. ``dt`` is populated only when
    the format carries it (.npz reserved key 'dt', .ply profile); dt-consuming
    tools must require an explicit dt when it is None."""
    path = Path(path)
    sfx = _suffix(path)
    if sfx == ".ply":
        return _load_ply_series(path, column, cache_dir)

    dt: float | None = None
    if sfx == ".npz":
        with _npz_members(path) as z:
            dt = _npz_dt(z)
            named = _npz_named_matrix(z)
            if named is not None:
                x, names = named
                table = {n: np.ascontiguousarray(x[:, i]) for i, n in enumerate(names)}
                colname = _resolve_column(table, column, "matrix")
                y = table[colname]
            else:
                picked, arr = _npz_pick(z, key)
                if arr.ndim == 2:
                    is_index = isinstance(column, int) or (isinstance(column, str) and column.isdigit())
                    if not is_index:  # check BEFORE int(column) so a name gives the friendly error
                        raise TabularError(
                            f"array has shape {arr.shape} — pass column= an index 0..{arr.shape[1] - 1}"
                        )
                    idx = int(column)
                    if not 0 <= idx < arr.shape[1]:
                        raise TabularError(
                            f"array has shape {arr.shape} — pass column= an index 0..{arr.shape[1] - 1}"
                        )
                    y = np.ascontiguousarray(arr[:, idx])
                    colname = f"col{idx}"
                elif arr.ndim == 1:
                    y = np.ascontiguousarray(arr)
                    colname = picked
                else:
                    raise TabularError(f"array '{picked}' has shape {arr.shape} — need 1-D or 2-D")
    else:
        table = load_table(path, key=key)
        colname = _resolve_column(table, column, "file")
        y = table[colname]

    n_dropped = 0
    if dropna:
        finite = np.isfinite(y)
        n_dropped = int(y.size - finite.sum())
        y = y[finite]
        if y.size < 3:
            raise TabularError(
                f"only {y.size} finite values in '{colname}' after dropping NaN/inf — need >= 3"
            )
    return SeriesLoad(
        y=np.ascontiguousarray(y, dtype=np.float64),
        dt=dt,
        n_dropped=n_dropped,
        column=str(colname),
        source=str(path),
    )


def load_matrix(
    path: str | Path,
    columns: list[str] | None = None,
    key: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """(n, d) float64 design matrix + column names (ml/engmath consumption)."""
    path = Path(path)
    if _suffix(path) == ".ply":
        raise TabularError(
            "'.ply' is a mesh dump, not a feature matrix — run feature_engineer_dump first "
            "and load the resulting .npz"
        )
    table = load_table(path, columns=columns, key=key, cache_dir=cache_dir)
    names = list(table.keys())
    x = np.column_stack([table[n] for n in names]) if names else np.empty((0, 0))
    if x.size == 0:
        raise TabularError(f"no numeric rows parsed from {path}")
    return np.ascontiguousarray(x, dtype=np.float64), names


def load_labels(
    path: str | Path,
    column: str | int,
    key: str | None = None,
) -> np.ndarray:
    """Label column: float64 when numeric, else str array — never coerced."""
    path = Path(path)
    sfx = _suffix(path)
    if sfx == ".json":
        numeric, raw = _load_json(path)
        if isinstance(column, str) and column in raw:
            return np.asarray([str(v) for v in raw[column]])
        if isinstance(column, str) and column in numeric:
            return numeric[column]
        resolved = _resolve_column({**numeric, **{k: np.zeros(1) for k in raw}}, column, "label")
        if resolved in raw:
            return np.asarray([str(v) for v in raw[resolved]])
        return numeric[resolved]
    if sfx == ".ply":
        raise TabularError("'.ply' has no label columns — use feature_engineer_dump")
    table = load_table(path, key=key)
    resolved = _resolve_column(table, column, "label")
    return table[resolved]


def drop_nan_rows(table: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    """Jointly drop rows where ANY column is non-finite; returns (table, n_dropped)."""
    if not table:
        return table, 0
    n = next(iter(table.values())).size
    keep = np.ones(n, dtype=bool)
    for arr in table.values():
        keep &= np.isfinite(arr)
    dropped = int(n - keep.sum())
    if dropped == 0:
        return table, 0
    return {k: v[keep] for k, v in table.items()}, dropped


def columns_of(path: str | Path, key: str | None = None) -> list[str]:
    """Cheap column introspection (powers ToolError hints)."""
    path = Path(path)
    sfx = _suffix(path)
    if sfx == ".ply":
        return ["<joint>[:posed|rest] e.g. armLowerL:posed"]
    if sfx == ".json":
        numeric, raw = _load_json(path)
        return list(numeric) + [f"{k} (non-numeric)" for k in raw]
    return list(load_table(path, key=key).keys())
