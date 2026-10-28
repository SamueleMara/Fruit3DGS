#!/usr/bin/env python3
"""Plain DBSCAN ablation for trained Gaussian PLYs.

This intentionally skips the instance-embedding path in filter_standalone.py.
It clusters foreground Gaussians directly in XYZ space using sklearn DBSCAN,
then writes a PLY with a cluster_id field and a second _no_bkg PLY.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

DBSCAN = None


def _load_runtime_deps() -> None:
    global DBSCAN
    if DBSCAN is not None:
        return
    try:
        from sklearn.cluster import DBSCAN as _DBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "direct_dbscan_ablation.py requires numpy and scikit-learn. "
            "Run it from the same environment used for the 3DGS project, or install "
            "the missing package shown in the original error."
        ) from exc
    DBSCAN = _DBSCAN


def _safe_float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _resolve_input_ply(model_dir: Path, ply_iteration: int, input_ply_name: str) -> Path:
    iter_dir = model_dir / "point_cloud" / f"iteration_{int(ply_iteration)}"
    if input_ply_name != "auto":
        path = iter_dir / input_ply_name
        if not path.exists():
            raise FileNotFoundError(f"Input PLY not found: {path}")
        return path

    candidates = [
        iter_dir / "scene_mask_filtered_renderer.ply",
        iter_dir / "point_cloud.ply",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find an input PLY. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


_PLY_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_DTYPE_TO_PLY = {
    ("i", 1): "char",
    ("u", 1): "uchar",
    ("i", 2): "short",
    ("u", 2): "ushort",
    ("i", 4): "int",
    ("u", 4): "uint",
    ("f", 4): "float",
    ("f", 8): "double",
}


def _read_ply_header(path: Path) -> tuple[list[str], int]:
    lines: list[str] = []
    header_bytes = 0
    with open(path, "rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"PLY header is incomplete: {path}")
            header_bytes += len(raw)
            line = raw.decode("ascii", errors="strict").strip()
            lines.append(line)
            if line == "end_header":
                break
    return lines, header_bytes


def _parse_vertex_dtype(header: list[str]) -> tuple[str, int, np.dtype]:
    ply_format = None
    vertex_count = None
    properties: list[tuple[str, str]] = []
    current_element = None

    for line in header:
        if not line or line.startswith("comment"):
            continue
        parts = line.split()
        if parts[0] == "format":
            ply_format = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("List properties in the vertex element are not supported by this ablation script.")
            ply_type = parts[1]
            name = parts[2]
            if ply_type not in _PLY_TO_DTYPE:
                raise ValueError(f"Unsupported PLY property type {ply_type!r} for field {name!r}")
            properties.append((name, ply_type))

    if ply_format is None:
        raise ValueError("PLY header does not define a format")
    if vertex_count is None:
        raise ValueError("PLY header does not define a vertex element")
    if not properties:
        raise ValueError("PLY vertex element has no scalar properties")

    endian = "<" if ply_format == "binary_little_endian" else ">" if ply_format == "binary_big_endian" else None
    if endian is None:
        raise ValueError(f"Unsupported PLY format {ply_format!r}. Expected binary_little_endian or binary_big_endian.")

    dtype = np.dtype([(name, endian + _PLY_TO_DTYPE[ply_type]) for name, ply_type in properties])
    return ply_format, vertex_count, dtype


def _read_vertex_table(path: Path):
    header, header_bytes = _read_ply_header(path)
    _, vertex_count, dtype = _parse_vertex_dtype(header)
    with open(path, "rb") as f:
        f.seek(header_bytes)
        vertex = np.fromfile(f, dtype=dtype, count=vertex_count)
    if vertex.shape[0] != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices but read {vertex.shape[0]} from {path}")
    names = vertex.dtype.names or ()
    for field in ("x", "y", "z"):
        if field not in names:
            raise ValueError(f"PLY missing required field {field!r}: {path}")
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)
    return vertex, xyz


def _semantic_scores(vertex, field: str, use_sigmoid: bool, missing_policy: str) -> np.ndarray | None:
    names = vertex.dtype.names or ()
    if field not in names:
        if missing_policy == "fail":
            raise ValueError(f"PLY has no semantic field {field!r}")
        return None
    sem = np.asarray(vertex[field], dtype=np.float64).reshape(-1)
    if use_sigmoid:
        sem = 1.0 / (1.0 + np.exp(-sem))
    return sem


def _normalize_xyz(xyz: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "none":
        return xyz.copy(), {"mode": mode, "scale": 1.0}

    center = np.median(xyz, axis=0)
    shifted = xyz - center[None, :]

    if mode == "unit_sphere":
        dist = np.linalg.norm(shifted, axis=1)
        scale = float(np.percentile(dist, 95))
    elif mode == "bbox_diag":
        lo = np.percentile(xyz, 1, axis=0)
        hi = np.percentile(xyz, 99, axis=0)
        scale = float(np.linalg.norm(hi - lo))
    else:
        raise ValueError(f"Unknown xyz normalization mode: {mode}")

    if not math.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return shifted / scale, {
        "mode": mode,
        "center": center.tolist(),
        "scale": scale,
    }


def _dtype_with_cluster(vertex, labels: np.ndarray):
    names = vertex.dtype.names or ()
    if "cluster_id" in names:
        out = np.empty(vertex.shape, dtype=vertex.dtype)
        for name in names:
            out[name] = vertex[name]
        out["cluster_id"] = labels.astype(np.int32)
        return out

    dtype = list(vertex.dtype.descr)
    dtype.append(("cluster_id", "<i4"))
    out = np.empty(vertex.shape, dtype=dtype)
    for name in names:
        out[name] = vertex[name]
    out["cluster_id"] = labels.astype(np.int32)
    return out


def _as_little_endian_dtype(dtype: np.dtype) -> np.dtype:
    fields = []
    for name in dtype.names or ():
        field_dtype = dtype.fields[name][0]
        if field_dtype.byteorder == ">" or (field_dtype.byteorder == "=" and sys.byteorder == "big"):
            field_dtype = field_dtype.newbyteorder("<")
        elif field_dtype.byteorder == "=" and sys.byteorder == "little":
            field_dtype = field_dtype.newbyteorder("<")
        fields.append((name, field_dtype))
    return np.dtype(fields)


def _ply_property_type(dtype: np.dtype) -> str:
    base = dtype.base
    key = (base.kind, base.itemsize)
    if key not in _DTYPE_TO_PLY:
        raise ValueError(f"Cannot write dtype {dtype} as a PLY scalar property")
    return _DTYPE_TO_PLY[key]


def _write_vertex_ply(path: Path, vertex_with_labels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype_le = _as_little_endian_dtype(vertex_with_labels.dtype)
    data = vertex_with_labels.astype(dtype_le, copy=False)

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {data.shape[0]}",
    ]
    for name in data.dtype.names or ():
        header.append(f"property {_ply_property_type(data.dtype.fields[name][0])} {name}")
    header.append("end_header")

    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        data.tofile(f)


def _summarize(labels: np.ndarray, total_points: int, fg_count: int) -> dict[str, Any]:
    labeled = labels >= 0
    cluster_ids = np.unique(labels[labeled])
    sizes = np.array([np.count_nonzero(labels == cid) for cid in cluster_ids], dtype=np.int64)

    summary: dict[str, Any] = {
        "total_points": int(total_points),
        "foreground_points": int(fg_count),
        "clustered_points": int(labeled.sum()),
        "noise_points": int((labels < 0).sum()),
        "num_clusters": int(cluster_ids.size),
        "noise_fraction_total": float((labels < 0).sum() / max(total_points, 1)),
        "noise_fraction_foreground": float((fg_count - labeled.sum()) / max(fg_count, 1)),
    }
    if sizes.size:
        summary.update(
            {
                "cluster_size_min": int(sizes.min()),
                "cluster_size_p50": float(np.percentile(sizes, 50)),
                "cluster_size_p90": float(np.percentile(sizes, 90)),
                "cluster_size_p95": float(np.percentile(sizes, 95)),
                "cluster_size_max": int(sizes.max()),
            }
        )
    else:
        summary.update(
            {
                "cluster_size_min": 0,
                "cluster_size_p50": 0.0,
                "cluster_size_p90": 0.0,
                "cluster_size_p95": 0.0,
                "cluster_size_max": 0,
            }
        )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    _load_runtime_deps()

    model_dir = Path(args.model_dir).resolve()
    input_ply = _resolve_input_ply(model_dir, args.ply_iteration, args.input_ply_name)

    t0 = time.perf_counter()
    vertex, xyz = _read_vertex_table(input_ply)
    sem = _semantic_scores(
        vertex,
        field=args.semantic_field,
        use_sigmoid=args.semantic_use_sigmoid,
        missing_policy=args.semantic_missing_policy,
    )

    if args.semantic_threshold is None or sem is None:
        fg_mask = np.ones((xyz.shape[0],), dtype=bool)
    else:
        fg_mask = sem > float(args.semantic_threshold)

    fg_idx = np.flatnonzero(fg_mask)
    labels = -np.ones((xyz.shape[0],), dtype=np.int32)

    xyz_work, norm_info = _normalize_xyz(xyz, args.xyz_normalize)
    if fg_idx.size > 0:
        clusterer = DBSCAN(
            eps=float(args.eps),
            min_samples=int(args.min_samples),
            metric="euclidean",
            algorithm=args.algorithm,
            n_jobs=int(args.n_jobs),
        )
        labels_fg = clusterer.fit_predict(xyz_work[fg_idx]).astype(np.int32)
        labels[fg_idx] = labels_fg

    elapsed = time.perf_counter() - t0

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_ply.parent / "direct_dbscan_ablation"
    if args.output_prefix == "auto":
        output_prefix = (
            f"direct_dbscan_{Path(input_ply).stem}"
            f"_eps{_safe_float_tag(float(args.eps))}"
            f"_ms{int(args.min_samples)}"
            f"_norm{args.xyz_normalize}"
        )
    else:
        output_prefix = args.output_prefix

    full_ply = output_dir / f"{output_prefix}.ply"
    no_bkg_ply = output_dir / f"{output_prefix}_no_bkg.ply"
    metrics_path = Path(args.metrics_path).resolve() if args.metrics_path else output_dir / f"{output_prefix}_metrics.json"

    vertex_with_labels = _dtype_with_cluster(vertex, labels)
    _write_vertex_ply(full_ply, vertex_with_labels)
    _write_vertex_ply(no_bkg_ply, vertex_with_labels[labels >= 0])

    metrics = {
        "input_ply": str(input_ply),
        "output_ply": str(full_ply),
        "output_no_bkg_ply": str(no_bkg_ply),
        "metrics_path": str(metrics_path),
        "eps": float(args.eps),
        "min_samples": int(args.min_samples),
        "algorithm": args.algorithm,
        "xyz_normalize": norm_info,
        "semantic_field": args.semantic_field,
        "semantic_threshold": args.semantic_threshold,
        "semantic_use_sigmoid": bool(args.semantic_use_sigmoid),
        "elapsed_seconds": elapsed,
    }
    metrics.update(_summarize(labels, total_points=xyz.shape[0], fg_count=fg_idx.size))

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run direct XYZ DBSCAN on a trained Gaussian PLY.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--ply_iteration", type=int, default=30000)
    parser.add_argument("--input_ply_name", default="auto")
    parser.add_argument("--eps", type=float, required=True, help="DBSCAN radius. Interpreted after xyz normalization.")
    parser.add_argument("--min_samples", type=int, required=True)
    parser.add_argument("--xyz_normalize", choices=["none", "unit_sphere", "bbox_diag"], default="none")
    parser.add_argument("--semantic_field", default="semantics")
    parser.add_argument("--semantic_threshold", type=float, default=None)
    parser.add_argument("--semantic_use_sigmoid", action="store_true")
    parser.add_argument("--semantic_missing_policy", choices=["all", "fail"], default="all")
    parser.add_argument("--algorithm", choices=["auto", "ball_tree", "kd_tree", "brute"], default="auto")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_prefix", default="auto")
    parser.add_argument("--metrics_path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
