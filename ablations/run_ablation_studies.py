#!/usr/bin/env python3
"""Manifest-driven ablation runner for Fruit_GS.

Use this script to:
  1) sweep direct XYZ DBSCAN parameters over all datasets,
  2) run fixed direct DBSCAN parameters over all datasets,
  3) benchmark vanilla 3DGS vs train_sem + filter_standalone,
  4) optionally run contrastive-gaussian-clustering on the same datasets.

The runner writes logs, command metadata, resource samples, and summary CSV files
under the configured results_root.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("YAML manifest requested, but PyYAML is not installed. Use JSON or install PyYAML.")
            return yaml.safe_load(f)
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def sanitize_name(value: Any) -> str:
    s = str(value)
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch == ".":
            out.append("p")
        else:
            out.append("_")
    return "".join(out).strip("_")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def maybe_rel(root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


class ResourceSampler:
    def __init__(self, pid: int, interval: float = 2.0):
        self.pid = int(pid)
        self.interval = float(interval)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 2.0))
        return self.summary()

    def _process_stats(self) -> dict[str, Any]:
        if psutil is None:
            return {}
        try:
            root = psutil.Process(self.pid)
            procs = [root] + root.children(recursive=True)
            rss = 0
            cpu = 0.0
            for proc in procs:
                try:
                    rss += proc.memory_info().rss
                    cpu += proc.cpu_percent(interval=None)
                except Exception:
                    pass
            return {"rss_mb": rss / (1024 * 1024), "cpu_percent": cpu}
        except Exception:
            return {}

    def _gpu_stats(self) -> dict[str, Any]:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return {}
        if out.returncode != 0 or not out.stdout.strip():
            return {}
        utils = []
        mems = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    utils.append(float(parts[0]))
                    mems.append(float(parts[1]))
                except ValueError:
                    pass
        if not utils:
            return {}
        return {
            "gpu_util_percent_max_device": max(utils),
            "gpu_mem_mb_max_device": max(mems),
            "gpu_util_percent_mean_device": sum(utils) / len(utils),
        }

    def _run(self) -> None:
        if psutil is not None:
            try:
                root = psutil.Process(self.pid)
                root.cpu_percent(interval=None)
                for child in root.children(recursive=True):
                    child.cpu_percent(interval=None)
            except Exception:
                pass

        while not self._stop.is_set():
            sample = {"t": time.time()}
            sample.update(self._process_stats())
            sample.update(self._gpu_stats())
            self.samples.append(sample)
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"num_samples": len(self.samples)}
        for key in [
            "rss_mb",
            "cpu_percent",
            "gpu_util_percent_max_device",
            "gpu_mem_mb_max_device",
            "gpu_util_percent_mean_device",
        ]:
            vals = [float(s[key]) for s in self.samples if key in s]
            if vals:
                out[f"{key}_max"] = max(vals)
                out[f"{key}_mean"] = sum(vals) / len(vals)
        return out


def run_command(cmd: list[str], run_dir: Path, env: dict[str, str] | None = None, sample_interval: float = 2.0) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"
    meta_path = run_dir / "command.json"
    samples_path = run_dir / "resource_samples.json"
    summary_path = run_dir / "summary.json"

    save_json(meta_path, {"cmd": cmd, "cmd_string": " ".join(shlex.quote(c) for c in cmd), "cwd": os.getcwd()})

    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=merged_env,
        )
        sampler = ResourceSampler(proc.pid, interval=sample_interval)
        sampler.start()
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
        returncode = proc.wait()
        resources = sampler.stop()
    elapsed = time.perf_counter() - t0

    save_json(samples_path, sampler.samples)
    summary = {
        "returncode": int(returncode),
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
        "resource_summary": resources,
    }
    save_json(summary_path, summary)
    return summary


def iter_enabled_datasets(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = []
    for ds in manifest.get("datasets", []):
        if ds.get("enabled", True):
            datasets.append(ds)
    return datasets


def direct_dbscan_grid(direct_cfg: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode == "wide":
        cfg = direct_cfg.get("wide", {})
        eps_values = as_list(cfg.get("eps_values"))
        min_samples_values = as_list(cfg.get("min_samples_values"))
        sem_raw = cfg.get("semantic_thresholds", direct_cfg.get("semantic_threshold"))
        sem_values = [None] if sem_raw is None else as_list(sem_raw)
        norm_values = as_list(cfg.get("xyz_normalize", direct_cfg.get("xyz_normalize", "none")))
        combos = []
        for eps, ms, sem, norm in itertools.product(eps_values, min_samples_values, sem_values, norm_values):
            combos.append({"eps": eps, "min_samples": ms, "semantic_threshold": sem, "xyz_normalize": norm})
        return combos

    fixed = direct_cfg.get("fixed", [])
    if isinstance(fixed, dict):
        fixed = [fixed]
    return fixed


def resolve_direct_dbscan_model_dir(root: Path, results_root: Path, direct_cfg: dict[str, Any], ds: dict[str, Any]) -> str:
    source = direct_cfg.get("model_source", "manifest_model_dir")
    ds_name = sanitize_name(ds["name"])

    if source == "manifest_model_dir":
        model_dir = maybe_rel(root, ds.get("model_dir"))
        if model_dir is None:
            raise ValueError(f"Dataset {ds['name']} missing model_dir")
        return model_dir
    if source == "benchmark_full_pipeline_model":
        return str(results_root / "benchmark" / ds_name / "full_pipeline_model")
    if source == "benchmark_vanilla_model":
        return str(results_root / "benchmark" / ds_name / "vanilla_model")

    raise ValueError(
        f"Unknown direct_dbscan.model_source {source!r}. "
        "Use manifest_model_dir, benchmark_full_pipeline_model, or benchmark_vanilla_model."
    )


def run_direct_dbscan(manifest: dict[str, Any], mode: str) -> None:
    root = Path(manifest.get("repo_root", ".")).resolve()
    results_root = Path(maybe_rel(root, manifest.get("results_root", "ablation_runs"))).resolve()
    python = manifest.get("python", sys.executable)
    direct_cfg = manifest.get("direct_dbscan", {})
    sample_interval = float(manifest.get("sample_interval_seconds", 2.0))

    rows = []
    for ds in iter_enabled_datasets(manifest):
        ds_name = ds["name"]
        model_dir = resolve_direct_dbscan_model_dir(root, results_root, direct_cfg, ds)

        for combo in direct_dbscan_grid(direct_cfg, mode):
            eps = float(combo["eps"])
            min_samples = int(combo["min_samples"])
            sem = combo.get("semantic_threshold", direct_cfg.get("semantic_threshold"))
            norm = combo.get("xyz_normalize", direct_cfg.get("xyz_normalize", "none"))
            run_name = f"{sanitize_name(ds_name)}_eps{sanitize_name(eps)}_ms{min_samples}_norm{sanitize_name(norm)}"
            if sem is not None:
                run_name += f"_sem{sanitize_name(sem)}"

            run_dir = results_root / "direct_dbscan" / mode / sanitize_name(ds_name) / run_name
            metrics_path = run_dir / "metrics.json"
            output_dir = run_dir / "ply"

            cmd = [
                python,
                str(root / "ablations" / "direct_dbscan_ablation.py"),
                "--model_dir",
                model_dir,
                "--ply_iteration",
                str(direct_cfg.get("ply_iteration", 30000)),
                "--input_ply_name",
                str(direct_cfg.get("input_ply_name", "auto")),
                "--eps",
                str(eps),
                "--min_samples",
                str(min_samples),
                "--xyz_normalize",
                str(norm),
                "--semantic_field",
                str(direct_cfg.get("semantic_field", "semantics")),
                "--semantic_missing_policy",
                str(direct_cfg.get("semantic_missing_policy", "all")),
                "--algorithm",
                str(direct_cfg.get("algorithm", "auto")),
                "--n_jobs",
                str(direct_cfg.get("n_jobs", -1)),
                "--output_dir",
                str(output_dir),
                "--metrics_path",
                str(metrics_path),
            ]
            if sem is not None:
                cmd.extend(["--semantic_threshold", str(sem)])
            if direct_cfg.get("semantic_use_sigmoid", False):
                cmd.append("--semantic_use_sigmoid")

            summary = run_command(cmd, run_dir, sample_interval=sample_interval)
            metric_data: dict[str, Any] = {}
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metric_data = json.load(f)
            rows.append(
                {
                    "dataset": ds_name,
                    "mode": mode,
                    "eps": eps,
                    "min_samples": min_samples,
                    "semantic_threshold": sem,
                    "xyz_normalize": norm,
                    "returncode": summary["returncode"],
                    "elapsed_seconds": metric_data.get("elapsed_seconds", summary["elapsed_seconds"]),
                    "num_clusters": metric_data.get("num_clusters"),
                    "clustered_points": metric_data.get("clustered_points"),
                    "noise_fraction_foreground": metric_data.get("noise_fraction_foreground"),
                    "output_ply": metric_data.get("output_ply"),
                    "metrics_path": str(metrics_path),
                }
            )

    out_csv = results_root / "direct_dbscan" / f"{mode}_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "mode",
            "eps",
            "min_samples",
            "semantic_threshold",
            "xyz_normalize",
            "returncode",
            "elapsed_seconds",
            "num_clusters",
            "clustered_points",
            "noise_fraction_foreground",
            "output_ply",
            "metrics_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Direct DBSCAN {mode} summary: {out_csv}")


def build_common_train_args(cfg: dict[str, Any]) -> list[str]:
    args = []
    for key, value in cfg.items():
        if key in {"extra_args"}:
            continue
        cli_key = "--" + key
        if isinstance(value, bool):
            if value:
                args.append(cli_key)
        elif isinstance(value, list):
            args.append(cli_key)
            args.extend(str(v) for v in value)
        else:
            args.extend([cli_key, str(value)])
    args.extend(str(x) for x in cfg.get("extra_args", []))
    return args


def run_benchmarks(manifest: dict[str, Any]) -> None:
    root = Path(manifest.get("repo_root", ".")).resolve()
    results_root = Path(maybe_rel(root, manifest.get("results_root", "ablation_runs"))).resolve()
    python = manifest.get("python", sys.executable)
    sample_interval = float(manifest.get("sample_interval_seconds", 2.0))
    bench_cfg = manifest.get("benchmark", {})
    env = bench_cfg.get("env", {})
    rows = []

    for ds in iter_enabled_datasets(manifest):
        ds_name = sanitize_name(ds["name"])
        colmap_dir = maybe_rel(root, ds.get("colmap_dir"))
        sem_mask_dir = maybe_rel(root, ds.get("sem_mask_dir"))
        mask_inst_dir = maybe_rel(root, ds.get("mask_inst_dir"))
        if colmap_dir is None:
            raise ValueError(f"Dataset {ds['name']} missing colmap_dir")

        base_dir = results_root / "benchmark" / ds_name
        vanilla_model = base_dir / "vanilla_model"
        full_model = base_dir / "full_pipeline_model"

        if bench_cfg.get("run_vanilla", True):
            cmd = [
                python,
                str(root / "train.py"),
                "-s",
                colmap_dir,
                "-m",
                str(vanilla_model),
            ]
            cmd.extend(build_common_train_args(bench_cfg.get("vanilla_train_args", {})))
            summary = run_command(cmd, base_dir / "vanilla_train", env=env, sample_interval=sample_interval)
            rows.append({"dataset": ds["name"], "pipeline": "vanilla", "stage": "train", **summary})

            if bench_cfg.get("run_render", True):
                cmd = [python, str(root / "render.py"), "-m", str(vanilla_model)]
                cmd.extend(str(x) for x in bench_cfg.get("render_args", []))
                summary = run_command(cmd, base_dir / "vanilla_render", env=env, sample_interval=sample_interval)
                rows.append({"dataset": ds["name"], "pipeline": "vanilla", "stage": "render", **summary})

        if bench_cfg.get("run_full_pipeline", True):
            if sem_mask_dir is None or mask_inst_dir is None:
                raise ValueError(f"Dataset {ds['name']} needs sem_mask_dir and mask_inst_dir for full pipeline")
            cmd = [
                python,
                str(root / "train_sem.py"),
                "-s",
                colmap_dir,
                "-m",
                str(full_model),
                "--mask_dir",
                sem_mask_dir,
            ]
            cmd.extend(build_common_train_args(bench_cfg.get("train_sem_args", {})))
            summary = run_command(cmd, base_dir / "full_train_sem", env=env, sample_interval=sample_interval)
            rows.append({"dataset": ds["name"], "pipeline": "full", "stage": "train_sem", **summary})

            cmd = [
                python,
                str(root / "filter_standalone.py"),
                "--colmap_dir",
                colmap_dir,
                "--model_dir",
                str(full_model),
                "--mask_dir",
                sem_mask_dir,
                "--mask_inst_dir",
                mask_inst_dir,
            ]
            cmd.extend(str(x) for x in bench_cfg.get("filter_standalone_args", []))
            summary = run_command(cmd, base_dir / "full_filter_standalone", env=env, sample_interval=sample_interval)
            rows.append({"dataset": ds["name"], "pipeline": "full", "stage": "filter_standalone", **summary})

            if bench_cfg.get("run_render", True):
                cmd = [python, str(root / "render.py"), "-m", str(full_model)]
                cmd.extend(str(x) for x in bench_cfg.get("render_args", []))
                summary = run_command(cmd, base_dir / "full_render_rgb", env=env, sample_interval=sample_interval)
                rows.append({"dataset": ds["name"], "pipeline": "full", "stage": "render_rgb", **summary})

            if bench_cfg.get("run_paper_render", False):
                cmd = [
                    python,
                    str(root / "render_images_standalone.py"),
                    "--colmap_dir",
                    colmap_dir,
                    "--model_dir",
                    str(full_model),
                ]
                cmd.extend(str(x) for x in bench_cfg.get("paper_render_args", []))
                summary = run_command(cmd, base_dir / "full_render_paper", env=env, sample_interval=sample_interval)
                rows.append({"dataset": ds["name"], "pipeline": "full", "stage": "render_paper", **summary})

    write_benchmark_summary(results_root / "benchmark" / "summary.csv", rows)


def run_contrastive(manifest: dict[str, Any]) -> None:
    root = Path(manifest.get("repo_root", ".")).resolve()
    results_root = Path(maybe_rel(root, manifest.get("results_root", "ablation_runs"))).resolve()
    python = manifest.get("python", sys.executable)
    sample_interval = float(manifest.get("sample_interval_seconds", 2.0))
    cfg = manifest.get("contrastive", {})
    contrastive_root = Path(maybe_rel(root, cfg.get("repo_dir", "contrastive-gaussian-clustering"))).resolve()
    env = cfg.get("env", {})
    rows = []

    for ds in iter_enabled_datasets(manifest):
        if ds.get("run_contrastive", cfg.get("enabled", False)) is False:
            continue
        ds_name = sanitize_name(ds["name"])
        colmap_dir = maybe_rel(root, ds.get("colmap_dir"))
        if colmap_dir is None:
            raise ValueError(f"Dataset {ds['name']} missing colmap_dir")
        objmasks = ds.get("contrastive_objmasks", cfg.get("objmasks", "seg_masks"))
        model_dir = results_root / "contrastive" / ds_name / "model"

        cmd = [
            python,
            str(contrastive_root / "train.py"),
            "-s",
            colmap_dir,
            "-m",
            str(model_dir),
            "--objmasks",
            str(objmasks),
        ]
        cmd.extend(build_common_train_args(cfg.get("train_args", {})))
        summary = run_command(cmd, results_root / "contrastive" / ds_name / "train", env=env, sample_interval=sample_interval)
        rows.append({"dataset": ds["name"], "pipeline": "contrastive", "stage": "train", **summary})

        if cfg.get("run_render", True):
            cmd = [python, str(contrastive_root / "render.py"), "-m", str(model_dir)]
            cmd.extend(str(x) for x in cfg.get("render_args", []))
            summary = run_command(cmd, results_root / "contrastive" / ds_name / "render", env=env, sample_interval=sample_interval)
            rows.append({"dataset": ds["name"], "pipeline": "contrastive", "stage": "render", **summary})

    write_benchmark_summary(results_root / "contrastive" / "summary.csv", rows)


def write_benchmark_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "pipeline",
        "stage",
        "returncode",
        "elapsed_seconds",
        "log_path",
        "resource_summary",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("resource_summary"), dict):
                out["resource_summary"] = json.dumps(out["resource_summary"], sort_keys=True)
            writer.writerow(out)
    print(f"[OK] Summary: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fruit_GS ablation studies from a manifest.")
    parser.add_argument("--config", required=True, help="JSON or YAML manifest.")
    parser.add_argument(
        "--section",
        action="append",
        choices=["direct_dbscan_wide", "direct_dbscan_fixed", "benchmark", "contrastive"],
        help="Section to run. May be passed multiple times. Default: manifest run_sections or direct_dbscan_wide.",
    )
    args = parser.parse_args()

    manifest = load_manifest(Path(args.config))
    sections = args.section or manifest.get("run_sections") or ["direct_dbscan_wide"]

    for section in sections:
        if section == "direct_dbscan_wide":
            run_direct_dbscan(manifest, mode="wide")
        elif section == "direct_dbscan_fixed":
            run_direct_dbscan(manifest, mode="fixed")
        elif section == "benchmark":
            run_benchmarks(manifest)
        elif section == "contrastive":
            run_contrastive(manifest)
        else:
            raise ValueError(f"Unknown section: {section}")


if __name__ == "__main__":
    main()
