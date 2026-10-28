#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_DATA_DIRS = ("images", "masks", "mask_instances", "sparse")
DEFAULT_SWEEP_PRESETS = (
    "65:155:0.0:0.2,"
    "65:155:0.0:0.3,"
    "100:135:0.4:0.6,"
    "100:110:0.03:0.04"
)


@dataclass(frozen=True)
class SweepParams:
    ms_min: int
    ms_max: int
    t_min: float
    t_max: float

    @property
    def tag(self) -> str:
        return (
            f"ms{self.ms_min}-{self.ms_max}_"
            f"t{format_float_tag(self.t_min)}-{format_float_tag(self.t_max)}"
        )


@dataclass(frozen=True)
class DatasetJob:
    name: str
    dataset_dir: Path
    model_dir: Path
    train_ply_path: Path
    cluster_ply_path: Path
    sweep_dir: Path
    gt_count: int | None


def default_workspace_root() -> Path:
    env_root = os.environ.get("WORKSPACE_ROOT")
    if env_root:
        return Path(env_root)
    workspace = Path("/workspace")
    if workspace.exists():
        return workspace
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    workspace = default_workspace_root()
    parser = argparse.ArgumentParser(
        description=(
            "Train Fruit3DGS semantic models and sweep filter_standalone.py "
            "seeded-DBSCAN parameters against GT fruit counts."
        )
    )
    parser.add_argument("--workspace_root", type=Path, default=workspace)
    parser.add_argument("--repo_root", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--input_root",
        type=Path,
        default=None,
        help=(
            "Root containing COLMAP dataset folders. If omitted, tries "
            "<workspace>/samuele/10_07_2026, <workspace>/samuele/input, "
            "then <workspace>/samuele/datasets."
        ),
    )
    parser.add_argument("--dataset_glob", default="*")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset names. Defaults to names in --gt_file.",
    )
    parser.add_argument("--gt_file", type=Path, default=None)
    parser.add_argument("--allow_missing_datasets", action="store_true")
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument(
        "--model_subdir",
        default=None,
        help=(
            "Per-dataset model subdirectory under --output_root/<dataset>. "
            "Defaults to output_semantic for the 10_07_2026 layout, otherwise train_sem. "
            "Use '' to disable."
        ),
    )
    parser.add_argument("--sweep_output_root", type=Path, default=None)
    parser.add_argument("--train_iterations", type=int, default=30000)
    parser.add_argument("--filter_iterations", type=int, default=1000)
    parser.add_argument("--topk_contrib", type=int, default=2)
    parser.add_argument("--percent_dense", type=float, default=0.001)
    parser.add_argument("--sem_threshold", type=float, default=0.3)
    parser.add_argument("--filter_fg_thresh", type=float, default=0.7)
    parser.add_argument("--filter_contribs", type=int, default=1)
    parser.add_argument("--filter_device", default="cpu")
    parser.add_argument("--train_data_device", default="cuda")
    parser.add_argument("--cluster_alg", choices=["dbscan", "hdbscan"], default="dbscan")
    parser.add_argument("--bo_n_calls", type=int, default=100)
    parser.add_argument("--bo_patience", type=int, default=8)
    parser.add_argument(
        "--sweep_presets",
        default=DEFAULT_SWEEP_PRESETS,
        help=(
            "Comma/semicolon-separated presets as ms_min:ms_max:t_min:t_max. "
            "Ignored if all four *_values options are provided."
        ),
    )
    parser.add_argument("--ms_min_values", default=None)
    parser.add_argument("--ms_max_values", default=None)
    parser.add_argument("--t_min_values", default=None)
    parser.add_argument("--t_max_values", default=None)
    parser.add_argument("--train_extra_args", default="")
    parser.add_argument("--filter_extra_args", default="")
    parser.add_argument("--overwrite_models", action="store_true")
    parser.add_argument("--overwrite_sweep", action="store_true")
    parser.add_argument("--cluster_only", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    workspace = args.workspace_root.expanduser().resolve()
    args.workspace_root = workspace
    args.repo_root = (args.repo_root or workspace).expanduser().resolve()
    if args.input_root is None:
        candidates = [
            workspace / "samuele" / "10_07_2026",
            workspace / "samuele" / "input",
            workspace / "samuele" / "datasets",
        ]
        args.input_root = next((p for p in candidates if p.is_dir()), candidates[0])
    args.input_root = args.input_root.expanduser().resolve()

    if args.output_root is None:
        args.output_root = args.input_root if args.input_root.name == "10_07_2026" else workspace / "output"
    args.output_root = args.output_root.expanduser().resolve()

    if args.model_subdir is None:
        args.model_subdir = "output_semantic" if args.output_root == args.input_root else "train_sem"

    default_sweep_root = (
        args.input_root / "filter_sweep_results"
        if args.input_root.name == "10_07_2026"
        else workspace / "results" / "train_sem_filter_sweep"
    )
    args.sweep_output_root = (
        args.sweep_output_root or default_sweep_root
    ).expanduser().resolve()
    args.gt_file = (args.gt_file or (args.repo_root / "GT_values.txt")).expanduser().resolve()
    return args


def format_float_tag(value: float) -> str:
    return f"{float(value):.4g}".replace("-", "m").replace(".", "p")


def parse_number_list(text: str, value_type: type) -> list[Any]:
    values = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(value_type(raw))
    if not values:
        raise ValueError(f"Empty value list: {text!r}")
    return values


def parse_sweep_presets(args: argparse.Namespace) -> list[SweepParams]:
    value_fields = [args.ms_min_values, args.ms_max_values, args.t_min_values, args.t_max_values]
    if all(v is not None for v in value_fields):
        ms_mins = parse_number_list(args.ms_min_values, int)
        ms_maxs = parse_number_list(args.ms_max_values, int)
        t_mins = parse_number_list(args.t_min_values, float)
        t_maxs = parse_number_list(args.t_max_values, float)
        presets = [
            SweepParams(ms_min, ms_max, t_min, t_max)
            for ms_min in ms_mins
            for ms_max in ms_maxs
            for t_min in t_mins
            for t_max in t_maxs
        ]
    else:
        chunks = str(args.sweep_presets).replace(";", ",").split(",")
        presets = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [p.strip() for p in chunk.split(":")]
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid sweep preset {chunk!r}; expected ms_min:ms_max:t_min:t_max"
                )
            presets.append(SweepParams(int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])))

    clean: list[SweepParams] = []
    seen: set[tuple[int, int, float, float]] = set()
    for p in presets:
        if p.ms_min > p.ms_max:
            raise ValueError(f"Invalid preset {p}: ms_min > ms_max")
        if p.t_min > p.t_max:
            raise ValueError(f"Invalid preset {p}: t_min > t_max")
        key = (p.ms_min, p.ms_max, p.t_min, p.t_max)
        if key not in seen:
            clean.append(p)
            seen.add(key)
    if not clean:
        raise ValueError("No sweep presets were configured")
    return clean


def load_gt_values(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace(",", " ")
        if ":" in line:
            name, value = line.split(":", 1)
            parts = [name.strip(), value.strip()]
        else:
            parts = line.split()
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        values[parts[0]] = int(parts[1])
    return values


def list_dataset_dirs(input_root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in input_root.glob(pattern) if path.is_dir())


def find_dataset_dir(input_root: Path, name: str) -> Path | None:
    direct = input_root / name
    if direct.is_dir():
        return direct
    lowered = name.lower()
    for candidate in input_root.iterdir() if input_root.is_dir() else []:
        if candidate.is_dir() and candidate.name.lower() == lowered:
            return candidate
    return None


def validate_dataset_dir(dataset_dir: Path) -> None:
    missing = [name for name in REQUIRED_DATA_DIRS if not (dataset_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"{dataset_dir} is missing required folders: {', '.join(missing)}")
    sparse0 = dataset_dir / "sparse" / "0"
    if not sparse0.is_dir():
        raise FileNotFoundError(f"{dataset_dir} is missing sparse/0")


def model_dir_for(dataset_dir: Path, output_root: Path, model_subdir: str) -> Path:
    base = output_root / dataset_dir.name
    subdir = str(model_subdir).strip().strip("/")
    return base / subdir if subdir else base


def build_job(
    name: str,
    dataset_dir: Path,
    output_root: Path,
    model_subdir: str,
    train_iterations: int,
    sweep_output_root: Path,
    gt_count: int | None,
) -> DatasetJob:
    model_dir = model_dir_for(dataset_dir, output_root, model_subdir)
    iter_dir = model_dir / "point_cloud" / f"iteration_{train_iterations}"
    return DatasetJob(
        name=name,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        train_ply_path=iter_dir / "point_cloud.ply",
        cluster_ply_path=iter_dir / "filtered_scene_clusters_cc.ply",
        sweep_dir=sweep_output_root / name,
        gt_count=gt_count,
    )


def is_non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def train_output_ready(job: DatasetJob) -> bool:
    return is_non_empty_file(job.train_ply_path)


def cluster_output_files(job: DatasetJob) -> list[Path]:
    iteration_dir = job.train_ply_path.parent
    return [
        iteration_dir / "scene_clusters.ply",
        iteration_dir / "filtered_scene_clusters.ply",
        iteration_dir / "filtered_scene_clusters_cc.ply",
        iteration_dir / "filtered_scene_clusters_cc_no_bkg.ply",
    ]


def clear_cluster_outputs(job: DatasetJob, dry_run: bool) -> None:
    for path in cluster_output_files(job):
        if path.exists():
            print(f"[INFO] Removing previous clustering artifact: {path}")
            if not dry_run:
                path.unlink()


def backup_cluster_outputs(job: DatasetJob, backup_dir: Path, dry_run: bool) -> dict[str, str]:
    backups: dict[str, str] = {}
    for path in cluster_output_files(job):
        if not path.is_file():
            continue
        dst = backup_dir / path.name
        backups[str(path)] = str(dst)
        print(f"[INFO] Backing up current clustering artifact: {path} -> {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
    return backups


def restore_cluster_outputs(backups: dict[str, str], dry_run: bool) -> None:
    if not backups:
        return
    for original, backup in backups.items():
        original_path = Path(original)
        backup_path = Path(backup)
        if not backup_path.is_file():
            continue
        print(f"[WARN] Restoring previous clustering artifact: {backup_path} -> {original_path}")
        if not dry_run:
            original_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, original_path)


def assert_filter_runtime_available(args: argparse.Namespace) -> None:
    """Fail before touching PLYs if the standalone filter cannot use CUDA.

    filter_standalone.py still loads the 3DGS scene/cameras through code paths
    that call .cuda(), even when --device cpu is selected for later filtering.
    """
    if args.dry_run or args.train_only:
        return
    cmd = [
        args.python,
        "-c",
        (
            "import torch, sys; "
            "sys.exit(0 if torch.cuda.is_available() else 3)"
        ),
    ]
    proc = subprocess.run(cmd, cwd=str(args.repo_root), text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "filter_standalone.py requires a CUDA-visible container during scene loading. "
            "Run inside the GPU container and verify `python -c \"import torch; "
            "print(torch.cuda.is_available())\"` returns True before launching the sweep."
        )


def shlex_split_optional(text: str) -> list[str]:
    text = str(text or "").strip()
    return shlex.split(text) if text else []


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(
    cmd: list[str],
    dry_run: bool,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(cmd)
    (stdout_log.parent / "command.txt").write_text(command_text + "\n")
    print(f"[RUN] {command_text}")
    if dry_run:
        return 0
    with stdout_log.open("w") as stdout, stderr_log.open("w") as stderr:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=stdout, stderr=stderr, text=True)
    (stdout_log.parent / "returncode.txt").write_text(f"{proc.returncode}\n")
    if proc.returncode != 0:
        print(
            f"[ERROR] Command failed with return code {proc.returncode}. "
            f"stdout={stdout_log} stderr={stderr_log}"
        )
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def train_command(args: argparse.Namespace, job: DatasetJob) -> list[str]:
    cmd = [
        args.python,
        str(args.repo_root / "train_sem.py"),
        "-s",
        str(job.dataset_dir),
        "-m",
        str(job.model_dir),
        "--mask_dir",
        str(job.dataset_dir / "masks"),
        "--topk_contrib",
        str(args.topk_contrib),
        "--percent_dense",
        str(args.percent_dense),
        "--sem_threshold",
        str(args.sem_threshold),
        "--iterations",
        str(args.train_iterations),
        "--test_iterations",
        str(args.train_iterations),
        "--save_iterations",
        str(args.train_iterations),
        "--data_device",
        str(args.train_data_device),
        "--disable_viewer",
    ]
    cmd.extend(shlex_split_optional(args.train_extra_args))
    return cmd


def filter_command(
    args: argparse.Namespace,
    job: DatasetJob,
    preset: SweepParams,
    metrics_path: Path,
) -> list[str]:
    cmd = [
        args.python,
        str(args.repo_root / "filter_standalone.py"),
        "--colmap_dir",
        str(job.dataset_dir),
        "--model_dir",
        str(job.model_dir),
        "--mask_dir",
        str(job.dataset_dir / "masks"),
        "--mask_inst_dir",
        str(job.dataset_dir / "mask_instances"),
        "--ply_iteration",
        str(args.train_iterations),
        "--contribs",
        str(args.filter_contribs),
        "--iterations",
        str(args.filter_iterations),
        "--fg_thresh",
        str(args.filter_fg_thresh),
        "--cluster_alg",
        str(args.cluster_alg),
        "--device",
        str(args.filter_device),
        "--skip_visualization",
        "--ms_min",
        str(preset.ms_min),
        "--ms_max",
        str(preset.ms_max),
        "--t_min",
        str(preset.t_min),
        "--t_max",
        str(preset.t_max),
        "--bo_n_calls",
        str(args.bo_n_calls),
        "--bo_patience",
        str(args.bo_patience),
        "--metrics_out",
        str(metrics_path),
    ]
    if job.gt_count is not None:
        cmd.extend(["--gt_count", str(job.gt_count)])
    cmd.extend(shlex_split_optional(args.filter_extra_args))
    return cmd


def best_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    abs_error = row.get("count_abs_error")
    f1 = row.get("f1_count")
    pred = row.get("pred_count")
    abs_error_f = float(abs_error) if abs_error is not None else float("inf")
    f1_f = float(f1) if f1 is not None else -1.0
    pred_f = float(pred) if pred is not None else float("inf")
    return (abs_error_f, -f1_f, pred_f)


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def copy_best_artifacts(job: DatasetJob, metrics: dict[str, Any], preset: SweepParams) -> dict[str, Any]:
    best_dir = job.sweep_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    copied = {
        "best_filtered_ply": copy_if_exists(
            job.cluster_ply_path,
            best_dir / f"{job.name}_{preset.tag}_filtered_scene_clusters_cc.ply",
        ),
        "best_filtered_no_bkg_ply": copy_if_exists(
            job.cluster_ply_path.with_name("filtered_scene_clusters_cc_no_bkg.ply"),
            best_dir / f"{job.name}_{preset.tag}_filtered_scene_clusters_cc_no_bkg.ply",
        ),
    }
    best_metrics = dict(metrics)
    best_metrics.update(copied)
    write_json(best_dir / "best_metrics.json", best_metrics)
    return best_metrics


def run_filter_sweep(args: argparse.Namespace, job: DatasetJob, presets: list[SweepParams]) -> dict[str, Any]:
    if args.overwrite_sweep and job.sweep_dir.exists() and not args.dry_run:
        print(f"[INFO] Removing previous sweep dir: {job.sweep_dir}")
        shutil.rmtree(job.sweep_dir)

    all_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_preset: SweepParams | None = None

    for preset_idx, preset in enumerate(presets, start=1):
        run_dir = job.sweep_dir / "runs" / preset.tag
        metrics_path = run_dir / "metrics.json"
        print(f"[INFO] Sweep {preset_idx}/{len(presets)} for {job.name}: {preset.tag}")

        if metrics_path.is_file() and not args.overwrite_sweep:
            metrics = load_json(metrics_path)
            print(f"[SKIP] Existing metrics: {metrics_path}")
        else:
            backups = backup_cluster_outputs(job, run_dir / "pre_run_backup", dry_run=args.dry_run)
            try:
                clear_cluster_outputs(job, dry_run=args.dry_run)
                cmd = filter_command(args, job, preset, metrics_path)
                run_command(
                    cmd,
                    dry_run=args.dry_run,
                    cwd=args.repo_root,
                    stdout_log=run_dir / "stdout.log",
                    stderr_log=run_dir / "stderr.log",
                )
                if args.dry_run:
                    metrics = {
                        "dataset": job.name,
                        "gt_count": job.gt_count,
                        **asdict(preset),
                        "dry_run": True,
                        "run_dir": str(run_dir),
                    }
                    write_json(metrics_path, metrics)
                elif not metrics_path.is_file():
                    raise RuntimeError(f"filter_standalone.py did not write metrics: {metrics_path}")
                else:
                    metrics = load_json(metrics_path)
            except Exception:
                restore_cluster_outputs(backups, dry_run=args.dry_run)
                raise

        row = {
            "dataset": job.name,
            "gt_count": job.gt_count,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            **asdict(preset),
            **metrics,
        }
        all_rows.append(row)

        if best is None or best_sort_key(row) < best_sort_key(best):
            best = row
            best_preset = preset
            if not args.dry_run:
                best = copy_best_artifacts(job, row, preset)
            print(
                "[BEST] "
                f"{job.name}: pred={row.get('pred_count')} gt={job.gt_count} "
                f"abs_error={row.get('count_abs_error')} preset={preset.tag}"
            )

    write_json(job.sweep_dir / "all_runs.json", all_rows)
    write_csv(job.sweep_dir / "all_runs.csv", all_rows)
    if best is not None:
        write_json(job.sweep_dir / "best_metrics.json", best)
    return {
        "dataset": job.name,
        "gt_count": job.gt_count,
        "best_preset": best_preset.tag if best_preset else None,
        "best_metrics_path": str(job.sweep_dir / "best_metrics.json"),
        "best_pred_count": best.get("pred_count") if best else None,
        "best_count_abs_error": best.get("count_abs_error") if best else None,
        "best_f1_count": best.get("f1_count") if best else None,
        "sweep_dir": str(job.sweep_dir),
    }


def discover_jobs(args: argparse.Namespace, gt_values: dict[str, int]) -> list[DatasetJob]:
    if args.datasets:
        names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    elif gt_values:
        names = list(gt_values)
    else:
        names = [path.name for path in list_dataset_dirs(args.input_root, args.dataset_glob)]

    jobs: list[DatasetJob] = []
    missing: list[str] = []
    for name in names:
        dataset_dir = find_dataset_dir(args.input_root, name)
        if dataset_dir is None:
            missing.append(name)
            continue
        validate_dataset_dir(dataset_dir)
        jobs.append(
            build_job(
                name=name,
                dataset_dir=dataset_dir,
                output_root=args.output_root,
                model_subdir=args.model_subdir,
                train_iterations=args.train_iterations,
                sweep_output_root=args.sweep_output_root,
                gt_count=gt_values.get(name),
            )
        )

    if missing and not args.allow_missing_datasets:
        raise FileNotFoundError(
            f"Datasets listed but not found under {args.input_root}: {', '.join(missing)}"
        )
    if missing:
        print(f"[WARN] Missing datasets skipped: {', '.join(missing)}")
    return jobs


def main() -> None:
    args = resolve_defaults(parse_args())
    train_script = args.repo_root / "train_sem.py"
    filter_script = args.repo_root / "filter_standalone.py"

    if not args.input_root.is_dir():
        raise SystemExit(f"Invalid input root: {args.input_root}")
    if not train_script.is_file():
        raise SystemExit(f"Missing train_sem.py: {train_script}")
    if not filter_script.is_file():
        raise SystemExit(f"Missing filter_standalone.py: {filter_script}")

    gt_values = load_gt_values(args.gt_file)
    presets = parse_sweep_presets(args)
    jobs = discover_jobs(args, gt_values)
    if not jobs:
        raise SystemExit(f"No dataset folders found under {args.input_root}")

    print(f"[INFO] Python: {args.python}")
    print(f"[INFO] Repo root: {args.repo_root}")
    print(f"[INFO] Input root: {args.input_root}")
    print(f"[INFO] Model output root: {args.output_root}")
    print(f"[INFO] Sweep output root: {args.sweep_output_root}")
    print(f"[INFO] GT file: {args.gt_file} ({len(gt_values)} entries)")
    print(f"[INFO] Datasets: {', '.join(job.name for job in jobs)}")
    print(f"[INFO] Sweep presets: {', '.join(p.tag for p in presets)}")
    assert_filter_runtime_available(args)

    failures: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for job_idx, job in enumerate(jobs, start=1):
        print(f"\n[DATASET {job_idx}/{len(jobs)}] {job.name}")
        print(f"[INFO] Dataset dir: {job.dataset_dir}")
        print(f"[INFO] Model dir: {job.model_dir}")
        print(f"[INFO] GT count: {job.gt_count}")

        try:
            if job.model_dir.exists() and args.overwrite_models and not args.cluster_only:
                print(f"[INFO] Removing existing model dir: {job.model_dir}")
                if not args.dry_run:
                    shutil.rmtree(job.model_dir)

            if args.cluster_only:
                if not train_output_ready(job):
                    raise RuntimeError(f"--cluster_only requires trained model PLY: {job.train_ply_path}")
                print(f"[INFO] train_sem.py skipped (--cluster_only): {job.train_ply_path}")
            elif train_output_ready(job):
                print(f"[INFO] train_sem.py already complete: {job.train_ply_path}")
            else:
                job.model_dir.parent.mkdir(parents=True, exist_ok=True)
                train_log_dir = job.sweep_dir / "train"
                run_command(
                    train_command(args, job),
                    dry_run=args.dry_run,
                    cwd=args.repo_root,
                    stdout_log=train_log_dir / "stdout.log",
                    stderr_log=train_log_dir / "stderr.log",
                )
                if not args.dry_run and not train_output_ready(job):
                    raise RuntimeError(f"train_sem.py did not produce expected file: {job.train_ply_path}")

            if args.train_only:
                summary_rows.append(
                    {
                        "dataset": job.name,
                        "gt_count": job.gt_count,
                        "model_dir": str(job.model_dir),
                        "train_ply_path": str(job.train_ply_path),
                        "status": "trained",
                    }
                )
                continue

            summary_rows.append(run_filter_sweep(args, job, presets))
        except Exception as exc:
            failure = {"dataset": job.name, "error": str(exc)}
            if isinstance(exc, subprocess.CalledProcessError):
                failure["returncode"] = exc.returncode
                failure["command"] = shlex.join(str(part) for part in exc.cmd)
            failures.append(failure)
            print(f"[WARN] Failed {job.name}: {exc}")

    args.sweep_output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.sweep_output_root / "batch_summary.json", summary_rows)
    write_csv(args.sweep_output_root / "batch_summary.csv", summary_rows)
    write_json(args.sweep_output_root / "failures.json", failures)

    print(f"\n[INFO] Done. Summary: {args.sweep_output_root / 'batch_summary.json'}")
    print(f"[INFO] Failures: {len(failures)}")
    if failures:
        for failure in failures:
            print(f"[WARN] {failure['dataset']}: {failure['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
