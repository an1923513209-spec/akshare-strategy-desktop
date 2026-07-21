"""Durable per-task files for desktop worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class TaskFiles:
    job_id: str
    directory: Path
    input_path: Path
    output_path: Path
    log_path: Path
    result_path: Path
    error_path: Path


def create_task(cache_root: str | Path, payload: Mapping[str, Any]) -> TaskFiles:
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    directory = Path(cache_root).resolve() / "jobs" / job_id
    directory.mkdir(parents=True, exist_ok=False)
    files = TaskFiles(
        job_id=job_id,
        directory=directory,
        input_path=directory / "input.json",
        output_path=directory / "output.pkl",
        log_path=directory / "job.log",
        result_path=directory / "result.json",
        error_path=directory / "error.json",
    )
    files.input_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    files.log_path.write_text(
        f"{datetime.now().isoformat(timespec='seconds')} task {job_id} created\n",
        encoding="utf-8",
    )
    return files


def write_task_result(files: TaskFiles, payload: Mapping[str, Any]) -> None:
    results = payload.get("results")
    errors = payload.get("errors")
    decision_meta = payload.get("decision_meta")
    summary = {
        "job_id": files.job_id,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "result_count": len(results) if isinstance(results, list) else 1,
        "error_count": len(errors) if isinstance(errors, list) else 0,
        "model_version": decision_meta.get("model_version")
        if isinstance(decision_meta, Mapping)
        else None,
    }
    files.result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_task_error(files: TaskFiles, category: str, error: object) -> None:
    payload = {
        "job_id": files.job_id,
        "failed_at": datetime.now().isoformat(timespec="seconds"),
        "status": "failed",
        "category": str(category),
        "error": str(error),
    }
    files.error_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def classify_task_error(error: object) -> str:
    text = str(error or "").lower()
    if "timeout" in text or ("超过" in text and "秒" in text):
        return "timeout"
    if "schema" in text:
        return "schema_mismatch"
    if "production model" in text or ("model" in text and "missing" in text):
        return "model_missing"
    if "proxy" in text or "network" in text or "connection" in text:
        return "network_failure"
    if "factor" in text or "source" in text:
        return "factor_source_missing"
    if "sample" in text or "数据不足" in text:
        return "insufficient_data"
    return "worker_failure"


def latest_task(cache_root: str | Path) -> Path | None:
    root = Path(cache_root).resolve() / "jobs"
    if not root.exists():
        return None
    directories = [path for path in root.iterdir() if path.is_dir()]
    return max(directories, key=lambda path: path.name) if directories else None
