"""Human-readable Markdown report serialization."""

from __future__ import annotations

from pathlib import Path

from rag_eval.contracts import EvaluationReport


def render_markdown_report(report: EvaluationReport) -> str:
    manifest = report.run_manifest
    summary = report.summary
    lines = [
        f"# RAG Evaluation: {manifest.run_id}",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Dataset | `{manifest.dataset_id}` ({manifest.dataset_version}) |",
        f"| Dataset hash | `{manifest.dataset_hash}` |",
        f"| Adapter | `{manifest.adapter}` |",
        f"| Model | `{manifest.model or ''}` |",
        f"| Prompt version | `{manifest.prompt_version or ''}` |",
        f"| Seed | {manifest.seed} |",
        f"| Code revision | `{manifest.code_revision}` |",
        f"| Config hash | `{manifest.config_hash}` |",
        f"| Started | {manifest.started_at.isoformat()} |",
        f"| Ended | {manifest.ended_at.isoformat()} |",
        "",
        "## Coverage",
        "",
        "| Expected | Completed | Errors | Skipped | Coverage | Error rate | Gate |",
        "|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {summary.expected_samples} | {summary.completed_samples} | "
            f"{summary.error_samples} | {summary.skipped_samples} | "
            f"{summary.coverage:.4f} | {summary.error_rate:.4f} | "
            f"{'PASS' if summary.within_thresholds else 'FAIL'} |"
        ),
        "",
        "## Metrics",
        "",
        "| Metric | Value | Denominator |",
        "|---|---:|---:|",
    ]
    for name in sorted(report.metrics.values):
        value = report.metrics.values[name]
        denominator = report.metrics.denominators.get(name, 0)
        lines.append(f"| `{name}` | {value:.6f} | {denominator} |")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(path: str | Path, report: EvaluationReport) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown_report(report), encoding="utf-8")
    return output
