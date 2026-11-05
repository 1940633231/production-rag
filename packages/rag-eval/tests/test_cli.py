from __future__ import annotations

import json
from pathlib import Path

from conftest import retrieval_sample, write_dataset
from rag_eval.cli import main


def test_validate_cli_success_and_failure(tmp_path: Path, capsys) -> None:
    valid_dir = write_dataset(tmp_path, [retrieval_sample()])

    assert main(["validate", str(valid_dir)]) == 0
    assert "VALID" in capsys.readouterr().out

    invalid_dir = write_dataset(
        tmp_path / "invalid",
        [retrieval_sample("duplicate"), retrieval_sample("duplicate")],
    )
    assert main(["validate", str(invalid_dir)]) == 1
    assert "duplicate sample id" in capsys.readouterr().err


def test_run_cli_with_fake_adapter(tmp_path: Path, capsys) -> None:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset: {json.dumps(str(dataset_dir))}",
                "adapter: rag_eval.registry:FakeAdapter",
                f"cache_dir: {json.dumps(str(tmp_path / 'cache'))}",
                f"output_dir: {json.dumps(str(tmp_path / 'reports'))}",
                "min_coverage: 1.0",
                "max_errors: 0",
                "k_values: [5]",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["run", "--config", str(config_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["failed"] is False
    assert payload["report"]["summary"]["coverage"] == 1.0


def test_compare_and_gate_cli_exit_codes(tmp_path: Path, capsys) -> None:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    config_path = tmp_path / "eval.yaml"
    baseline_path = tmp_path / "baseline.json"
    config_path.write_text(
        "\n".join(
            [
                f"dataset: {json.dumps(str(dataset_dir))}",
                "adapter: rag_eval.registry:FakeAdapter",
                f"cache_dir: {json.dumps(str(tmp_path / 'cache'))}",
                f"output_dir: {json.dumps(str(tmp_path / 'reports'))}",
                "min_coverage: 1.0",
                "max_errors: 0",
                "k_values: [5]",
                "thresholds:",
                "  recall@5: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "run",
                "--config",
                str(config_path),
                "--save-baseline",
                str(baseline_path),
                "--json",
            ]
        )
        == 0
    )
    run_payload = json.loads(capsys.readouterr().out)
    report_path = Path(run_payload["run_directory"]) / "report.json"

    assert main(["gate", "--config", str(config_path), "--baseline", str(baseline_path), "--report", str(report_path)]) == 0
    capsys.readouterr()
    assert main(["compare", str(report_path), str(baseline_path)]) == 0
    capsys.readouterr()

    current = json.loads(report_path.read_text(encoding="utf-8"))
    current["metrics"]["values"]["recall@5"] = 0.25
    regressed_path = tmp_path / "regressed.json"
    regressed_path.write_text(json.dumps(current), encoding="utf-8")

    assert main(["gate", "--config", str(config_path), "--baseline", str(baseline_path), "--report", str(regressed_path)]) == 1
