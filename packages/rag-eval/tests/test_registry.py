from __future__ import annotations

from pathlib import Path

from rag_eval.registry import load_adapter


def test_adapter_module_can_be_loaded_from_current_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    module_path = tmp_path / "local_eval_adapter.py"
    module_path.write_text(
        "\n".join(
            [
                "from rag_eval.registry import FakeAdapter",
                "",
                "class LocalAdapter(FakeAdapter):",
                "    name = 'local-test'",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    adapter = load_adapter("local_eval_adapter:LocalAdapter")

    assert adapter.name == "local-test"
