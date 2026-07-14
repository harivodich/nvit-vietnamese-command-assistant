import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def test_cli_json_runs_complete_pipeline() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_assistant.py"),
            "--json",
            "gọi cho mẹ",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    body: dict[str, Any] = json.loads(completed.stdout)
    assert body["intent"] == "call_contact"
    assert body["action"]["status"] == "mocked"
    assert body["action"]["payload"]["target"] == "mẹ"
