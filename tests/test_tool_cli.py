"""dsp-tool CLI tests: dispatch to the real _impls, argument passing, error envelopes,
and exit codes — the contract a non-MCP consumer (proje8) relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsp_server import tool_cli
from test_video import _rigid_frames, _save_frames


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # AppContext.from_config() writes npz/plots under DSP_DATA_DIR; keep it in tmp.
    monkeypatch.setenv("DSP_DATA_DIR", str(tmp_path / "data"))


def _clip(d: Path) -> Path:
    return _save_frames(_rigid_frames(), d)  # a clean recoverable translation


def test_run_tool_dispatches_to_real_impl(tmp_path: Path):
    text, code = tool_cli.run_tool("verify_motion_consistency", {"frames": str(_clip(tmp_path / "clip"))})
    assert code == 0
    rep = json.loads(text)
    assert "error" not in rep
    assert "inconsistent_fraction" in rep and "n_pairs" in rep


def test_unknown_tool_is_error_exit_2():
    text, code = tool_cli.run_tool("no_such_tool", {})
    assert code == 2
    body = json.loads(text)
    assert "unknown tool" in body["error"] and "available" in body["hint"]


def test_bad_argument_names_is_error_exit_2(tmp_path: Path):
    text, code = tool_cli.run_tool("verify_motion_consistency", {"not_a_real_param": 1})
    assert code == 2
    assert "bad args" in json.loads(text)["error"]


def test_tool_level_toolerror_is_valid_json_exit_0():
    # A missing source is a TOOL error (the _impl catches it), not a CLI error: valid
    # JSON on stdout, exit 0, with an `error` field the caller inspects.
    text, code = tool_cli.run_tool("verify_motion_consistency", {"frames": "/no/such/dir"})
    assert code == 0
    assert "error" in json.loads(text)


def test_main_prints_json_and_returns_code(tmp_path: Path, capsys: pytest.CaptureFixture):
    rc = tool_cli.main(
        ["verify_motion_consistency", "--args-json", json.dumps({"frames": str(_clip(tmp_path / "clip"))})]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "inconsistent_fraction" in out


def test_main_rejects_malformed_json(capsys: pytest.CaptureFixture):
    rc = tool_cli.main(["verify_motion_consistency", "--args-json", "{not json}"])
    assert rc == 2
    assert "not valid JSON" in json.loads(capsys.readouterr().out)["error"]


def test_main_rejects_non_object_json(capsys: pytest.CaptureFixture):
    rc = tool_cli.main(["verify_motion_consistency", "--args-json", "[1, 2, 3]"])
    assert rc == 2
    assert "must be a JSON object" in json.loads(capsys.readouterr().out)["error"]
