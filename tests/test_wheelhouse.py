"""docker/use_wheelhouse.py: only wheels whose name AND version match a pin are installed from the wheelhouse."""
import importlib.util
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "docker" / "use_wheelhouse.py"
spec = importlib.util.spec_from_file_location("use_wheelhouse", SCRIPT)
uw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uw)


def run(tmp_path, monkeypatch, pins, wheels):
    (tmp_path / "constraints.txt").write_text("\n".join(pins) + "\n")
    wh = tmp_path / "wheels"
    wh.mkdir()
    for w in wheels:
        (wh / w).write_bytes(b"")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))
    uw.main(str(tmp_path / "constraints.txt"), str(wh))
    return calls


def test_only_pinned_name_and_version_are_installed_offline(tmp_path, monkeypatch):
    calls = run(tmp_path, monkeypatch, ["torch==2.9.1+cpu", "huggingface_hub==0.36.2", "numpy==2.5.4"],
                ["torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl", "huggingface_hub-0.36.2-py3-none-any.whl",
                 "numpy-9.9.9-cp312-cp312-manylinux_2_28_x86_64.whl", "colorama-0.4.6-py2.py3-none-any.whl"])
    assert len(calls) == 1
    cmd = calls[0]
    assert "--no-index" in cmd and "--no-deps" in cmd
    names = sorted(Path(c).name for c in cmd if c.endswith(".whl"))
    assert names == ["huggingface_hub-0.36.2-py3-none-any.whl", "torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"]


def test_an_empty_or_missing_wheelhouse_does_nothing(tmp_path, monkeypatch, capsys):
    assert run(tmp_path, monkeypatch, ["torch==2.9.1+cpu"], []) == []
    uw.main(str(tmp_path / "constraints.txt"), str(tmp_path / "nonexistent"))
    assert "downloaded" in capsys.readouterr().out
