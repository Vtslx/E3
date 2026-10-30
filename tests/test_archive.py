from __future__ import annotations

from pathlib import Path

import pytest

from sprm.errors import ArchiveError
from sprm.verification import archive as archive_module
from sprm.verification.archive import RunArchive, list_runs, verify_run


def _create(tmp_path: Path, *, gate: str = "v1", slug: str = "probe") -> RunArchive:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: probe\n", encoding="utf-8")
    return RunArchive.create(
        root=tmp_path / "archive",
        gate=gate,
        slug=slug,
        config_sha256="0" * 64,
        config_path=config_path,
        argv=["pytest"],
        echo=False,
    )


def test_seal_and_verify_round_trip(tmp_path: Path) -> None:
    run = _create(tmp_path)
    run.write_json("artifacts/payload.json", {"ok": True})
    run.write_text("REPORT.md", "report\n")
    run.seal(status="PASS", verdict="PASS", result={"value": 1}, report="report\n", claims="claims\n")
    assert verify_run(run.run_dir) == []
    assert (run.run_dir / "MANIFEST.json").is_file()
    assert (run.run_dir / "MANIFEST.sha256").is_file()
    entries = list_runs(tmp_path / "archive")
    assert len(entries) == 1
    assert entries[0]["status"] == "PASS"
    with pytest.raises(ArchiveError, match="sealed"):
        run.write_json("artifacts/after.json", {})


def test_tampering_is_detected(tmp_path: Path) -> None:
    run = _create(tmp_path)
    run.write_json("artifacts/payload.json", {"ok": True})
    run.seal(status="PASS", verdict="PASS", result={}, report="report", claims="claims")
    (run.run_dir / "artifacts" / "payload.json").write_text("{\"ok\": false}\n", encoding="utf-8")
    problems = verify_run(run.run_dir)
    assert any("digest mismatch" in problem for problem in problems)


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    assert verify_run(tmp_path / "absent") == [f"missing manifest: {(tmp_path / 'absent' / 'MANIFEST.json').resolve()}"]


def test_run_directories_are_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive_module, "utc_stamp", lambda: "20260101T000000Z")
    first = _create(tmp_path)
    first.seal(status="PASS", verdict="PASS", result={}, report="report", claims="claims")
    with pytest.raises(ArchiveError, match="refusing to overwrite"):
        _create(tmp_path)


def test_invalid_status_is_rejected(tmp_path: Path) -> None:
    run = _create(tmp_path)
    with pytest.raises(ArchiveError, match="invalid status"):
        run.seal(status="MAYBE", verdict="?", result={}, report="", claims="")
