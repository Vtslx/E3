"""Apply the design's M2 decision rule to archived D1 attribution runs.

Usage:
    python tools/decision_gate.py verification/<run> [verification/<run> ...]

The tool reads RESULT.json only. It separates the memory effect (C3 - C1) from the writeback
effect (C4 - C3) and applies the user's stop rule: if the 32k far bucket is still only a few
points above chance after allocation and stability work, stop A or move to SparSpec instead
of adding samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _checks(result: dict, prefix: str) -> dict[str, dict]:
    return {
        check["name"]: check
        for check in result.get("checks", [])
        if check["name"].startswith(prefix)
    }


def _far_accuracy(result: dict, arm: str) -> tuple[int, int, float]:
    stats = result.get("by_arm", {}).get(arm, {}).get("by_context_length", {}).get("32768", {})
    correct = int(stats.get("correct", 0))
    total = int(stats.get("total", 0))
    return correct, total, (correct / total if total else float("nan"))


def summarize(run_dir: Path) -> dict:
    result = json.loads((run_dir / "RESULT.json").read_text())
    memory_checks = _checks(result, "memory_")
    writeback_checks = _checks(result, "writeback_")
    memory_passed = all(check["passed"] for check in memory_checks.values()) and bool(memory_checks)
    writeback_passed = all(check["passed"] for check in writeback_checks.values()) and bool(writeback_checks)
    c1 = _far_accuracy(result, "C1_stream")
    c3 = _far_accuracy(result, "C3_memory")
    c4 = _far_accuracy(result, "C4_memory_w1")
    memory = result.get("memory_effect", {})
    writeback = result.get("writeback_effect", {})
    return {
        "run": run_dir.name,
        "status": result.get("status"),
        "memory_passed": memory_passed,
        "writeback_passed": writeback_passed,
        "memory_gain": float(memory.get("pooled_gain", float("nan"))),
        "writeback_gain": float(writeback.get("pooled_gain", float("nan"))),
        "memory_mcnemar_p": float(memory.get("paired_far", {}).get("mcnemar_p", float("nan"))),
        "writeback_mcnemar_p": float(writeback.get("paired_far", {}).get("mcnemar_p", float("nan"))),
        "far_c1": c1,
        "far_c3": c3,
        "far_c4": c4,
    }


def decide(summary: dict) -> str:
    """Apply the M2 stop rule and attribute which half failed."""
    far_c3 = summary["far_c3"][2]
    chance = 0.25
    few_points = (far_c3 - chance) < 0.05
    if summary["memory_passed"] and summary["writeback_passed"]:
        return "KEEP A: both memory and writeback effects pass."
    if summary["memory_passed"] and not summary["writeback_passed"]:
        return (
            "KEEP A MEMORY (W2 + future-query allocation); DROP OR REDESIGN W1. "
            "The memory effect passes; writeback is null."
        )
    if few_points and summary["memory_mcnemar_p"] > 0.10:
        return (
            "STOP A OR SWITCH TO SparSpec (M2): the 32k far accuracy is within a few points "
            "of chance and the memory effect is not significant. Do not add samples."
        )
    return (
        "INCONCLUSIVE: neither effect passes and the far bucket is not clearly at chance. "
        "Change the mechanism (allocation/stability/budget) before adding samples."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        summary = summarize(run)
        print(f"run: {summary['run']}")
        print(f"  D1 status: {summary['status']}")
        print(
            f"  memory effect: pass={summary['memory_passed']} gain={summary['memory_gain']:+.4f} "
            f"p={summary['memory_mcnemar_p']:.4f} | far C1={summary['far_c1']} C3={summary['far_c3']}"
        )
        print(
            f"  writeback effect: pass={summary['writeback_passed']} gain={summary['writeback_gain']:+.4f} "
            f"p={summary['writeback_mcnemar_p']:.4f} | far C3={summary['far_c3']} C4={summary['far_c4']}"
        )
        print(f"  decision: {decide(summary)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
