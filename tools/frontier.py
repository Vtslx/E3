"""Build the compression-vs-far-accuracy frontier from archived D1 runs.

The x-axis uses **resident attention bytes at 32k** (the honest number the current
implementation pays), and a dashed **logical** series that assumes the memory slot is the
only copy (bf16) and the W1-only anchor/replay buffers are gone. The global-attention-only
ratio is also tabulated: it is the number that looks like 8x, but it is not the resident
footprint.

Usage:
    python tools/frontier.py verification/<d1-512> verification/<d1-1024> verification/<d1-2048>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_run(run_dir: Path) -> dict | None:
    result = json.loads((run_dir / "RESULT.json").read_text())
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    extra = manifest.get("extra", {})
    slots = int(extra.get("slot_budget") or config["tier2"]["num_slots"])
    sink = int(config["tier1"]["sink_tokens"])
    window = int(config["tier1"]["window_tokens"])
    model = manifest.get("model", {})
    global_layers = len(model.get("global_layers", [])) or 6
    kv_width = int(model.get("num_kv_heads", 4)) * int(model.get("head_dim", 128))
    dtype_bytes = 2 if model.get("dtype") in {"bfloat16", "float16"} else 4
    per_token_global = global_layers * 2 * kv_width * dtype_bytes

    c0: list[float] = []
    c3: list[float] = []
    memory: list[float] = []
    for path in (run_dir / "artifacts" / "samples").glob("*.json"):
        sample = json.loads(path.read_text())
        if int(sample["sample"]["context_length"]) < 32768:
            continue
        exact = sample["arms"].get("C0_full")
        memory_arm = sample["arms"].get("C3_memory")
        if exact and not exact["error"]:
            c0.append(float(exact["result"]["kv_bytes_per_token"]))
        if memory_arm and not memory_arm["error"]:
            c3.append(float(memory_arm["result"]["kv_bytes_per_token"]))
            memory.append(float(memory_arm["result"]["memory_bytes"]))
    if not c0 or not c3:
        return None

    def far(arm: str) -> float:
        stats = result.get("by_arm", {}).get(arm, {}).get("by_context_length", {}).get("32768", {})
        return float(stats.get("accuracy", float("nan")))

    c0_avg = _avg(c0)
    c3_avg = _avg(c3)
    memory_avg = _avg(memory)
    core_fp32 = slots * kv_width * 4 * 2 * global_layers
    w1_overhead = max(0.0, memory_avg - core_fp32)
    no_w1 = c3_avg - w1_overhead / 32768
    bf16_core = no_w1 - (core_fp32 / 2) / 32768
    logical = (sink + window + slots) * per_token_global / 32768 + (c0_avg - per_token_global)
    return {
        "run": run_dir.name,
        "slots": slots,
        "far_c1": far("C1_stream"),
        "far_c3": far("C3_memory"),
        "far_c0": far("C0_full"),
        "c0_bpt": c0_avg,
        "c3_bpt": c3_avg,
        "memory_mib": memory_avg / 1024 / 1024,
        "resident_comp": c0_avg / c3_avg,
        "no_w1_comp": c0_avg / no_w1,
        "bf16_comp": c0_avg / bf16_core,
        "logical_comp": c0_avg / logical,
        "global_comp": 32768 / (sink + window + slots),
    }


def _svg(rows: list[dict]) -> str:
    width, height = 820, 460
    left, right, top, bottom = 78, 30, 52, 64
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(comp: float) -> float:
        clamped = min(max(comp, 1.0), 16.0)
        return left + (math.log2(clamped) / math.log2(16.0)) * plot_w

    def y_pos(acc: float) -> float:
        return top + (1.0 - min(max(acc, 0.0), 1.0)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-size="17" font-weight="bold">'
        "32k far accuracy vs compression (LFM3-1.7B, attention allocation)</text>",
        f'<text x="{left}" y="44" font-size="11" fill="#555">'
        "solid: measured resident bytes; dashed: logical bf16 / no W1-only buffers</text>",
    ]
    for comp in (1, 2, 4, 8, 16):
        x = x_pos(comp)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e4e4e4"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 20}" font-size="11" text-anchor="middle">'
            f"{comp}x</text>"
        )
    for acc in (0, 0.25, 0.5, 0.75, 1.0):
        y = y_pos(acc)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e4e4e4"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="11" text-anchor="end">'
            f"{int(acc * 100)}%</text>"
        )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>'
    )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>')
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 18}" font-size="12" text-anchor="middle">'
        "compression vs exact full KV (log scale)</text>"
    )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2}" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_h / 2})">32k far accuracy</text>'
    )

    # C1 baseline
    if rows and not math.isnan(rows[0]["far_c1"]):
        y = y_pos(rows[0]["far_c1"])
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#999" stroke-dasharray="2,4"/>'
        )
        parts.append(
            f'<text x="{left + 6}" y="{y - 6:.1f}" font-size="10" fill="#777">'
            f"C1 streaming {rows[0]['far_c1'] * 100:.1f}%</text>"
        )

    # C0 exact
    if rows and not math.isnan(rows[0]["far_c0"]):
        parts.append(
            f'<circle cx="{x_pos(1.0):.1f}" cy="{y_pos(rows[0]["far_c0"]):.1f}" r="5" fill="#222"/>'
        )
        parts.append(
            f'<text x="{x_pos(1.0) + 8:.1f}" y="{y_pos(rows[0]["far_c0"]) - 8:.1f}" '
            f'font-size="10">C0 exact {rows[0]["far_c0"] * 100:.1f}%</text>'
        )

    def polyline(key: str, colour: str, dash: str) -> None:
        ordered = sorted(rows, key=lambda row: row[key])
        points = " ".join(f"{x_pos(row[key]):.1f},{y_pos(row['far_c3']):.1f}" for row in ordered)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-dasharray="{dash}"/>'
        )
        for row in ordered:
            x, y = x_pos(row[key]), y_pos(row["far_c3"])
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}"/>')
            parts.append(
                f'<text x="{x + 7:.1f}" y="{y - 8:.1f}" font-size="10" fill="{colour}">'
                f"{row['slots']} slots</text>"
            )

    polyline("resident_comp", "#1f5fbf", "0")
    polyline("logical_comp", "#1f8f4e", "5,4")
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/frontier"))
    args = parser.parse_args()
    rows = [row for run in args.runs if (row := load_run(run)) is not None]
    if not rows:
        raise SystemExit("no D1 runs with 32k samples")
    rows.sort(key=lambda row: row["slots"])
    lines = [
        "# Compression vs 32k far accuracy",
        "",
        "Measured resident bytes at 32k; logical assumes a bf16 memory slot is the only copy and",
        "the W1-only anchor/replay buffers are removed. The global-attention ratio is the",
        "sink+slots+window cache alone, not the resident footprint.",
        "",
        "| slots | far C1 | far C3 | resident | no W1 buffers | bf16 core | logical | global attn | C3 MiB |",
        "|-------|--------|--------|----------|---------------|-----------|---------|-------------|--------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['slots']} | {row['far_c1'] * 100:.1f}% | {row['far_c3'] * 100:.1f}% "
            f"| {row['resident_comp']:.2f}x | {row['no_w1_comp']:.2f}x | {row['bf16_comp']:.2f}x "
            f"| {row['logical_comp']:.2f}x | {row['global_comp']:.2f}x | {row['memory_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The measured frontier is the solid line: more compression at 512 costs far accuracy;",
            "the 2048 point buys 69.4% far accuracy at 2.28x resident. The 8x design target is only",
            "the global-attention/logical projection; the resident implementation pays for the fp32",
            "memory duplicate, the anchor refs, and the replay buffer. Cutting W1 removes the",
            "last two; a bf16 memory and a direct associative read (no duplicated slot cache)",
            "close the remaining gap.",
            "",
            "![frontier](frontier.svg)",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".svg").write_text(_svg(rows) + "\n")
    for row in rows:
        print(
            f"{row['slots']:>4} slots  far C1={row['far_c1'] * 100:5.1f}%  C3={row['far_c3'] * 100:5.1f}%  "
            f"resident={row['resident_comp']:.2f}x  noW1={row['no_w1_comp']:.2f}x  "
            f"bf16={row['bf16_comp']:.2f}x  logical={row['logical_comp']:.2f}x  global={row['global_comp']:.2f}x"
        )
    print(f"wrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
