"""汇总四个基准场景里 graph 相对 hybrid 的分数构成差异（写成 UTF-8 表格）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEYS = ("lmdGross", "lmdNet", "exp", "orundum")


def main() -> int:
    reports = [
        Path("outputs/benchmark_m1/graph_vs_hybrid_benchmark.json"),
        Path("outputs/benchmark_252_342_m1/graph_vs_hybrid_benchmark.json"),
    ]
    rows: list[str] = []
    rows.append("| 场景 | hybrid | graph | 差 | lmdGross | lmdNet | exp | orundum | roomEfficiency |")
    rows.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for report_path in reports:
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for case in report["cases"]:
            core = case["case"]
            label = (
                f"{core['layout']}/{core['mode']}/{core['shiftCount']}x{core['shiftHours']}"
                f"/{core['dronePolicy']}"
            )
            runs = case.get("runs") or {}
            hybrid = runs.get("hybrid")
            graph = runs.get("graph")
            if not hybrid or not graph:
                continue
            hybrid_break = hybrid.get("scoreBreakdown") or {}
            graph_break = graph.get("scoreBreakdown") or {}
            daily_delta = {}
            for key in KEYS:
                daily_delta[key] = round(
                    float((graph.get("dailyExpected") or {}).get(key) or 0.0)
                    - float((hybrid.get("dailyExpected") or {}).get(key) or 0.0),
                    2,
                )
            rows.append(
                "| {} | {:.3f} | {:.3f} | {:+.3f} | {:+.2f} | {:+.2f} | {:+.2f} | {:+.2f} | {:+.3f} |".format(
                    label,
                    float(hybrid["score"]),
                    float(graph["score"]),
                    float(graph["score"]) - float(hybrid["score"]),
                    daily_delta["lmdGross"],
                    daily_delta["lmdNet"],
                    daily_delta["exp"],
                    daily_delta["orundum"],
                    float(graph_break.get("roomEfficiency") or 0.0)
                    - float(hybrid_break.get("roomEfficiency") or 0.0),
                )
            )
    output = Path("outputs/normal_gap_summary.md")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"written {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
