"""对比同一基准里 hybrid 与 graph 的分数构成与逐房间差异。

用法::

    python tools/diff_benchmark_case.py --report outputs/benchmark_m1/graph_vs_hybrid_benchmark.json \
        --case 243 / normal / 2x12h / drones=none
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_cases(path: Path) -> list[dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return report["cases"]


def case_label(case: dict) -> str:
    core = case.get("case") or {}
    return f"{core.get('layout')} / {core.get('mode')} / {core.get('shiftCount')}x{core.get('shiftHours')}h / drones={core.get('dronePolicy')}"


def payload(case: dict, algorithm: str) -> dict | None:
    runs = case.get("runs") or {}
    if algorithm in runs:
        return runs[algorithm]
    for key, run in runs.items():
        if key.lower().startswith(algorithm):
            return run
    return None


def room_map(run: dict) -> dict[tuple[str, str], dict]:
    mapping: dict[tuple[str, str], dict] = {}
    for room in run.get("roomScores") or []:
        mapping[(room["shift"], room["roomId"])] = room
    return mapping


def print_run(run: dict) -> None:
    daily = run.get("dailyExpected") or {}
    print(
        "  score={:.3f} wall={}s shifts={}".format(
            float(run.get("score") or 0.0),
            run.get("wallSeconds"),
            run.get("shiftCount"),
        )
    )
    print(
        "  daily="
        + ", ".join(
            f"{key}={daily.get(key)}" for key in ("lmdGross", "lmdNet", "exp", "orundum", "pureGoldDelta")
        )
    )
    breakdown = run.get("scoreBreakdown") or {}
    print(
        "  breakdown="
        + ", ".join(f"{key}={value}" for key, value in sorted(breakdown.items()))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--case", default=None, help="子串匹配，例如 '342 / normal'")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    cases = load_cases(Path(args.report))
    for case in cases:
        label = case_label(case)
        if args.case and args.case not in label:
            continue
        hybrid = payload(case, "hybrid")
        graph = payload(case, "graph")
        print(f"=== {label} ===")
        if not hybrid or not graph:
            print("  missing run")
            continue
        print(" hybrid:")
        print_run(hybrid)
        print(" graph:")
        print_run(graph)
        hybrid_daily = hybrid.get("dailyExpected") or {}
        graph_daily = graph.get("dailyExpected") or {}
        delta = {
            key: round(float(graph_daily.get(key) or 0.0) - float(hybrid_daily.get(key) or 0.0), 3)
            for key in ("lmdGross", "lmdNet", "exp", "orundum", "pureGoldDelta", "shardDelta")
        }
        print(f"  daily delta (graph - hybrid) = {delta}")
        hybrid_rooms = room_map(hybrid)
        graph_rooms = room_map(graph)
        rows = []
        for key in sorted(set(hybrid_rooms) & set(graph_rooms)):
            left = hybrid_rooms[key]
            right = graph_rooms[key]
            left_ops = tuple(left.get("operators") or ())
            right_ops = tuple(right.get("operators") or ())
            rows.append(
                (
                    float(right.get("score") or 0.0) - float(left.get("score") or 0.0),
                    key,
                    float(left.get("score") or 0.0),
                    float(right.get("score") or 0.0),
                    left_ops,
                    right_ops,
                )
            )
        rows.sort(key=lambda row: row[0])
        print(f"  rooms in both plans: {len(rows)}; total room-score delta={sum(r[0] for r in rows):.3f}")
        for delta_score, key, left_score, right_score, left_ops, right_ops in rows[: args.top]:
            print(
                f"  {key} hybrid={left_score:.2f} graph={right_score:.2f} d={delta_score:+.2f}"
            )
            if left_ops != right_ops:
                only_hybrid = [name for name in left_ops if name not in right_ops]
                only_graph = [name for name in right_ops if name not in left_ops]
                print(f"      hybrid-only: {only_hybrid} | graph-only: {only_graph}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
