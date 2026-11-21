"""M3b 诊断：同一个布局/模式在两种读出层下的逐房间差异（找掉分点）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import (
    GraphOptimizerConfig,
    GraphScheduleOptimizer,
)
from arknights_schedule_generator.optimizer import parse_layout
from arknights_schedule_generator.power import apply_right_side_preset
from arknights_schedule_generator.roster import load_roster_xlsx

DATA = Path("data/cache")
ROSTER = Path("examples/fixtures/yituliu_full_roster_maxed.xlsx")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="333")
    parser.add_argument("--mode", default="max-orundum")
    parser.add_argument("--shift-count", type=int, default=2)
    parser.add_argument("--shift-hours", type=int, default=12)
    parser.add_argument("--drone-policy", default="none")
    parser.add_argument("--right-side", default="full")
    args = parser.parse_args()

    game_data = GameData.load(DATA)
    roster = load_roster_xlsx(ROSTER)
    results = {}
    for readout in ("per_room", "affinity"):
        optimizer = GraphScheduleOptimizer(
            game_data,
            roster,
            shard_formula="rock",
            graph_config=GraphOptimizerConfig(readout=readout, use_cache=False),
        )
        layout = apply_right_side_preset(parse_layout(args.layout), args.right_side)
        result = optimizer.optimize(
            layout,
            mode=args.mode,
            shift_count=args.shift_count,
            shift_hours=args.shift_hours,
            drone_policy=args.drone_policy,
        )
        results[readout] = result
        report = result.production_report
        print(
            f"== {readout}: score={result.score:.3f} "
            f"orundum={report.dailyExpected.orundum:.2f} "
            f"lmd={report.dailyExpected.lmdGross:.1f} "
            f"shifts={len(result.shifts)}"
        )
        print("   breakdown " + json.dumps(result.score_breakdown, ensure_ascii=False))
        for shift in result.shifts[:2]:
            cells = [
                f"{room.room_id}/{room.target}:{room.score:.3f}:"
                + "|".join(skill.operator_name for skill in room.operators)
                for room in shift.rooms
                if room.room_type in {"TRADING", "MANUFACTURE"}
            ]
            print(f"   {shift.name}  " + "  ".join(cells))
    print()
    print("per-room deltas (affinity - per_room):")
    left, right = results["per_room"], results["affinity"]
    for shift_index, (a_shift, b_shift) in enumerate(zip(left.shifts, right.shifts)):
        for a_room, b_room in zip(a_shift.rooms, b_shift.rooms):
            delta = b_room.score - a_room.score
            if abs(delta) > 0.01:
                print(
                    f"  shift{b_shift.name} {a_room.room_id:14s} "
                    f"{str(a_room.target):10s} {a_room.score:8.3f} -> {b_room.score:8.3f} "
                    f"({delta:+.3f})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
