"""对比同一场景下的宿舍占用与最差收支（baseline vs 现行代码）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--layout", default="252")
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--shift-count", type=int, default=2)
    parser.add_argument("--shift-hours", type=int, default=12)
    args = parser.parse_args()

    game_data = GameData.load(Path(args.data_dir))
    roster = load_roster_xlsx(Path(args.roster))
    optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
    result = optimizer.optimize(
        parse_layout(args.layout),
        mode=args.mode,
        shift_count=args.shift_count,
        shift_hours=args.shift_hours,
        drone_policy="none",
    )
    audit = result.diagnostic_insertion_search["moraleCycleAudit"]
    ledger = audit["moraleLedgerByOperator"]
    ordered = sorted(ledger.items(), key=lambda item: item[1]["balance"])
    active_by_plan = [
        {skill.operator_name for room in shift.rooms for skill in room.operators}
        for shift in result.shifts
    ]
    print(
        f"score={result.score:.3f} plans={len(result.shifts)} "
        f"levels={list(result.layout.dormitory_levels)}"
    )
    print(f"seats={[len(room.operators) for shift in result.shifts for room in shift.dormitories]}")
    print(f"per-plan occupants={[sum(len(r.operators) for r in s.dormitories) for s in result.shifts]}")
    print("top-5 worst balance:")
    for name, item in ordered[:5]:
        rested = [
            index
            for index, shift in enumerate(result.shifts)
            if any(
                skill.operator_name == name
                for room in shift.dormitories
                for skill in room.operators
            )
        ]
        on_duty = [
            index for index, names in enumerate(active_by_plan) if name in names
        ]
        print(
            f"  {name}: balance={item['balance']:.2f} recovery={item['recovery']:.2f} "
            f"consumption={item['consumption']:.2f} rests={rested} onDuty={on_duty}"
        )
    total = sum(item["recovery"] for item in ledger.values())
    print(f"totalRecovery={total:.1f} worst={ordered[0][1]['balance']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
