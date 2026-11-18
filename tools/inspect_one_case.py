"""打印单场景的心情审计摘要 + 宿舍分配（M1 手工核对用）。"""

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
    parser.add_argument("--layout", default="243")
    parser.add_argument("--mode", default="balanced-orundum")
    parser.add_argument("--shift-count", type=int, default=2)
    parser.add_argument("--shift-hours", type=int, default=12)
    parser.add_argument("--drone-policy", default="none")
    args = parser.parse_args()

    game_data = GameData.load(Path(args.data_dir))
    roster = load_roster_xlsx(Path(args.roster))
    optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
    result = optimizer.optimize(
        parse_layout(args.layout),
        mode=args.mode,
        shift_count=args.shift_count,
        shift_hours=args.shift_hours,
        drone_policy=args.drone_policy,
    )
    audit = result.diagnostic_insertion_search["moraleCycleAudit"]
    ledger = audit["moraleLedgerByOperator"]
    active = {
        skill.operator_name
        for shift in result.shifts
        for room in shift.rooms
        for skill in room.operators
    }
    print(f"score={result.score:.3f} cycleHours={audit['cycleHours']} plans={len(result.shifts)}")
    print(
        "dormitoryLevels="
        f"{list(result.layout.dormitory_levels)} "
        f"rooms={[len(shift.dormitories) for shift in result.shifts]}"
    )
    print(
        "worstBalance="
        f"{min(item['balance'] for item in ledger.values()):.3f} "
        f"operators={len(ledger)} active={len(active)}"
    )
    print(
        "totalRecovery="
        f"{sum(item['recovery'] for item in ledger.values()):.1f} "
        f"totalConsumption={sum(item['consumption'] for item in ledger.values()):.1f}"
    )
    print(
        "appliedRecovery="
        f"{sum(item['appliedRecovery'] for item in ledger.values()):.1f}"
    )
    print(
        "dormitorySeats="
        f"{sum(len(room.operators) for shift in result.shifts for room in shift.dormitories)} "
        f"distinctSleepers={len({skill.operator_name for shift in result.shifts for room in shift.dormitories for skill in room.operators})}"
    )
    for shift in result.shifts:
        print(f"-- {shift.name} start={shift.start} hours={shift.duration_hours}")
        for room in shift.dormitories:
            rates = ", ".join(
                f"{skill.operator_name}" for skill in room.operators
            )
            print(f"   {room.room_id} L{room.room_level}: {rates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
