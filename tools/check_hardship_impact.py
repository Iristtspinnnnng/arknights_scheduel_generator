"""确认基准/扫描场景里是否真的触发了患难之交（决定要不要重跑基准）。"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx


def main() -> int:
    game_data = GameData.load(Path("data/cache"))
    roster = load_roster_xlsx(Path("examples/fixtures/yituliu_full_roster_maxed.xlsx"))
    cases = [
        ("243", "balanced-orundum", 2, 12),
        ("243", "normal", 2, 12),
        ("342", "normal", 2, 12),
        ("333", "balanced-orundum", 3, 8),
    ]
    for layout, mode, count, hours in cases:
        optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
        result = optimizer.optimize(
            parse_layout(layout),
            mode=mode,
            shift_count=count,
            shift_hours=hours,
            drone_policy="none",
        )
        audit = result.diagnostic_insertion_search["moraleCycleAudit"]
        fiammetta = [
            shift.name
            for shift in result.shifts
            if any(
                skill.operator_name == "菲亚梅塔"
                for room in [*shift.rooms, *shift.dormitories]
                for skill in room.operators
            )
        ]
        print(
            f"{layout}/{mode}/{count}x{hours}: score={result.score:.3f} "
            f"swaps={audit.get('hardshipSwaps')} fiammettaPlacements={fiammetta} "
            f"version={audit.get('version')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
