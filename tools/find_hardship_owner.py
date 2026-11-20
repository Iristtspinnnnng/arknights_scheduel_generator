"""找出满练表里拥有「患难之交」（互换心情）的干员，以及她在现有排班里是否被安排进宿舍。"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx

OUT = Path("outputs/hardship_owner.md")


def main() -> int:
    game_data = GameData.load(Path("data/cache"))
    roster = load_roster_xlsx(Path("examples/fixtures/yituliu_full_roster_maxed.xlsx"))
    optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
    lines: list[str] = ["# 患难之交持有者", ""]
    owners: list[str] = []
    for operator in optimizer.roster:
        for skill in optimizer.skills_by_operator.get(operator.name, []):
            plain = skill.description or ""
            if "互换心情" in plain or "患难" in plain:
                lines.append(f"- {operator.name} / {skill.buff_id} / {skill.room_type}：{plain}")
                owners.append(operator.name)
    lines.append(f"\n共 {len(set(owners))} 名：{sorted(set(owners))}")
    lines.append("")

    layout = parse_layout(layout_raw := sys.argv[1] if len(sys.argv) > 1 else "243")
    result = optimizer.optimize(layout, mode="balanced-orundum", shift_count=2, shift_hours=12)
    lines.append(f"## 在 {layout_raw} 平衡搓玉的排班里她出现在哪")
    lines.append("")
    for shift in result.shifts:
        active = [
            skill.operator_name for room in shift.rooms for skill in room.operators
        ]
        sleeping = [
            skill.operator_name for room in shift.dormitories for skill in room.operators
        ]
        for owner in sorted(set(owners)):
            if owner in active:
                lines.append(f"- 班次 {shift.name}：在岗")
            elif owner in sleeping:
                lines.append(f"- 班次 {shift.name}：在宿舍")
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
