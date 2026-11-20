"""检查：给定贸易站规格与排除集，graph 短名单是否覆盖全量最优组合。

对 ``TRADING/O_GOLD`` L3 3 工位、无排除集：
- 用父类穷举（``_room_combos``，模拟器打分）找精确分最高的组合；
- 看 graph 短名单里有没有它、排第几；
- 顺便对比短名单长度与最优精确分。
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import GraphScheduleOptimizer
from arknights_schedule_generator.optimizer import RoomSpec, ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx


def main() -> int:
    game_data = GameData.load(Path("data/cache"))
    roster = load_roster_xlsx(Path("examples/fixtures/yituliu_full_roster_maxed.xlsx"))
    layout = parse_layout("342")
    spec = RoomSpec("trading_1", "TRADING", "O_GOLD", 3, 3, 3)
    duration = 12.0

    hybrid = ScheduleOptimizer(game_data, roster, shard_formula="rock")
    exhaustive = hybrid._room_combos(spec, set(), duration)
    exact_rows = [
        (
            hybrid._score_room_combo(spec, list(combo.operators), duration),
            tuple(skill.operator_name for skill in combo.operators),
        )
        for combo in exhaustive
    ]
    exact_rows.sort(key=lambda row: row[0], reverse=True)

    graph = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
    shortlist = graph._room_combos(spec, set(), duration)
    shortlist_names = [tuple(skill.operator_name for skill in combo.operators) for combo in shortlist]

    print(f"layout={layout.label} trading_levels={layout.trading_levels} 穷举={len(exhaustive)} 短名单={len(shortlist)}")
    print("穷举精确分 top5:")
    for score, names in exact_rows[:5]:
        rank = shortlist_names.index(names) + 1 if names in shortlist_names else None
        print(f"  {score:8.3f} {' '.join(names)}  短名单名次={rank}")
    print("短名单里精确分 top5（重算）:")
    scored = sorted(
        (
            (hybrid._score_room_combo(spec, list(combo.operators), duration), combo)
            for combo in shortlist
        ),
        key=lambda row: row[0],
        reverse=True,
    )
    for score, combo in scored[:5]:
        print(f"  {score:8.3f} {' '.join(skill.operator_name for skill in combo.operators)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
