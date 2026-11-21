"""确认「短名单漏掉穷举最优」到底是不是真的覆盖缺口（而不是同名不同技能变体）。

`tools/probe_shortlist_coverage_under_exclusions.py` 只比**人名集合**，所以「同名但换了一个
技能变体」会被误判成漏掉。这里对同一批房间同时比较：

* 人名集合是否出现在短名单里；
* 变体键（`char/buff_id`）是否出现在短名单里；
* 父类穷举最优组合的**变体键**是否在短名单里。

三种结果的含义完全不同：
- 人名在、变体不在 → 是 `_pick_variant` 的选择问题，不是覆盖问题；
- 人名不在 → 真的没提名到，属于覆盖缺口。
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import GraphScheduleOptimizer
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx

DATA = Path("data/cache")
ROSTER = Path("examples/fixtures/yituliu_full_roster_maxed.xlsx")

CASES = (
    ("243", "normal", 2, 12, "none"),
    ("333", "balanced-orundum", 3, 8, "auto"),
)
SAMPLE = 6


def variant_key(skills) -> frozenset:
    return frozenset((skill.operator_name, skill.buff_id) for skill in skills)


def main() -> int:
    game_data = GameData.load(DATA)
    roster = load_roster_xlsx(ROSTER)
    for layout_raw, mode, count, hours, drones in CASES:
        optimizer = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
        captured: list[tuple] = []
        original = optimizer._joint_production_room_assignments

        def patched(specs, boundary_excluded, _original=original, **kwargs):
            result = _original(specs, boundary_excluded, **kwargs)
            duration = kwargs["duration_hours"]
            for spec in specs:
                room = result[spec.room_id]
                chosen = tuple(sorted(s.operator_name for s in room.operators))
                shortlist = optimizer._room_combos(spec, boundary_excluded, duration)
                rank = None
                for index, combo in enumerate(shortlist):
                    if tuple(sorted(s.operator_name for s in combo.operators)) == chosen:
                        rank = index
                        break
                captured.append((spec, frozenset(boundary_excluded), duration, rank, room.score))
            return result

        optimizer._joint_production_room_assignments = patched
        result = optimizer.optimize(
            parse_layout(layout_raw),
            mode=mode,
            shift_count=count,
            shift_hours=hours,
            drone_policy=drones,
        )
        optimizer._joint_production_room_assignments = original
        print(f"=== {layout_raw} {mode} {count}x{hours} score={result.score:.3f}")
        worst = sorted(
            (row for row in captured if row[3] is not None), key=lambda row: -(row[3] or 0)
        )[:SAMPLE]
        for spec, excluded, duration, rank, score in worst:
            exhaustive = ScheduleOptimizer._room_combos(optimizer, spec, set(excluded), duration)
            if not exhaustive:
                continue
            best = exhaustive[0]
            shortlist = optimizer._room_combos(spec, set(excluded), duration)
            best_names = frozenset(s.operator_name for s in best.operators)
            best_variants = variant_key(best.operators)
            names_present = any(
                frozenset(s.operator_name for s in combo.operators) == best_names
                for combo in shortlist
            )
            variants_present = any(
                variant_key(combo.operators) == best_variants for combo in shortlist
            )
            print(
                f"    {spec.room_id:14s} {spec.target or '-':10s} rank={rank:3d}/{len(shortlist):3d} "
                f"chosen={score:8.3f} exhaustiveTop={best.score:8.3f} "
                f"names={names_present} variants={variants_present}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
