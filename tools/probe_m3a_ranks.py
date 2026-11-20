"""M3a 复核：解码时每个房间到底选到自己**精确排名**的第几名？

handoff §3 第 6 步把 M3a 写成「短名单 top-K 用 `_score_room_combo` 重排后再进 beam」。
如果图优化器的短名单本来就是**全部精确打分并按精确分排序**，那 M3a 是个 no-op；
真正决定「逐房吃人」的是 beam 的**选择**（跨房间/跨班次），不是候选顺序。
本脚本量后者：对每个房间记录解码选中的组合在「本次排除集合下的精确排名」。
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import GraphScheduleOptimizer
from arknights_schedule_generator.optimizer import parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx

DATA = Path("data/cache")
ROSTER = Path("examples/fixtures/yituliu_full_roster_maxed.xlsx")

CASES = (
    ("243", "normal", 2, 12, "none"),
    ("342", "normal", 2, 12, "none"),
    ("333", "balanced-orundum", 3, 8, "auto"),
)


def main() -> int:
    game_data = GameData.load(DATA)
    roster = load_roster_xlsx(ROSTER)
    for layout_raw, mode, count, hours, drones in CASES:
        optimizer = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
        rows: list[tuple[str, str, str, int, object, int]] = []
        original = optimizer._joint_production_room_assignments

        def patched(specs, boundary_excluded, *, duration_hours, _original=original, **kwargs):
            result = _original(
                specs, boundary_excluded, duration_hours=duration_hours, **kwargs
            )
            for spec in specs:
                room = result[spec.room_id]
                chosen = tuple(sorted(skill.operator_name for skill in room.operators))
                full = optimizer._room_combos(spec, boundary_excluded, duration_hours)
                rank = None
                for index, combo in enumerate(full):
                    if tuple(sorted(s.operator_name for s in combo.operators)) == chosen:
                        rank = index
                        break
                rows.append(
                    (spec.room_id, spec.room_type, spec.target or "-", len(chosen), rank, len(full))
                )
            return result

        optimizer._joint_production_room_assignments = patched
        result = optimizer.optimize(
            parse_layout(layout_raw),
            mode=mode,
            shift_count=count,
            shift_hours=hours,
            drone_policy=drones,
        )
        ranks = [row[4] for row in rows if row[4] is not None]
        missing = sum(1 for row in rows if row[4] is None)
        print(
            f"=== {layout_raw} {mode} {count}x{hours} score={result.score:.3f} "
            f"rooms={len(rows)} unranked={missing}"
        )
        if ranks:
            ranks_sorted = sorted(ranks)
            print(
                f"    chosen rank: min {ranks_sorted[0]} / median "
                f"{ranks_sorted[len(ranks_sorted) // 2]} / max {ranks_sorted[-1]}"
                f"  (shortlist sizes {sorted({row[5] for row in rows})})"
            )
            worst = sorted((row for row in rows if row[4] is not None), key=lambda r: -(r[4] or 0))[:5]
            for row in worst:
                print(f"    worst: {row[0]:14s} {row[2]:10s} n={row[3]} rank={row[4]} of {row[5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
