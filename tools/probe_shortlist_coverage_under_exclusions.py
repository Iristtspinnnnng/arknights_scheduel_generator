"""决定性测量：图短名单在**带排除集**时是否漏掉穷举最优组合。

「逐房吃人」有三个候选病因，本脚本用来分辨最后一个：
1. 候选顺序 —— 已排除（短名单本来就全部精确打分 + 精确排序）；
2. 候选条数 / beam 宽度 —— 已排除（放宽后分数一格不动）；
3. **短名单在具体排除集下的覆盖**：短名单是按 `boundary_excluded` 重建的，
   如果它漏掉了该排除集下的穷举最优组合，那 beam 再宽也没用。

做法：把解码时每个房间的 `(spec, 排除集)` 抓下来，对其中最差的若干个
分别算「图短名单 top-1」与「父类穷举 top-1」，比较精确分。
只采样最差的几个房间，避免把 4638 组穷举全跑一遍。
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
                captured.append(
                    (spec, frozenset(boundary_excluded), duration, rank, room.score, len(shortlist))
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
        optimizer._joint_production_room_assignments = original
        print(f"=== {layout_raw} {mode} {count}x{hours} score={result.score:.3f}")
        worst = sorted(
            (row for row in captured if row[3] is not None), key=lambda row: -(row[3] or 0)
        )[:SAMPLE]
        total_gap = 0.0
        for spec, excluded, duration, rank, score, size in worst:
            exhaustive = ScheduleOptimizer._room_combos(optimizer, spec, set(excluded), duration)
            best = exhaustive[0] if exhaustive else None
            best_score = best.score if best is not None else float("nan")
            gap = best_score - score
            total_gap += max(0.0, gap)
            names = ",".join(s.operator_name for s in best.operators) if best else "-"
            present = False
            if best is not None:
                wanted = frozenset(s.operator_name for s in best.operators)
                present = any(
                    frozenset(s.operator_name for s in combo.operators) == wanted
                    for combo in optimizer._room_combos(spec, set(excluded), duration)
                )
            print(
                f"    {spec.room_id:14s} {spec.target or '-':10s} rank={rank:3d}/{size:3d} "
                f"chosen={score:8.3f} exhaustiveTop={best_score:8.3f} gap={gap:+7.3f} "
                f"topInShortlist={present}  [{names}]"
            )
        print(f"    sampled coverage gap sum = {total_gap:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
