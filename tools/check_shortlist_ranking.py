"""在 graph 的实际排班上，检查贸易站短名单的**排序**是否把更好的组合放走了。

对 graph 排班里每间贸易站房间：
1. 用「相邻班次在岗者」当排除集，重建它的短名单（``_room_combos``，按读出层预测分排序）；
2. 用模拟器精确分 ``_score_room_combo`` 重算前 24 个组合；
3. 与 graph 实际选中的组合、以及短名单里精确分最高的组合比较。

若「精确分第一名」不是被选中的那个，说明问题在**读出层排序**（短名单有货但排错），
而不是短名单覆盖不足。

    python tools/check_shortlist_ranking.py --layout 342 --mode normal \
        --report outputs/shortlist_ranking_342.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import GraphScheduleOptimizer
from arknights_schedule_generator.optimizer import parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx


def names(room) -> tuple[str, ...]:
    return tuple(skill.operator_name for skill in room.operators)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", default="examples/fixtures/yituliu_full_roster_maxed.xlsx")
    parser.add_argument("--data-dir", default="data/cache")
    parser.add_argument("--layout", default="342")
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--shift-count", type=int, default=2)
    parser.add_argument("--shift-hours", type=int, default=12)
    parser.add_argument("--drone-policy", default="none")
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--report", default="outputs/shortlist_ranking.md")
    args = parser.parse_args()

    game_data = GameData.load(Path(args.data_dir))
    roster = load_roster_xlsx(Path(args.roster))
    layout = parse_layout(args.layout)
    optimizer = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
    result = optimizer.optimize(
        layout,
        mode=args.mode,
        shift_count=args.shift_count,
        shift_hours=args.shift_hours,
        drone_policy=args.drone_policy,
    )

    lines: list[str] = []
    add = lines.append
    add(f"# 贸易站短名单排序检查：{args.layout} / {args.mode}（graph score={result.score:.3f}）")
    add("")
    add("说明：精确分 = `_score_room_combo`（模拟器口径，和 hybrid 的 room.score 同一把尺）；")
    add("预测分 = 读出层 `combo.score`。排除集 = 相邻班次在岗者。")
    add("")

    for shift_index, shift in enumerate(result.shifts):
        neighbours: set[str] = set()
        for offset in (-1, 1):
            other = result.shifts[(shift_index + offset) % len(result.shifts)]
            neighbours.update(skill.operator_name for room in other.rooms for skill in room.operators)
        add(f"## 班次 {shift_index + 1}（{shift.name}），排除 {len(neighbours)} 名相邻班次在岗者")
        add("")
        for room in shift.rooms:
            if room.room_type != "TRADING":
                continue
            spec = optimizer._spec_from_assignment(room)
            combos = optimizer._room_combos(spec, set(neighbours), float(shift.duration_hours))[
                : args.depth
            ]
            rows = []
            for rank, combo in enumerate(combos, 1):
                exact = optimizer._score_room_combo(
                    spec, list(combo.operators), float(shift.duration_hours)
                )
                rows.append((rank, combo, exact))
            chosen = names(room)
            exact_best = max(rows, key=lambda row: row[2]) if rows else None
            chosen_row = next(
                (row for row in rows if tuple(skill.operator_name for skill in row[1].operators) == chosen),
                None,
            )
            add(f"### {room.room_id} {spec.room_type}/{spec.target} L{spec.room_level}")
            add("")
            add(f"- graph 选中：{' '.join(chosen)}")
            if chosen_row is not None:
                add(
                    f"  - 预测分 {chosen_row[1].score:.3f}（短名单第 {chosen_row[0]} 名），精确分 {chosen_row[2]:.2f}"
                )
            else:
                add("  - ⚠️ 选中组合不在本次重建的短名单前 %d 名里" % args.depth)
            if exact_best is not None:
                add(
                    "  - 短名单里精确分最高："
                    f"{' '.join(skill.operator_name for skill in exact_best[1].operators)} "
                    f"（精确 {exact_best[2]:.2f}，预测 {exact_best[1].score:.3f}，短名单第 {exact_best[0]} 名）"
                )
                if chosen_row is not None:
                    gap = exact_best[2] - chosen_row[2]
                    add(f"  - 精确分差（最好 − 选中）= {gap:+.2f}")
            add("")
            add("| 短名单名次 | 预测分 | 精确分 | 人选 |")
            add("| --- | --- | --- | --- |")
            for rank, combo, exact in rows[:8]:
                add(
                    f"| {rank} | {combo.score:.3f} | {exact:.2f} | "
                    f"{' '.join(skill.operator_name for skill in combo.operators)} |"
                )
            add("")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {report} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
