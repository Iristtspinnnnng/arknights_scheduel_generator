"""在每个优化器的**基础解码层**上对比贸易站选人（不受错峰展开重复人干扰）。

对 hybrid / graph 各自的 ``_build_shifts`` 结果（2 个基础班次）：
1. 用模拟器精确分重算每间生产房间；
2. 逐房间给出人选与分差；
3. 对 graph 差的房间，用它当时真实的排除集重建短名单，回答两个问题：
   - hybrid 选中的组合在短名单里吗？精确分排第几？（覆盖问题）
   - 短名单里精确分最高的组合，被 graph 选中了吗？差多少？（排序/搜索问题）
   - 若没选中，是它的成员被 graph 的别的房间占用了吗？（全局分配问题）

    python tools/diff_normal_base_decode.py --layout 342 --mode normal \
        --report outputs/base_decode_342.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import GraphScheduleOptimizer
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx


def names(room) -> tuple[str, ...]:
    return tuple(skill.operator_name for skill in room.operators)


def build(optimizer, layout, mode: str, durations: list[float]):
    return optimizer._build_shifts(
        layout,
        optimizer._room_specs(layout, mode),
        shift_durations=durations,
        shift_times=["08:00", "20:00"],
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", default="examples/fixtures/yituliu_full_roster_maxed.xlsx")
    parser.add_argument("--data-dir", default="data/cache")
    parser.add_argument("--layout", default="342")
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--report", default="outputs/base_decode.md")
    args = parser.parse_args()

    game_data = GameData.load(Path(args.data_dir))
    roster = load_roster_xlsx(Path(args.roster))
    layout = parse_layout(args.layout)
    durations = [12.0, 12.0]

    graph_optimizer = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
    graph_optimizer.operator_anchor_preference = set()
    graph_shifts = build(graph_optimizer, layout, args.mode, durations)
    hybrid_optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
    hybrid_optimizer.operator_anchor_preference = set()
    hybrid_shifts = build(hybrid_optimizer, layout, args.mode, durations)

    lines: list[str] = []
    add = lines.append
    add(f"# 基础解码层的贸易站选人对比：{args.layout} / {args.mode}")
    add("")
    add("两张表都用 `_score_room_combo`（模拟器口径）重算，不取读出层预测分。")
    add("")

    # 基础解码的排除集：班次 0 无排除；班次 1 排除班次 0 的在岗者（最后一个班次还排除首个）。
    exclusions: list[set[str]] = []
    previous: set[str] = set()
    first: set[str] = set()
    for index, shift in enumerate(hybrid_shifts):
        excluded = set(previous)
        if index == len(hybrid_shifts) - 1 and first:
            excluded |= first
        exclusions.append(excluded)
        active = {skill.operator_name for room in shift.rooms for skill in room.operators}
        if index == 0:
            first = set(active)
        previous = set(active)

    for shift_index, hybrid_shift in enumerate(hybrid_shifts):
        if shift_index >= len(graph_shifts):
            break
        graph_shift = graph_shifts[shift_index]
        graph_rooms = {room.room_id: room for room in graph_shift.rooms}
        add(f"## 基础班次 {shift_index + 1}（排除 {len(exclusions[shift_index])} 人）")
        add("")
        add("| 房间 | hybrid（精确） | graph（精确） | 差 |")
        add("| --- | --- | --- | --- |")
        deltas: list[tuple[str, float]] = []
        for hybrid_room in hybrid_shift.rooms:
            if hybrid_room.room_type not in {"TRADING", "MANUFACTURE", "POWER"}:
                continue
            graph_room = graph_rooms.get(hybrid_room.room_id)
            if graph_room is None:
                continue
            spec = hybrid_optimizer._spec_from_assignment(hybrid_room)
            left = hybrid_optimizer._score_room_combo(
                spec, list(hybrid_room.operators), float(hybrid_shift.duration_hours)
            )
            right = hybrid_optimizer._score_room_combo(
                spec, list(graph_room.operators), float(graph_shift.duration_hours)
            )
            deltas.append((hybrid_room.room_id, right - left))
            add(
                f"| {hybrid_room.room_id} {hybrid_room.room_type}/{hybrid_room.target} "
                f"| {' '.join(names(hybrid_room))} ({left:.2f}) "
                f"| {' '.join(names(graph_room))} ({right:.2f}) | {right - left:+.2f} |"
            )
        add("")
        add(f"本班次精确分差合计（graph − hybrid）= {sum(d for _, d in deltas):+.2f}")
        add("")

        # 对 graph 明显更差的贸易站房间做短名单检查
        for room_id, delta in deltas:
            if delta > -1.0:
                continue
            hybrid_room = next(room for room in hybrid_shift.rooms if room.room_id == room_id)
            graph_room = graph_rooms[room_id]
            spec = graph_optimizer._spec_from_assignment(graph_room)
            excluded = exclusions[shift_index]
            combos = graph_optimizer._room_combos(
                spec, set(excluded), float(graph_shift.duration_hours)
            )
            graph_active = {
                skill.operator_name for room in graph_shift.rooms for skill in room.operators
            }
            hybrid_combo = names(hybrid_room)
            hybrid_row = None
            best = None
            for rank, combo in enumerate(combos, 1):
                exact = hybrid_optimizer._score_room_combo(
                    spec, list(combo.operators), float(graph_shift.duration_hours)
                )
                combo_names = tuple(skill.operator_name for skill in combo.operators)
                if combo_names == hybrid_combo:
                    hybrid_row = (rank, combo.score, exact)
                if best is None or exact > best[2]:
                    best = (rank, combo.score, exact, combo_names)
            graph_row = None
            for rank, combo in enumerate(combos, 1):
                if tuple(skill.operator_name for skill in combo.operators) == names(graph_room):
                    exact = hybrid_optimizer._score_room_combo(
                        spec, list(combo.operators), float(graph_shift.duration_hours)
                    )
                    graph_row = (rank, combo.score, exact)
                    break
            add(f"### {room_id} {spec.room_type}/{spec.target} L{spec.room_level}（graph 差 {delta:+.2f}）")
            add("")
            add(
                "- hybrid 组合（精确 {:.2f}）在 graph 短名单里：{}".format(
                    hybrid_optimizer._score_room_combo(
                        spec, list(hybrid_room.operators), float(graph_shift.duration_hours)
                    ),
                    f"有，第 {hybrid_row[0]} 名，预测 {hybrid_row[1]:.3f}"
                    if hybrid_row
                    else "**没有**（短名单覆盖不足）",
                )
            )
            if graph_row:
                add(
                    "- graph 选中组合：短名单第 {} 名，预测 {:.3f}，精确 {:.2f}".format(
                        graph_row[0], graph_row[1], graph_row[2]
                    )
                )
            else:
                add("- graph 选中组合：不在（无排除集的）短名单里，说明它是在 beam 里用剩余干员兜出来的")
            if best:
                add(
                    "- 短名单精确分最高：第 {} 名 {:.2f}（预测 {:.3f}，{}）".format(
                        best[0], best[2], best[1], " ".join(best[3])
                    )
                )
                if graph_row:
                    add(f"  - 与 graph 选中相差 {best[2] - graph_row[2]:+.2f}")
                if hybrid_row:
                    add(f"  - 与 hybrid 组合相差 {best[2] - hybrid_row[2]:+.2f}")
            if hybrid_row is None and best:
                landed = {
                    skill.operator_name: room.room_id
                    for room in graph_shift.rooms
                    for skill in room.operators
                }
                add("- hybrid 组合成员的归属（graph 班次内）：")
                for operator in hybrid_combo:
                    where = landed.get(operator, "（离岗/宿舍）")
                    add(
                        f"  - {operator} → {where}"
                        + ("（同房）" if where == room_id else "")
                        + (
                            ""
                            if where in {"（离岗/宿舍）", room_id}
                            else f"；该成员被 {where} 占用了"
                        )
                    )
            add("")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {report} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
