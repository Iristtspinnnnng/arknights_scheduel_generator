"""诊断：同一场景下 hybrid 与 graph 的选人差异，按**房间集合家族**配对后比较精确分。

要点：错峰展开会把基础班次重排/复制，两个优化器的班次顺序不一定对齐，所以不能按
班次下标逐行比。这里先把每条排班的每个班次压成「房间 -> 人选」的指纹，按指纹把班次
归族，再用 ``_score_room_combo``（模拟器精确分，hybrid 的 room.score 就是它）重算，
避免拿 graph 的**读出层预测分**和 hybrid 的精确分直接相减。

    python tools/diagnose_normal_gap.py --layout 342 --mode normal \
        --report outputs/normal_gap_342.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import GraphScheduleOptimizer
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx


def names(room) -> tuple[str, ...]:
    return tuple(skill.operator_name for skill in room.operators)


def shift_signature(shift) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(sorted((room.room_id, names(room)) for room in shift.rooms))


def exact_room_scores(optimizer, shift) -> dict[str, float]:
    """用模拟器精确分重算一间房间的价值（不看 ``RoomAssignment.score``）。"""
    scores: dict[str, float] = {}
    for room in shift.rooms:
        spec = optimizer._spec_from_assignment(room)
        if room.room_type in {"TRADING", "MANUFACTURE", "POWER"}:
            scores[room.room_id] = optimizer._score_room_combo(
                spec, list(room.operators), float(shift.duration_hours)
            )
        else:
            scores[room.room_id] = float(room.score)
    return scores


def group_shifts(optimizer, shifts) -> dict[tuple, list]:
    families: dict[tuple, list] = defaultdict(list)
    for shift in shifts:
        families[shift_signature(shift)].append(shift)
    return families


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", default="examples/fixtures/yituliu_full_roster_maxed.xlsx")
    parser.add_argument("--data-dir", default="data/cache")
    parser.add_argument("--layout", default="342")
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--shift-count", type=int, default=2)
    parser.add_argument("--shift-hours", type=int, default=12)
    parser.add_argument("--drone-policy", default="none")
    parser.add_argument("--report", default="outputs/normal_gap.md")
    args = parser.parse_args()

    game_data = GameData.load(Path(args.data_dir))
    roster = load_roster_xlsx(Path(args.roster))
    layout = parse_layout(args.layout)
    kwargs = dict(
        mode=args.mode,
        shift_count=args.shift_count,
        shift_hours=args.shift_hours,
        drone_policy=args.drone_policy,
    )
    graph_optimizer = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
    graph_result = graph_optimizer.optimize(layout, **kwargs)
    hybrid_optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
    hybrid_result = hybrid_optimizer.optimize(layout, **kwargs)

    lines: list[str] = []
    add = lines.append
    add(f"# 选人差异诊断：{args.layout} / {args.mode} / {args.shift_count}x{args.shift_hours}h / drones={args.drone_policy}")
    add("")
    add(f"- hybrid score = {hybrid_result.score:.3f}（{len(hybrid_result.shifts)} 个班次）")
    add(f"- graph  score = {graph_result.score:.3f}（{len(graph_result.shifts)} 个班次）")
    for label, result in (("hybrid", hybrid_result), ("graph", graph_result)):
        daily = result.production_report.dailyExpected.to_dict()
        add(
            f"- {label} daily：lmdGross={daily['lmdGross']:.1f} lmdNet={daily['lmdNet']:.1f} "
            f"exp={daily['exp']:.1f} orundum={daily['orundum']:.1f} "
            f"officeSpeed={daily['officeSpeed']:.1f} meetingSpeed={daily['meetingSpeed']:.1f}"
        )
    add("")

    hybrid_families = group_shifts(hybrid_optimizer, hybrid_result.shifts)
    graph_families = group_shifts(graph_optimizer, graph_result.shifts)
    add(f"- hybrid 班次族 = {[len(v) for v in hybrid_families.values()]}；graph 班次族 = {[len(v) for v in graph_families.values()]}")
    add("")

    # 以 hybrid 的班次族为基准，找 graph 里「最像」的族（同结构、算子尽量重合）
    add("## 班次族对比（精确分重算）")
    add("")
    for family_index, (signature, shifts) in enumerate(hybrid_families.items(), 1):
        template = shifts[0]
        hybrid_scores = exact_room_scores(hybrid_optimizer, template)
        best_key = None
        best_overlap = -1
        for graph_signature, graph_shifts in graph_families.items():
            overlap = sum(
                1
                for room_id, room_names in signature
                for other_id, other_names in graph_signature
                if room_id == other_id and room_names == other_names
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = graph_signature
        graph_template = graph_families[best_key][0] if best_key is not None else None
        add(f"### 族 {family_index}（hybrid 出现 {len(shifts)} 次，graph 对应族出现 {len(graph_families.get(best_key, []))} 次）")
        add("")
        if graph_template is None:
            add("（graph 没有对应族）")
            continue
        graph_scores = exact_room_scores(graph_optimizer, graph_template)
        graph_rooms = {room.room_id: room for room in graph_template.rooms}
        add("| 房间 | hybrid | graph | 精确分差 |")
        add("| --- | --- | --- | --- |")
        family_delta = 0.0
        for room in template.rooms:
            other = graph_rooms.get(room.room_id)
            if other is None:
                continue
            left = hybrid_scores[room.room_id]
            right = graph_scores[room.room_id]
            family_delta += right - left
            left_names = " ".join(names(room)) or "—"
            right_names = " ".join(names(other)) or "—"
            add(
                f"| {room.room_id} {room.room_type}/{room.target} "
                f"| {left_names} ({left:.2f}) | {right_names} ({right:.2f}) | {right - left:+.2f} |"
            )
        add("")
        add(f"整族精确分差（graph − hybrid）= {family_delta:+.2f} × 出现 {len(shifts)} 次")
        add("")

    # hybrid 的贸易站人选在 graph 对应族里去了哪
    add("## hybrid 的贸易/制造人选在 graph 对应族里的去向")
    add("")
    for family_index, (signature, shifts) in enumerate(hybrid_families.items(), 1):
        template = shifts[0]
        best_key = max(
            graph_families,
            key=lambda graph_signature: sum(
                1
                for room_id, room_names in signature
                for other_id, other_names in graph_signature
                if room_id == other_id and room_names == other_names
            ),
        )
        graph_template = graph_families[best_key][0]
        graph_room_of = {
            skill.operator_name: room.room_id
            for room in graph_template.rooms
            for skill in room.operators
        }
        add(f"### 族 {family_index}")
        add("")
        for room in template.rooms:
            if room.room_type not in {"TRADING", "MANUFACTURE"}:
                continue
            graph_room = next(
                (item for item in graph_template.rooms if item.room_id == room.room_id),
                None,
            )
            if graph_room is None or names(graph_room) == names(room):
                add(f"- {room.room_id}：人选相同（{' '.join(names(room))}）")
                continue
            add(f"- {room.room_id}：hybrid 用 {' '.join(names(room))}；graph 用 {' '.join(names(graph_room))}")
            for skill in room.operators:
                landed = graph_room_of.get(skill.operator_name, "（离岗/宿舍）")
                mark = "同房" if landed == room.room_id else "被挪走"
                add(f"  - {skill.operator_name} → {landed}（{mark}）")
        add("")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {report} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
