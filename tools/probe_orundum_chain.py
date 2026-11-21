"""诊断：搓玉（O_DIAMOND）为什么拿不到分 —— 目标函数对齐 + 源石碎片链路。

回答两个问题：

1. **逐房间的目标（`room_output_score`）和全局目标（`score_breakdown_for`）是不是同一个口径？**
   本脚本对每间生产房把它自己的短名单 top-K 逐个替进排班，再算**整条排班**的分，
   然后看「房间分名次」和「总分名次」的秩相关。秩相关低/为负 -> 逐房间解码在优化别的东西。
2. **源石碎片是不是真正的约束？模型有没有把它当约束？**
   算出制造站的碎片产能上限（10 玉/碎片）、当前排班的玉产量与碎片收支，
   再对比社区参考值，看「表现差」是产能上限卡住，还是目标函数把边际收益定成了负的。

用法::

    python tools/probe_orundum_chain.py --layout 243 --mode balanced-orundum --drone-policy none
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import GraphScheduleOptimizer
from arknights_schedule_generator.models import ShiftPlan
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.power import apply_right_side_preset
from arknights_schedule_generator.production import (
    ORUNDUM_PER_SHARD_ORDER,
    ProductionSimulator,
    SHARDS_PER_ORUNDUM_ORDER,
    score_breakdown_for,
)
from arknights_schedule_generator.roster import load_roster_xlsx

DATA = Path("data/cache")
ROSTER = Path("examples/fixtures/yituliu_full_roster_maxed.xlsx")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            result[order[index]] = average
        position = end + 1
    return result


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 3:
        return 0.0
    rank_l = ranks(left)
    rank_r = ranks(right)
    n = len(left)
    mean_l = sum(rank_l) / n
    mean_r = sum(rank_r) / n
    numerator = sum((rank_l[i] - mean_l) * (rank_r[i] - mean_r) for i in range(n))
    denominator = (
        sum((value - mean_l) ** 2 for value in rank_l)
        * sum((value - mean_r) ** 2 for value in rank_r)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def substitute_room(
    shifts: list[ShiftPlan], room_index: int, skills
) -> list[ShiftPlan]:
    """把同组班次里的同一间房换成 ``skills``（其余房间不动）。"""
    target = {
        skill.operator_name for skill in shifts[0].rooms[room_index].operators
    }
    updated: list[ShiftPlan] = []
    for shift in shifts:
        room = shift.rooms[room_index]
        names = {skill.operator_name for skill in room.operators}
        if names != target:
            updated.append(shift)
            continue
        rooms = list(shift.rooms)
        rooms[room_index] = replace(room, operators=list(skills))
        updated.append(
            ShiftPlan(
                name=shift.name,
                start=shift.start,
                duration_hours=shift.duration_hours,
                rooms=rooms,
                dormitories=list(shift.dormitories),
            )
        )
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="243")
    parser.add_argument("--mode", default="balanced-orundum")
    parser.add_argument("--drone-policy", default="none")
    parser.add_argument("--shift-count", type=int, default=2)
    parser.add_argument("--shift-hours", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--readout", default="per_room")
    args = parser.parse_args()

    game_data = GameData.load(DATA)
    roster = load_roster_xlsx(ROSTER)
    graph_config = None
    if args.readout != "per_room":
        from arknights_schedule_generator.graph_optimizer import GraphOptimizerConfig

        graph_config = GraphOptimizerConfig(readout=args.readout)
    optimizer_cls = ScheduleOptimizer if args.readout == "hybrid" else GraphScheduleOptimizer
    optimizer = (
        optimizer_cls(game_data, roster, shard_formula="rock")
        if graph_config is None
        else optimizer_cls(game_data, roster, shard_formula="rock", graph_config=graph_config)
    )
    layout = apply_right_side_preset(parse_layout(args.layout), "full")
    result = optimizer.optimize(
        layout,
        mode=args.mode,
        shift_count=args.shift_count,
        shift_hours=args.shift_hours,
        drone_policy=args.drone_policy,
    )
    shifts = result.shifts
    duration = float(shifts[0].duration_hours)
    report = result.production_report
    simulator = ProductionSimulator(
        game_data, shard_formula="rock", drone_policy=args.drone_policy, calibration_profile="guide"
    )

    print(f"== {args.layout} {args.mode} {args.shift_count}x{args.shift_hours} "
          f"readout={args.readout} score={result.score:.3f} shifts={len(shifts)}")
    daily = report.dailyExpected
    print(
        f"   orundum/day={daily.orundum:.1f} shards/day(delta)={daily.shardDelta:+.3f} "
        f"lmdGross={daily.lmdGross:.0f} lmdNet={daily.lmdNet:.0f} "
        f"gold={daily.pureGoldDelta:+.2f} exp={daily.exp:.0f}"
    )
    print("   breakdown " + json.dumps({k: round(v, 2) for k, v in score_breakdown_for(report.dailyExpected).items()}, ensure_ascii=False))
    implication = daily.orundum / ORUNDUM_PER_SHARD_ORDER * SHARDS_PER_ORUNDUM_ORDER
    verdict = "碎片是硬约束（缺口）" if daily.shardDelta < -0.05 else "碎片不是缺口"
    print(
        f"   源石碎片需求（玉/10）={implication:.2f}/天，"
        f"碎片收支={daily.shardDelta:+.3f}/天 -> {verdict}"
    )

    # ---- 1. 逐房间目标 vs 全局目标 -------------------------------------------
    print()
    print("-- 目标对齐：房间分名次 vs 整条排班总分名次（同一间房的短名单 top-K）")
    header = f"{'房间':14s} {'目标':10s} {'K':>3s} {'房间分':>9s} {'总分':>9s} {'Δ房间':>8s} {'Δ总分':>8s} {'秩相关':>7s}"
    print(header)
    for room_index, room in enumerate(shifts[0].rooms):
        if room.room_type not in {"TRADING", "MANUFACTURE"}:
            continue
        spec = optimizer._spec_from_assignment(room)
        # 合法替换集合：同班其他房间 + 相邻班次的全部干员，再放开本房间自己的三人。
        excluded: set[str] = set()
        for other in shifts[0].rooms:
            excluded.update(skill.operator_name for skill in other.operators)
        excluded |= optimizer._boundary_active_names(shifts, 0)
        excluded -= {skill.operator_name for skill in room.operators}
        combos = optimizer._room_combos(spec, excluded, duration)
        if len(combos) < 3:
            continue
        room_scores: list[float] = []
        total_scores: list[float] = []
        for combo in combos[: args.top_k]:
            candidate = substitute_room(shifts, room_index, combo.operators)
            # 注意：最终排班本身**允许**相邻班次共用干员（`_active_boundaries_are_valid`
            # 只是建造期的启发式；实测 243 平衡搓玉的最终排班相邻班次共用 13 人），
            # 所以只检查「同一班次内不重复」。
            names = [skill.operator_name for room in candidate[0].rooms for skill in room.operators]
            if len(set(names)) != len(names):
                continue
            candidate_report = simulator.evaluate(candidate)
            room_scores.append(combo.score)
            total_scores.append(candidate_report.score)
        if len(room_scores) < 3:
            continue
        correlation = spearman(room_scores, total_scores)
        print(
            f"{room.room_id:14s} {str(room.target):10s} {len(room_scores):>3d} "
            f"{room_scores[0]:>9.2f} {total_scores[0]:>9.2f} "
            f"{room_scores[0] - room_scores[-1]:>8.2f} {total_scores[0] - total_scores[-1]:>8.2f} "
            f"{correlation:>7.2f}"
        )

    # ---- 2. 碎片产能上限 ------------------------------------------------------
    print()
    print("-- 产能上限：单间制造站能产多少碎片、单间贸易站能吃多少")
    template_shard = next(
        (room for room in shifts[0].rooms if room.target == "F_DIAMOND"),
        replace(shifts[0].rooms[0], room_type="MANUFACTURE", target="F_DIAMOND"),
    )
    for level in (3, 2, 1):
        slots = optimizer._room_capacity("MANUFACTURE", level)
        spec = replace(
            optimizer._spec_from_assignment(template_shard),
            room_type="MANUFACTURE",
            target="F_DIAMOND",
            room_level=level,
            slots=slots,
            capacity=slots,
        )
        combos = optimizer._room_combos(spec, set(), 24.0)
        if not combos:
            continue
        best = combos[0]
        probe_room = replace(
            template_shard,
            room_level=level,
            slots=slots,
            product_capacity=None,
            operators=list(best.operators),
        )
        production = simulator.evaluate_room(probe_room, 24.0)
        shards = production.vector.shardDelta
        lmd_cost = production.vector.materialCosts.get("4001", 0.0)
        print(
            f"   F_DIAMOND L{level} ({slots} 工位) 最佳组 roomScore={best.score:.2f} "
            f"-> {shards:.2f} 碎片/24h = {shards * ORUNDUM_PER_SHARD_ORDER / SHARDS_PER_ORUNDUM_ORDER:.1f} 玉/天"
            f"（碎片成本 {lmd_cost:.0f} 龙门币/24h）"
        )
    template_trade = next(
        (room for room in shifts[0].rooms if room.target == "O_DIAMOND"), None
    )
    if template_trade is not None:
        spec = replace(
            optimizer._spec_from_assignment(template_trade),
            room_type="TRADING",
            target="O_DIAMOND",
        )
        combos = optimizer._room_combos(spec, set(), 24.0)
        print("   O_DIAMOND L3 (3 工位) 前三组：")
        for combo in combos[:3]:
            room = replace(template_trade, operators=list(combo.operators))
            production = simulator.evaluate_room(room, 24.0)
            print(
                f"      roomScore={combo.score:8.2f} -> {production.vector.orundum:7.1f} 玉/24h "
                f"需要 {-production.vector.shardDelta:6.2f} 碎片/24h"
            )

    # ---- 3. 影子价格：把全局分项对玉/碎片/LMD 的边际价值列出来 ------------------
    print()
    print("-- 影子价格（当前口径）")
    vector = report.dailyExpected
    baseline = score_breakdown_for(vector)
    for label, delta in (("+10 玉（+1 碎片消耗）", {"orundum": 10.0, "shardDelta": -1.0}),
                         ("+1 碎片产出", {"shardDelta": 1.0}),
                         ("+1000 龙门币收入", {"lmdGross": 1000.0, "lmdNet": 1000.0}),
                         ("+1 赤金", {"pureGoldDelta": 1.0})):
        probe = replace(vector)
        for key, value in delta.items():
            setattr(probe, key, getattr(probe, key) + value)
        if "lmdNet" in delta:
            probe.materialCosts = dict(vector.materialCosts)
        after = score_breakdown_for(probe)
        print(
            f"   {label:24s} Δ分数 = {sum(after.values()) - sum(baseline.values()):+7.2f}"
        )
    print()
    print("   仅 +10 玉（不进碎片）: ", end="")
    probe = replace(vector)
    probe.orundum += 10.0
    print(f"Δ = {sum(score_breakdown_for(probe).values()) - sum(baseline.values()):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
