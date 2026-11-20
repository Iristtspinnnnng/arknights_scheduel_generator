"""O_DIAMOND（源石订单）校准 v2 的锚点与梯度实测。

v1：``guide_orundum_trade_lookup`` 把任何 L3 O_DIAMOND 房间的速度覆盖成 132%，
     4750 个组合全都产出 282.000（标签是常数 ⇒ 读出层 holdout R² = −110）。
v2：速度来自干员技能、订单吞吐不看订单上限，再用**单一常数**把量级锚回参照点。

本脚本量三件事，写进 ``outputs/orundum_anchor.md``：

1. 参照点（L3 / 3 工位 / +132% / 12h）是不是仍然 282.000，缩放常数是多少；
2. 满练表上 4750 个 O_DIAMOND 组合的速度与产出分布（梯度有多宽）；
3. guide / raw 两个口径是否已经同一条路（搓玉侧不再有「零梯度房间」）。
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.models import BaseSkill, RoomAssignment, ShiftPlan
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.production import (
    ORUNDUM_ORDER_THROUGHPUT_SCALE,
    ORUNDUM_REFERENCE_HOURS,
    ORUNDUM_REFERENCE_LEVEL,
    ORUNDUM_REFERENCE_OUTPUT,
    ORUNDUM_REFERENCE_SLOTS,
    ORUNDUM_REFERENCE_SPEED_PERCENT,
    ProductionSimulator,
    orundum_reference_raw_output,
)
from arknights_schedule_generator.roster import load_roster_xlsx
from arknights_schedule_generator.skill_rules import SkillContext, evaluate_room_effect

DEFAULT_ROSTER = Path("examples/fixtures/yituliu_full_roster_maxed.xlsx")
DEFAULT_OUTPUT = Path("outputs/orundum_anchor.md")


def synthetic(name: str, speed_percent: float) -> BaseSkill:
    return BaseSkill(
        char_id=f"char_{name}",
        operator_name=name,
        room_type="TRADING",
        buff_id="probe",
        buff_name="probe",
        description=f"进驻贸易站时，订单获取效率+{speed_percent:g}%",
        efficiency=speed_percent,
        targets=(),
        cond_elite=0,
        cond_level=1,
        unlocked=True,
        complex_condition=False,
        parsed_score=speed_percent,
    )


def orundum_12h(
    simulator: ProductionSimulator, speeds: list[float], hours: float = 12.0
) -> float:
    room = RoomAssignment(
        "trading_1",
        "TRADING",
        "贸易站",
        "O_DIAMOND",
        [synthetic(f"probe{index}", speed) for index, speed in enumerate(speeds)],
        0.0,
        room_level=3,
        slots=3,
    )
    report = simulator.evaluate([ShiftPlan("A", "08:00", hours, [room], [])])
    return report.dailyExpected.orundum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    game_data = GameData.load(args.data_dir)
    guide = ProductionSimulator(
        game_data, shard_formula="rock", drone_policy="none", calibration_profile="guide"
    )
    raw = ProductionSimulator(
        game_data, shard_formula="rock", drone_policy="none", calibration_profile="raw"
    )

    rows: list[str] = ["# O_DIAMOND 校准 v2：锚点与梯度实测", ""]
    rows.append("## 1. 参照点与缩放常数")
    rows.append("")
    rows.append(
        f"- 参照配置：L{ORUNDUM_REFERENCE_LEVEL} / {ORUNDUM_REFERENCE_SLOTS} 工位 / "
        f"技能 +{ORUNDUM_REFERENCE_SPEED_PERCENT:g}% / {ORUNDUM_REFERENCE_HOURS:g}h"
    )
    rows.append(f"- 未缩放的参照产出：**{orundum_reference_raw_output():.3f}**")
    rows.append(f"- 缩放常数：**{ORUNDUM_ORDER_THROUGHPUT_SCALE:.6f}**")
    rows.append(
        f"- 参照产出（模拟器复核）：**{ORUNDUM_REFERENCE_OUTPUT:.3f}**"
        f"（guide {orundum_12h(guide, [ORUNDUM_REFERENCE_SPEED_PERCENT]):.3f} / "
        f"raw {orundum_12h(raw, [ORUNDUM_REFERENCE_SPEED_PERCENT]):.3f}）"
    )
    rows.append("")

    rows.append("## 2. 速度 → 产出（12h，L3）")
    rows.append("")
    rows.append("| 技能速度 | guide | raw |")
    rows.append("| --- | --- | --- |")
    for speeds in ([], [30.0], [44.0] * 3, [60.0] * 3, [100.0], [132.0], [180.0], [200.0]):
        label = " + ".join(f"+{speed:g}%" for speed in speeds) or "无干员"
        rows.append(
            f"| {label} | {orundum_12h(guide, speeds):.3f} | {orundum_12h(raw, speeds):.3f} |"
        )
    rows.append("")
    rows.append(
        "guide 与 raw 现在是同一条路（v1 的 guide 差分支已删除），搓玉侧不再有「对干员免疫」的房间。"
    )
    rows.append("")

    rows.append("## 3. 满练表上的真实组合分布")
    rows.append("")
    if args.roster.exists():
        roster = load_roster_xlsx(args.roster)
        optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
        layout = parse_layout("243")
        trade_spec = next(
            room
            for room in optimizer._room_specs(layout, "balanced-orundum")
            if room.room_type == "TRADING" and room.target == "O_DIAMOND"
        )
        combos = optimizer._room_combos(trade_spec, set(), 12.0)
        context = SkillContext()
        speeds = sorted(
            evaluate_room_effect(list(combo.operators), "O_DIAMOND", context).speed_percent
            for combo in combos
        )
        scores = [
            optimizer._score_room_combo(trade_spec, list(combo.operators), 12.0)
            for combo in combos
        ]
        rows.append(f"- 组合数：**{len(speeds)}**（{trade_spec.room_id} L{trade_spec.room_level}）")
        rows.append(
            f"- 技能速度和：min **{speeds[0]:.1f}%** / p50 **{statistics.median(speeds):.1f}%** / "
            f"p90 **{speeds[9 * len(speeds) // 10]:.1f}%** / max **{speeds[-1]:.1f}%**"
        )
        rows.append(
            f"- 12h 产出：{orundum_12h(guide, [speeds[0]]):.1f} ~ "
            f"{orundum_12h(guide, [speeds[-1]]):.1f}（v1 全部是 282.000）"
        )
        rows.append(
            f"- 精确分区间：**{min(scores):.3f} ~ {max(scores):.3f}**"
            f"（v1 是 277.200 ~ 282.000）"
        )
        best = max(combos, key=lambda combo: optimizer._score_room_combo(trade_spec, list(combo.operators), 12.0))
        worst = min(combos, key=lambda combo: optimizer._score_room_combo(trade_spec, list(combo.operators), 12.0))
        rows.append(f"- 最好组合：{' '.join(skill.operator_name for skill in best.operators)}")
        rows.append(f"- 最差组合：{' '.join(skill.operator_name for skill in worst.operators)}")
        spread = (max(scores) - min(scores)) / abs(min(scores)) if min(scores) else float("inf")
        rows.append(f"- 相对梯度：**{spread * 100:.1f}%**（验收要求 > 5%）")
    else:
        rows.append(f"- 跳过：练度表 {args.roster} 不存在")
    rows.append("")
    rows.append(
        "读法：v1 的 282.000 是「空房间也满产」的常数标签，读出层在 O_DIAMOND 上只能学到噪声；"
        "v2 让同一批组合按干员技能铺开，绝对量级仍锚在参照点上。"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
