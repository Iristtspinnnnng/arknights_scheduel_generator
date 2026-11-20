"""为下一次窗口定锚：O_DIAMOND 校准改成「速度参与」后，怎么保持 282/天 不变。

- guide 口径：任意 L3 O_DIAMOND 房间的速度都被覆盖成 132%，12h 产出 282.0 orundum；
- raw 口径：速度参与（但会被订单上限裁剪）；
- 这里量出「raw 口径下，正好 132% 速度」的 12h 产出，得到缩放比 r = 282 / raw@132%，
  下个窗口按 r 缩放即可在不移动基准的前提下恢复梯度。
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.models import BaseSkill, RoomAssignment, ShiftPlan
from arknights_schedule_generator.production import ProductionSimulator

OUT = Path("outputs/orundum_anchor.md")


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


def orundum_for(profile: str, speeds: list[float], hours: float = 12.0) -> float:
    game_data = GameData.load(Path("data/cache"))
    simulator = ProductionSimulator(
        game_data,
        shard_formula="rock",
        drone_policy="none",
        calibration_profile=profile,
    )
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


def main() -> int:
    rows: list[str] = ["# O_DIAMOND 校准锚点", ""]
    guide_reference = orundum_for("guide", [0.0])
    rows.append(f"- guide 口径（速度被覆盖成 132%）：12h = **{guide_reference:.3f}** orundum")
    for speeds in ([0.0], [32.0], [32.0, 32.0, 32.0], [60.0], [100.0]):
        value = orundum_for("raw", speeds)
        ratio = guide_reference / value if value else float("nan")
        rows.append(
            f"- raw 口径，速度加成 {speeds}：12h = {value:.3f} orundum"
            f"（要达到 guide 的 {guide_reference:.3f}，缩放比 r = {ratio:.4f}）"
        )
    rows.append("")
    rows.append(
        "读法：下个窗口把 O_DIAMOND 改成走 `trade_order_profile` + `room_speed_multiplier`"
        " 之后，再乘以一个常数，使「参照速度」下仍等于 guide 的基准值；"
        "这样只有**不同干员之间**的相对差异变化，绝对量级不变。"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
