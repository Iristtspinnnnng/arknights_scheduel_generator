"""查数据：菲亚梅塔的宿舍技能文本 + O_DIAMOND 贸易站是否对干员敏感。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.models import RoomAssignment
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.production import ProductionSimulator
from arknights_schedule_generator.roster import load_roster_xlsx

OUT = Path("outputs/shard_probe.md")


def main() -> int:
    lines: list[str] = []
    add = lines.append

    building = json.loads(Path("data/cache/building_data.json").read_text(encoding="utf-8"))
    add("# 患难之交 / 源石类配方的数据核对")
    add("")
    add("## 含「患难」的 buff")
    add("")
    hits = [
        (key, value)
        for key, value in (building.get("buffs") or {}).items()
        if "患难" in str(value.get("description") or "")
    ]
    add(f"命中 {len(hits)} 条")
    add("")
    for key, value in hits:
        add(f"- `{key}` {value.get('buffName')} / {value.get('roomType')}")
        add(f"  - {value.get('description')}")
    add("")

    add("## 宿舍技能里含「心情恢复」且描述带「满」的 buff（找重置类）")
    add("")
    count = 0
    for key, value in (building.get("buffs") or {}).items():
        if value.get("roomType") != "DORMITORY":
            continue
        description = str(value.get("description") or "")
        if "心情" in description and ("满" in description or "换" in description):
            add(f"- `{key}` {value.get('buffName')}：{description}")
            count += 1
    add(f"共 {count} 条")
    add("")

    add("## 源石类配方：制造站/贸易站是否吃干员技能")
    add("")
    game_data = GameData.load(Path("data/cache"))
    roster = load_roster_xlsx(Path("examples/fixtures/yituliu_full_roster_maxed.xlsx"))
    optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
    spec = parse_layout("243")
    specs = {room.room_type: room for room in optimizer._room_specs(spec, "balanced-orundum")}
    trade_spec = next(
        room
        for room in optimizer._room_specs(spec, "balanced-orundum")
        if room.room_type == "TRADING" and room.target == "O_DIAMOND"
    )
    combos = optimizer._room_combos(trade_spec, set(), 12.0)
    add(f"- O_DIAMOND 规格 {trade_spec.room_id} L{trade_spec.room_level}，全量组合 {len(combos)} 个")
    scores = sorted(
        (
            optimizer._score_room_combo(trade_spec, list(combo.operators), 12.0),
            tuple(skill.operator_name for skill in combo.operators),
        )
        for combo in combos
    )
    add(f"- 精确分区间：{scores[0][0]:.3f} ~ {scores[-1][0]:.3f}")
    add(f"- 最差组合：{' '.join(scores[0][1])}")
    add(f"- 最好组合：{' '.join(scores[-1][1])}")
    add("")

    # 用同一批组合，直接看模拟器输出的 orundum（guide vs 非 guide）
    add("### 同一批组合下 O_DIAMOND 的 orundum 输出")
    add("")
    for profile in ("guide", "raw"):
        simulator = ProductionSimulator(
            game_data,
            shard_formula="rock",
            drone_policy="none",
            calibration_profile=profile,
        )
        rows = []
        for combo in combos[:50]:
            room = RoomAssignment(
                "trading_1",
                "TRADING",
                "贸易站",
                "O_DIAMOND",
                list(combo.operators),
                0.0,
                room_level=trade_spec.room_level,
                slots=trade_spec.slots,
            )
            production = simulator.evaluate_room(room, 12.0) if hasattr(simulator, "evaluate_room") else None
            if production is None:
                report = simulator.evaluate(
                    [
                        __import__(
                            "arknights_schedule_generator.models", fromlist=["ShiftPlan"]
                        ).ShiftPlan("A", "08:00", 12.0, [room], [])
                    ]
                )
                rows.append(report.dailyExpected.orundum)
            else:
                rows.append(production.vector.orundum)
        add(
            f"- `{profile}`：前 50 个组合的 orundum 取值 {len(set(round(v, 6) for v in rows))} 种"
            f"，区间 {min(rows):.3f} ~ {max(rows):.3f}"
        )
    add("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
