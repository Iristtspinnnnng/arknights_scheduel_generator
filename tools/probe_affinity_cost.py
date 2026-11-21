"""M3b 探针：量 `_score_room_combo` 的单次成本，以及各房间规格的样本预算。

M3b（房间节点 + 亲和度矩阵）需要**跨规格池化**拟合一张共享读出层，
所以先要知道一次模拟器打分多少钱、一个规格的候选池有多大。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import GraphScheduleOptimizer
from arknights_schedule_generator.optimizer import RoomSpec
from arknights_schedule_generator.roster import load_roster_xlsx

DATA = Path("data/cache")
ROSTER = Path("examples/fixtures/yituliu_full_roster_maxed.xlsx")

SPECS = (
    ("TRADING", "O_GOLD"),
    ("TRADING", "O_DIAMOND"),
    ("MANUFACTURE", "F_GOLD"),
    ("MANUFACTURE", "F_EXP"),
    ("MANUFACTURE", "F_DIAMOND"),
    ("POWER", None),
)


def main() -> int:
    game_data = GameData.load(DATA)
    roster = load_roster_xlsx(ROSTER)
    optimizer = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
    for room_type, target in SPECS:
        for level in (1, 3):
            slots = optimizer._room_capacity(room_type, level)
            spec = RoomSpec(
                f"{room_type.lower()}_probe", room_type, target, slots, level, slots
            )
            variants = optimizer._best_variant_map(spec, 12.0)
            names = sorted(variants)
            if not names:
                continue
            pool = optimizer._combo_candidate_pool(room_type, target, set())
            # 采样 60 个组合：先全部消掉缓存效应，再计时
            combos = [tuple(names[index % len(names)] for index in range(slots))]
            started = time.perf_counter()
            evaluated = 0
            for offset in range(60):
                picked = tuple(
                    names[(offset * 3 + index * 7) % len(names)] for index in range(slots)
                )
                if len(set(picked)) != len(picked):
                    continue
                optimizer._score_room_combo(
                    spec, [variants[name] for name in picked], 12.0
                )
                evaluated += 1
            elapsed = time.perf_counter() - started
            print(
                f"{room_type:12s} {str(target):10s} L{level} slots={slots} "
                f"pool={len(pool):3d} variants={len(variants):3d} "
                f"evals={evaluated:3d} total={elapsed * 1000:7.1f}ms "
                f"per_eval={elapsed / max(1, evaluated) * 1000:5.2f}ms"
            )
            _ = combos
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
