"""M3a 的第三档实验：把**联合 beam 宽度**放大，看「逐房吃人」是不是 beam 剪枝造成的。

诊断链路：
1. 图短名单本来就是「全部精确打分 + 按精确分排序」，所以 handoff 里的 M3a
   （top-K 再用模拟器重排）是 no-op —— 证据见 `tools/probe_m3a_ranks.py` 的排名统计；
2. 放宽 `JOINT_PRODUCTION_CANDIDATE_LIMIT` 64 → 192 实测**分数一格不动**、墙钟翻倍
   （被排除的次优组合本来就更差，放宽只是多算）；
3. 那剩下的可能就是 beam 宽度：beam 每推进一个房间会把状态剪到 `beam_width`，
   被剪掉的中间状态可能是「某个房间先吃点亏、后面房间能拿到更好组合」的那一支。

本脚本不动代码，直接用 monkeypatch 试几个 beam 宽度，打印分数与墙钟。
"""

from __future__ import annotations

import sys
import time
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
    ("333", "balanced-orundum", 3, 8, "auto"),
)
BEAM_WIDTHS = (256, 1024, 4096)


def main() -> int:
    game_data = GameData.load(DATA)
    roster = load_roster_xlsx(ROSTER)
    for layout_raw, mode, count, hours, drones in CASES:
        print(f"=== {layout_raw} {mode} {count}x{hours} drones={drones}")
        for beam in BEAM_WIDTHS:
            optimizer = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
            original = optimizer._joint_production_room_assignments

            def patched(specs, boundary_excluded, *, duration_hours, _original=original, _beam=beam, **kwargs):
                kwargs.pop("beam_width", None)
                return _original(
                    specs,
                    boundary_excluded,
                    duration_hours=duration_hours,
                    beam_width=_beam,
                    **kwargs,
                )

            optimizer._joint_production_room_assignments = patched
            started = time.perf_counter()
            result = optimizer.optimize(
                parse_layout(layout_raw),
                mode=mode,
                shift_count=count,
                shift_hours=hours,
                drone_policy=drones,
            )
            elapsed = time.perf_counter() - started
            print(f"    beam={beam:5d} score={result.score:9.3f} wall={elapsed:6.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
