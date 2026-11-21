"""覆盖修复的两个候选旋钮：`beam`（一元预筛宽度）与 `lexicographic_prefix`。

诊断结论（`tools/probe_shortlist_coverage_under_exclusions.py`）：带排除集时短名单会漏掉
穷举最优组合。最可能的机制是 `rank_combinations(beam=18)` 只让**一元预测前 18 名**参加组合
枚举，候选池里第 19~32 名的干员永远进不了短名单（除非被闭包/前缀/机制超边提名）。

本脚本用**只改配置、不改代码**的方式试四个组合，打印分数与墙钟：

    baseline        beam=18, prefix=96   （现状）
    wide-unary      beam=32              （全部 32 人参加枚举）
    wide-prefix     prefix=192
    both

三个场景：243 不搓玉 2×12、333 平衡搓玉 3×8、342 不搓玉 2×12（图卷积目前小负的场景）。
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_optimizer import (
    GraphOptimizerConfig,
    GraphScheduleOptimizer,
)
from arknights_schedule_generator.optimizer import parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx

DATA = Path("data/cache")
ROSTER = Path("examples/fixtures/yituliu_full_roster_maxed.xlsx")

CASES = (
    ("243", "normal", 2, 12, "none"),
    ("333", "balanced-orundum", 3, 8, "auto"),
    ("342", "normal", 2, 12, "none"),
)
VARIANTS = (
    ("baseline", {"beam": 18, "lexicographic_prefix": 96}),
    ("wide-unary", {"beam": 32, "lexicographic_prefix": 96}),
    ("wide-prefix", {"beam": 18, "lexicographic_prefix": 192}),
    ("both", {"beam": 32, "lexicographic_prefix": 192}),
)


def main() -> int:
    game_data = GameData.load(DATA)
    roster = load_roster_xlsx(ROSTER)
    print(f"{'variant':12s} {'case':28s} {'score':>9s} {'shortlist':>9s} {'wall':>7s}")
    for label, overrides in VARIANTS:
        for layout, mode, count, hours, drones in CASES:
            config = GraphOptimizerConfig(**overrides)
            optimizer = GraphScheduleOptimizer(
                game_data, roster, shard_formula="rock", graph_config=config
            )
            started = time.perf_counter()
            result = optimizer.optimize(
                parse_layout(layout),
                mode=mode,
                shift_count=count,
                shift_hours=hours,
                drone_policy=drones,
            )
            elapsed = time.perf_counter() - started
            shortlist = result.diagnostic_insertion_search.get("graphModel", {}).get(
                "shortlistCount"
            )
            counters = getattr(optimizer, "_runtime_counters", {}) or {}
            shortlist_count = counters.get("graphShortlist.count", 0)
            print(
                f"{label:12s} {layout + ' ' + mode + ' ' + str(count) + 'x' + str(hours):28s} "
                f"{result.score:9.3f} {shortlist_count:9.0f} {elapsed:6.1f}s"
            )
            del shortlist
    return 0


if __name__ == "__main__":
    sys.exit(main())
