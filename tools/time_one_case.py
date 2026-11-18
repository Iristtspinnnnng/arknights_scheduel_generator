"""单场景计时/对照：``optimize()`` 墙钟与分数（M1 前后对比用）。

不依赖仓库内 data/cache，用绝对路径指向主工作区，所以可以在 baseline worktree 里跑。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--layout", default="243")
    parser.add_argument("--mode", default="balanced-orundum")
    parser.add_argument("--shift-count", type=int, default=2)
    parser.add_argument("--shift-hours", type=int, default=12)
    parser.add_argument("--drone-policy", default="none")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    game_data = GameData.load(Path(args.data_dir))
    roster = load_roster_xlsx(Path(args.roster))
    times: list[float] = []
    score = 0.0
    for _ in range(max(1, args.repeat)):
        optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
        started = time.perf_counter()
        result = optimizer.optimize(
            parse_layout(args.layout),
            mode=args.mode,
            shift_count=args.shift_count,
            shift_hours=args.shift_hours,
            drone_policy=args.drone_policy,
        )
        times.append(time.perf_counter() - started)
        score = result.score
    print(f"score={score:.3f} wall={','.join(f'{value:.2f}' for value in times)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
