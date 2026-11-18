"""§4 风格的可行性扫描：布局 × 模式 × 班次组合 × 心情收支。

用途（M1 验收要求 2）：

* 确认 16 组组合仍然全部通过心情硬门槛；
* 打印每组的最差收支（``moraleLedgerByOperator[*].balance``）与 ``cycleHours``，
  用来对照接手文档 §4 的「最差收支不低于 +4.0」。

用法::

    python tools/scan_m1_dormitory.py --output outputs/scan_m1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx


LAYOUTS = ("243", "252", "342", "333")
MODES = ("normal", "balanced-orundum")
SHIFT_PATTERNS = ((2, 12), (3, 8))


def run_scan(
    *,
    roster_path: Path,
    data_dir: Path,
    drone_policy: str,
    layouts: tuple[str, ...] = LAYOUTS,
    modes: tuple[str, ...] = MODES,
    shift_patterns: tuple[tuple[int, int], ...] = SHIFT_PATTERNS,
) -> list[dict[str, object]]:
    game_data = GameData.load(data_dir)
    roster = load_roster_xlsx(roster_path)
    rows: list[dict[str, object]] = []
    for raw_layout in layouts:
        for mode in modes:
            for shift_count, shift_hours in shift_patterns:
                layout = parse_layout(raw_layout)
                optimizer = ScheduleOptimizer(game_data, roster, shard_formula="rock")
                started = time.perf_counter()
                row: dict[str, object] = {
                    "layout": raw_layout,
                    "mode": mode,
                    "shiftPattern": f"{shift_count}x{shift_hours}",
                }
                try:
                    result = optimizer.optimize(
                        layout,
                        mode=mode,
                        shift_count=shift_count,
                        shift_hours=shift_hours,
                        drone_policy=drone_policy,
                    )
                except ValueError as exc:
                    row.update(
                        {
                            "status": "FAIL",
                            "error": str(exc)[:200],
                            "wallSeconds": round(time.perf_counter() - started, 2),
                        }
                    )
                    rows.append(row)
                    continue
                row["wallSeconds"] = round(time.perf_counter() - started, 2)
                audit = (result.diagnostic_insertion_search or {}).get(
                    "moraleCycleAudit"
                ) or {}
                ledger = audit.get("moraleLedgerByOperator") or {}
                balances = [
                    float(item.get("balance", 0.0)) for item in ledger.values()
                ]
                row.update(
                    {
                        "status": "OK",
                        "score": round(float(result.score), 3),
                        "cycleHours": audit.get("cycleHours"),
                        "planCount": len(result.shifts),
                        "dormitoryLevels": list(result.layout.dormitory_levels),
                        "worstBalance": round(min(balances), 3) if balances else None,
                        "worstBalanceOperator": min(
                            ledger,
                            key=lambda name: float(ledger[name]["balance"]),
                        )
                        if ledger
                        else None,
                        "totalRecovery": round(
                            sum(float(item["recovery"]) for item in ledger.values()), 3
                        ),
                        "totalConsumption": round(
                            sum(float(item["consumption"]) for item in ledger.values()), 3
                        ),
                        "unmodelledMoraleEffects": len(
                            audit.get("unmodelledMoraleEffects") or []
                        ),
                    }
                )
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        default="examples/fixtures/yituliu_full_roster_maxed.xlsx",
    )
    parser.add_argument("--data-dir", default="data/cache")
    parser.add_argument("--drone-policy", default="none")
    parser.add_argument(
        "--layouts",
        default=",".join(LAYOUTS),
        help="逗号分隔，默认 243,252,342,333",
    )
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="逗号分隔，默认 normal,balanced-orundum",
    )
    parser.add_argument(
        "--shift-patterns",
        default=",".join(f"{count}x{hours}" for count, hours in SHIFT_PATTERNS),
        help="逗号分隔，例如 2x12,3x8",
    )
    parser.add_argument("--output", default="outputs/scan_m1_dormitory.json")
    args = parser.parse_args()

    shift_patterns = tuple(
        (int(part.split("x")[0]), int(part.split("x")[1]))
        for part in args.shift_patterns.split(",")
        if part.strip()
    )
    rows = run_scan(
        roster_path=Path(args.roster),
        data_dir=Path(args.data_dir),
        drone_policy=args.drone_policy,
        layouts=tuple(part.strip() for part in args.layouts.split(",") if part.strip()),
        modes=tuple(part.strip() for part in args.modes.split(",") if part.strip()),
        shift_patterns=shift_patterns,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    header = f"{'布局':<4} {'模式':<16} {'班次':<7} {'结果':<5} {'分数':>8} {'周期h':>6} {'最差收支':>9} {'舍等级':<12} {'耗时s':>7}"
    print(header)
    print("-" * len(header))
    passed = 0
    worst = None
    for row in rows:
        if row["status"] == "OK":
            passed += 1
            balance = float(row["worstBalance"])
            worst = balance if worst is None else min(worst, balance)
            print(
                f"{row['layout']:<4} {row['mode']:<16} {row['shiftPattern']:<7} "
                f"{'OK':<5} {row['score']:>8.1f} {row['cycleHours']:>6.1f} "
                f"{balance:>9.1f} {','.join(map(str, row['dormitoryLevels'])):<12} "
                f"{row['wallSeconds']:>7.2f}"
            )
        else:
            print(
                f"{row['layout']:<4} {row['mode']:<16} {row['shiftPattern']:<7} "
                f"{'FAIL':<5} {'-':>8} {'-':>6} {'-':>9} {'-':<12} {row['wallSeconds']:>7.2f}"
            )
            print(f"     -> {row['error']}")
    print(f"\n通过 {passed}/{len(rows)}；最差收支 {worst}")
    print(f"明细写入 {output}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
