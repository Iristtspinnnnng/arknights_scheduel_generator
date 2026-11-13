"""新旧优化器的对照基准。

输出两类指标：

* **质量**：``ProductionReport.score``（同一个模拟器、同一套约束）、每日预期、
  以及每个房间的分数差异。
* **成本**：墙钟时间 + ``evaluateRoom`` 调用次数。

墙钟时间受机器负载影响很大，所以这里同时记录 ``evaluateRoom.comboVector``
调用次数——它是算法层面的确定性成本，跨机器可比。旧算法对一个房间做
``C(32,3)=4960`` 次模拟器评估，新算法只评估短名单里的几十个。
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .data import GameData
from .graph_conv import EncoderConfig
from .graph_features import build_operator_graph
from .graph_optimizer import GraphOptimizerConfig, GraphScheduleOptimizer
from .models import RosterOperator
from .optimizer import ScheduleOptimizer, apply_layout_variant, parse_layout
from .power import apply_right_side_preset

__all__ = [
    "BenchmarkCase",
    "DEFAULT_BENCHMARK_CASES",
    "run_benchmark",
    "render_markdown",
    "measure_shortlist_recall",
]


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    layout: str
    mode: str
    shift_count: int
    shift_hours: int
    drone_policy: str = "none"
    layout_variant: str = "default"
    right_side: str = "full"

    def describe(self) -> str:
        return (
            f"{self.layout} / {self.mode} / {self.shift_count}x{self.shift_hours}h"
            f" / drones={self.drone_policy}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layout": self.layout,
            "mode": self.mode,
            "shiftCount": self.shift_count,
            "shiftHours": self.shift_hours,
            "dronePolicy": self.drone_policy,
            "layoutVariant": self.layout_variant,
            "rightSide": self.right_side,
            "label": self.describe(),
        }


# 注意：自 v0.1.1 起 `optimizer.optimize()` 会做心情周期硬门槛校验
# （`morale.audit_morale_cycle`），不满足就抛 ValueError。历史上 5 制造站的布局
# （252/342）在满练表上凑不出可持续轮换，所以默认场景只用 243 / 333；心情模型补上
# 宿舍绿字（氛围）与控制中枢全局减耗之后 252/342 已可行（见
# `docs/handoff_graph_optimizer.md` §4），默认场景保持不变只是为了让历史基准可比。
DEFAULT_BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("243-balanced-2x12", "243", "balanced-orundum", 2, 12, "none"),
    BenchmarkCase("243-normal-2x12", "243", "normal", 2, 12, "none"),
    BenchmarkCase("333-balanced-3x8-auto", "333", "balanced-orundum", 3, 8, "auto"),
    BenchmarkCase("333-max-2x12-auto", "333", "max-orundum", 2, 12, "auto"),
)

_DAILY_KEYS = ("lmdGross", "lmdNet", "exp", "orundum", "pureGoldDelta", "shardDelta")


def run_benchmark(
    game_data: GameData,
    roster: Sequence[RosterOperator],
    *,
    cases: Iterable[BenchmarkCase] | None = None,
    graph_config: GraphOptimizerConfig | None = None,
    include_hybrid: bool = True,
    repetitions: int = 1,
    recall_sample_specs: int = 6,
) -> dict[str, Any]:
    """跑完整对照并返回可直接序列化的报告字典。"""
    case_list = list(cases or DEFAULT_BENCHMARK_CASES)
    graph_config = graph_config or GraphOptimizerConfig()
    repetitions = max(1, int(repetitions))

    report: dict[str, Any] = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "roster": {
            "rows": len(roster),
            "recruited": sum(1 for operator in roster if operator.recruited),
        },
        "graphConfig": {
            "encoder": EncoderConfig(
                hidden_dim=graph_config.hidden_dim, layers=graph_config.layers, seed=graph_config.seed
            ).to_dict(),
            "optimizer": {
                "shortlist": graph_config.shortlist,
                "beam": graph_config.beam,
                "coverageExtra": graph_config.coverage_extra,
                "mechanismKeep": graph_config.mechanism_keep,
                "fitSamples": graph_config.fit_samples,
                "candidateClosure": graph_config.candidate_closure,
                "preferScoreWithinTolerance": graph_config.prefer_score_within_tolerance,
            },
        },
        "cases": [],
    }

    graph = build_operator_graph(game_data, roster, allow_upgrades=False)
    report["graphAudit"] = graph.audit_report(limit=8)

    for case in case_list:
        layout = apply_right_side_preset(
            apply_layout_variant(parse_layout(case.layout), case.layout_variant),
            case.right_side,
        )
        case_payload: dict[str, Any] = {"case": case.to_dict(), "runs": {}}
        if include_hybrid:
            case_payload["runs"]["hybrid"] = _run_one(
                ScheduleOptimizer, game_data, roster, layout, case, repetitions
            )
        graph_optimizer = GraphScheduleOptimizer(
            game_data, roster, shard_formula="rock", graph_config=graph_config
        )
        case_payload["runs"]["graph"] = _run_one(
            GraphScheduleOptimizer,
            game_data,
            roster,
            layout,
            case,
            repetitions,
            instance=graph_optimizer,
        )
        case_payload["models"] = {
            key: value
            for key, value in (graph_optimizer.graph_stats.get("models") or {}).items()
            if value
        }
        case_payload["graphRuntime"] = graph_optimizer.graph_stats.get("graph", {})
        _attach_comparison(case_payload)
        report["cases"].append(case_payload)

    report["recall"] = measure_shortlist_recall(
        game_data, roster, cases=case_list, spec_limit=recall_sample_specs
    )
    report["summary"] = _summarize(report["cases"])
    return report


def _run_one(
    optimizer_class,
    game_data: GameData,
    roster: Sequence[RosterOperator],
    layout,
    case: BenchmarkCase,
    repetitions: int,
    *,
    instance=None,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for _ in range(repetitions):
        optimizer = instance or optimizer_class(game_data, roster, shard_formula="rock")
        started = time.perf_counter()
        try:
            result = optimizer.optimize(
                layout,
                mode=case.mode,
                shift_count=case.shift_count,
                shift_hours=case.shift_hours,
                drone_policy=case.drone_policy,
                profile_runtime=True,
            )
        except Exception as exc:  # 例如心情周期硬门槛不成立
            return {
                "algorithm": optimizer_class.__name__,
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
                "wallSeconds": round(time.perf_counter() - started, 3),
                "score": None,
            }
        elapsed = time.perf_counter() - started
        payload = _result_payload(result, elapsed, optimizer_class.__name__)
        if best is None or payload["wallSeconds"] < best["wallSeconds"]:
            best = payload
    assert best is not None
    return best


def _result_payload(result, elapsed: float, algorithm: str) -> dict[str, Any]:
    report = result.production_report
    daily = report.dailyExpected.to_dict() if report is not None else {}
    profile = result.runtime_profile or {}
    counters = profile.get("counters") or {}
    timings = profile.get("timingsSeconds") or {}
    conflicts = _conflicts(result.shifts)
    return {
        "algorithm": algorithm,
        "wallSeconds": round(elapsed, 3),
        "score": round(float(result.score), 3),
        "dailyExpected": {key: daily.get(key) for key in _DAILY_KEYS},
        "scoreBreakdown": {key: round(float(value), 4) for key, value in (report.scoreBreakdown or {}).items()}
        if report is not None
        else {},
        "shiftCount": len(result.shifts),
        "roomScores": [
            {
                "shift": shift.name,
                "roomId": room.room_id,
                "roomType": room.room_type,
                "target": room.target,
                "score": round(float(room.score), 3),
                "operators": [skill.operator_name for skill in room.operators],
            }
            for shift in result.shifts
            for room in shift.rooms
        ],
        "conflicts": conflicts,
        "warnings": len(result.warnings),
        "pureGoldPolicy": {
            key: (result.diagnostic_insertion_search.get("pureGoldBalancePolicy") or {}).get(key)
            for key in ("selectionRule", "status", "repeatCount", "withinTolerance", "finalPureGoldDelta")
        },
        "counters": {
            "evaluateRoom": counters.get("evaluateRoom.comboVector", 0),
            "roomCombosMiss": counters.get("roomCombos.cacheMiss", 0)
            + counters.get("graphRoomCombos.cacheMiss", 0),
            "insertionAttempts": counters.get("insertionAttempt.count", 0),
            "insertionAccepted": counters.get("insertionAttempt.status.improves", 0),
            "shortlistEvaluations": counters.get("graphShortlist.exactEvaluations", 0),
        },
        "insertionSeconds": round(float(timings.get("insertionAttempt.seconds", 0.0)), 3),
    }


def _conflicts(shifts) -> list[str]:
    conflicts: list[str] = []
    for shift in shifts:
        names: list[str] = []
        for room in [*shift.rooms, *shift.dormitories]:
            for skill in room.operators:
                names.append(skill.operator_name)
        seen: set[str] = set()
        for name in names:
            if name in seen:
                conflicts.append(f"{shift.name}:{name}")
            seen.add(name)
    active = [
        {skill.operator_name for room in shift.rooms for skill in room.operators}
        for shift in shifts
    ]
    for index in range(len(active)):
        overlap = active[index] & active[index - 1]
        if overlap:
            conflicts.append(f"boundary:{shifts[index].name}:{','.join(sorted(overlap))}")
    return conflicts


def _attach_comparison(case_payload: dict[str, Any]) -> None:
    runs = case_payload["runs"]
    graph = runs.get("graph")
    hybrid = runs.get("hybrid")
    if not graph or graph.get("failed"):
        if graph and graph.get("failed"):
            case_payload["comparison"] = {"failed": True, "error": graph.get("error")}
        return
    comparison: dict[str, Any] = {}
    if hybrid and not hybrid.get("failed"):
        comparison["scoreDelta"] = round(graph["score"] - hybrid["score"], 3)
        comparison["scoreRatio"] = (
            round(graph["score"] / hybrid["score"], 6) if hybrid["score"] else None
        )
        comparison["wallSecondsRatio"] = (
            round(graph["wallSeconds"] / hybrid["wallSeconds"], 4)
            if hybrid["wallSeconds"]
            else None
        )
        comparison["evaluateRoomRatio"] = (
            round(graph["counters"]["evaluateRoom"] / hybrid["counters"]["evaluateRoom"], 4)
            if hybrid["counters"]["evaluateRoom"]
            else None
        )
        graph_rooms = {(item["shift"], item["roomId"]): item for item in graph["roomScores"]}
        hybrid_rooms = {(item["shift"], item["roomId"]): item for item in hybrid["roomScores"]}
        shared = set(graph_rooms) & set(hybrid_rooms)
        better = sum(
            1
            for key in shared
            if graph_rooms[key]["score"] > hybrid_rooms[key]["score"] + 1e-6
        )
        worse = sum(
            1
            for key in shared
            if graph_rooms[key]["score"] < hybrid_rooms[key]["score"] - 1e-6
        )
        comparison["roomsStrictlyBetter"] = better
        comparison["roomsStrictlyWorse"] = worse
        comparison["roomsCompared"] = len(shared)
    elif hybrid and hybrid.get("failed"):
        comparison["hybridFailed"] = True
        comparison["hybridError"] = hybrid.get("error")
    case_payload["comparison"] = comparison


def _summarize(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ratios: list[float] = []
    deltas: list[float] = []
    wall_ratios: list[float] = []
    room_ratios: list[float] = []
    wins = 0
    losses = 0
    failed_cases: list[dict[str, Any]] = []
    for payload in cases:
        comparison = payload.get("comparison") or {}
        if comparison.get("failed") or comparison.get("hybridFailed"):
            failed_cases.append(
                {
                    "case": payload["case"]["label"],
                    "error": comparison.get("error") or comparison.get("hybridError"),
                }
            )
            continue
        if "scoreRatio" in comparison and comparison["scoreRatio"] is not None:
            ratios.append(float(comparison["scoreRatio"]))
            deltas.append(float(comparison["scoreDelta"]))
            if comparison["scoreDelta"] > 0.001:
                wins += 1
            elif comparison["scoreDelta"] < -0.001:
                losses += 1
        if comparison.get("wallSecondsRatio") is not None:
            wall_ratios.append(float(comparison["wallSecondsRatio"]))
        if comparison.get("evaluateRoomRatio") is not None:
            room_ratios.append(float(comparison["evaluateRoomRatio"]))
    return {
        "caseCount": len(cases),
        "evaluatedCaseCount": len(ratios),
        "failedCases": failed_cases,
        "wins": wins,
        "losses": losses,
        "ties": len(ratios) - wins - losses,
        "meanScoreRatio": round(sum(ratios) / len(ratios), 6) if ratios else None,
        "minScoreRatio": round(min(ratios), 6) if ratios else None,
        "meanScoreDelta": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "meanWallRatio": round(sum(wall_ratios) / len(wall_ratios), 4) if wall_ratios else None,
        "meanEvaluateRoomRatio": round(sum(room_ratios) / len(room_ratios), 4)
        if room_ratios
        else None,
        "graphConflicts": sum(len((payload["runs"].get("graph") or {}).get("conflicts") or []) for payload in cases),
        "hybridConflicts": sum(len((payload["runs"].get("hybrid") or {}).get("conflicts") or []) for payload in cases),
    }


def measure_shortlist_recall(
    game_data: GameData,
    roster: Sequence[RosterOperator],
    *,
    cases: Iterable[BenchmarkCase] | None = None,
    spec_limit: int = 6,
) -> dict[str, Any]:
    """短名单质量：图模型提名的候选能不能达到全量枚举的最优。

    这是整个替代是否成立的核心问题——旧算法穷举 ``C(32,3)`` 后取前 64，
    新算法只把短名单送进同一个模拟器。这里直接对每个房间规格比较：
    全量枚举的最优分、短名单的最优分、以及 top-16 组合的召回率。
    """
    case_list = list(cases or DEFAULT_BENCHMARK_CASES[:2])
    hybrid = ScheduleOptimizer(game_data, roster, shard_formula="rock")
    graph_optimizer = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
    rows: list[dict[str, Any]] = []
    seen_specs: set[tuple[str, str | None]] = set()
    for case in case_list:
        layout = apply_right_side_preset(
            apply_layout_variant(parse_layout(case.layout), case.layout_variant),
            case.right_side,
        )
        specs = hybrid._room_specs(layout, case.mode)
        for spec in specs:
            if spec.room_type not in {"TRADING", "MANUFACTURE"}:
                continue
            spec_key = (spec.room_type, spec.target)
            if spec_key in seen_specs:
                continue
            seen_specs.add(spec_key)
            if len(rows) >= spec_limit:
                break
            started = time.perf_counter()
            exhaustive = hybrid._room_combos(spec, set(), float(case.shift_hours))
            exhaustive_seconds = time.perf_counter() - started
            started = time.perf_counter()
            shortlist = graph_optimizer._room_combos(spec, set(), float(case.shift_hours))
            shortlist_seconds = time.perf_counter() - started
            if not exhaustive:
                continue
            exhaustive_best = exhaustive[0].score
            shortlist_best = shortlist[0].score if shortlist else float("-inf")
            exhaustive_top = {
                tuple(sorted(skill.operator_name for skill in combo.operators))
                for combo in exhaustive[:16]
            }
            shortlist_sets = {
                tuple(sorted(skill.operator_name for skill in combo.operators))
                for combo in shortlist
            }
            rows.append(
                {
                    "roomType": spec.room_type,
                    "target": spec.target,
                    "capacity": spec.capacity,
                    "exhaustiveCombos": len(exhaustive),
                    "exhaustiveBest": round(exhaustive_best, 3),
                    "shortlistCombos": len(shortlist),
                    "shortlistBest": round(shortlist_best, 3),
                    "top1Match": abs(shortlist_best - exhaustive_best) <= 1e-6,
                    "bestGap": round(exhaustive_best - shortlist_best, 3),
                    "top16Recall": round(len(exhaustive_top & shortlist_sets) / len(exhaustive_top), 4)
                    if exhaustive_top
                    else None,
                    "exhaustiveSeconds": round(exhaustive_seconds, 3),
                    "shortlistSeconds": round(shortlist_seconds, 3),
                    "exhaustiveRoomCombos": len(exhaustive),
                }
            )
        if len(rows) >= spec_limit:
            break
    best_matches = sum(1 for row in rows if row["top1Match"])
    recall_values = [row["top16Recall"] for row in rows if row["top16Recall"] is not None]
    return {
        "specs": rows,
        "specCount": len(rows),
        "top1MatchCount": best_matches,
        "meanTop16Recall": round(sum(recall_values) / len(recall_values), 4) if recall_values else None,
        "maxBestGap": round(max((row["bestGap"] for row in rows), default=0.0), 3),
    }


def _bond_coefficient_rows(report: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    """从各场景的模型描述里抽出一份「羁绊系数」示例表。"""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for payload in report.get("cases") or []:
        for key, model in (payload.get("models") or {}).items():
            bonds = (model or {}).get("bonds") or {}
            room_type, target = key.split(":")[0], key.split(":")[1]
            faction_rows = [
                item
                for item in (bonds.get("factionCounts") or [])
                if abs(float(item.get("coefficientPerOperator") or 0.0)) > 1e-6
            ][:limit]
            for item in faction_rows:
                identity = (room_type, target, f"faction_count:{item['tag']}")
                if identity in seen:
                    continue
                seen.add(identity)
                rows.append(
                    {
                        "room": room_type,
                        "target": target,
                        "feature": f"faction_count:{item['tag']}",
                        "coefficient": float(item["coefficientPerOperator"]),
                    }
                )
            for item in bonds.get("pairFeatures") or []:
                if abs(float(item.get("coefficient") or 0.0)) <= 1e-6:
                    continue
                identity = (room_type, target, str(item["feature"]))
                if identity in seen:
                    continue
                seen.add(identity)
                rows.append(
                    {
                        "room": room_type,
                        "target": target,
                        "feature": str(item["feature"]),
                        "coefficient": float(item["coefficient"]),
                    }
                )
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    """把基准报告渲染成人读的 Markdown。"""
    lines: list[str] = []
    add = lines.append
    summary = report.get("summary") or {}
    audit = report.get("graphAudit") or {}
    add("# 图卷积优化器 vs 混合搜索：对照基准")
    add("")
    add(f"- 生成时间：{report.get('generatedAt')}")
    add(f"- 练度表：{report.get('roster', {}).get('recruited')} 名已招募干员")
    add(f"- 数据版本：`{str(audit.get('dataVersion') or '').splitlines()[0]}`")
    add("")

    add("## 静态图体检")
    add("")
    add("| 项目 | 数值 |")
    add("| --- | --- |")
    add(f"| 干员节点 | {audit.get('nodeCount')} |")
    add(f"| 阵营组 | {audit.get('factionGroupCount')} |")
    edges = audit.get("edgeCountByRelation") or {}
    add(f"| 阵营边 | {edges.get('faction', 0)} |")
    add(f"| 点名羁绊边 | {edges.get('named', 0)} |")
    add(f"| 组合机制边 | {edges.get('combo', 0)} |")
    add(f"| 显式羁绊对数 | {audit.get('namedDependencyPairCount')} |")
    add(f"| 孤立干员 | {audit.get('isolatedOperatorCount')} |")
    add("")

    add("## 主结果")
    add("")
    add("| 场景 | 混合分 | 图卷积分 | 分数变化 | 混合耗时 | 图卷积耗时 | 模拟器调用比 |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for payload in report.get("cases") or []:
        runs = payload.get("runs") or {}
        hybrid = runs.get("hybrid")
        graph = runs.get("graph")
        if not graph:
            continue
        label = payload["case"]["label"]
        if graph.get("failed"):
            add(f"| {label} | - | 失败 | - | - | - | - |")
            continue
        if hybrid and hybrid.get("failed"):
            add(f"| {label} | 失败 | {graph['score']:.3f} | - | - | {graph['wallSeconds']:.2f}s | - |")
            continue
        if hybrid:
            comparison = payload.get("comparison") or {}
            add(
                f"| {label} | {hybrid['score']:.3f} | {graph['score']:.3f} | "
                f"{(comparison.get('scoreRatio') or 1) * 100 - 100:+.2f}% | "
                f"{hybrid['wallSeconds']:.2f}s | {graph['wallSeconds']:.2f}s | "
                f"{(comparison.get('evaluateRoomRatio') or 0):.3f}x |"
            )
        else:
            add(f"| {label} | - | {graph['score']:.3f} | - | - | {graph['wallSeconds']:.2f}s | - |")
    add("")
    add(
        f"汇总：{summary.get('wins')} 胜 / {summary.get('losses')} 负 / "
        f"{summary.get('ties')} 平，平均分数比 **{summary.get('meanScoreRatio')}**，"
        f"平均墙钟比 {summary.get('meanWallRatio')}，"
        f"平均模拟器调用比 {summary.get('meanEvaluateRoomRatio')}。"
    )
    add("")

    recall = report.get("recall") or {}
    if recall.get("specs"):
        add("## 短名单质量（能不能替代全量枚举）")
        add("")
        add("| 房间 | 目标 | 全量组合数 | 短名单组合数 | 全量最优 | 短名单最优 | 命中 | top16 召回 |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in recall["specs"]:
            add(
                f"| {row['roomType']} | {row['target'] or '-'} | {row['exhaustiveCombos']} | "
                f"{row['shortlistCombos']} | {row['exhaustiveBest']:.3f} | {row['shortlistBest']:.3f} | "
                f"{'是' if row['top1Match'] else '否'} | {row['top16Recall']} |"
            )
        add("")
        add(
            f"命中 {recall.get('top1MatchCount')}/{recall.get('specCount')} 个房间规格，"
            f"最大分数差 {recall.get('maxBestGap')}，平均 top16 召回 {recall.get('meanTop16Recall')}。"
        )
        add("")

    add("## 图价值模型拟合质量")
    add("")
    add("| 房间 | 目标 | 样本 | R²(holdout) | MAE(holdout) | MAE/均值 | Spearman | 近常数 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for payload in report.get("cases") or []:
        for key, model in (payload.get("models") or {}).items():
            parts = key.split(":")
            add(
                f"| {parts[0]} | {parts[1]} | {model.get('training', {}).get('samples')} | "
                f"{model.get('r2Holdout')} | {model.get('maeHoldout')} | "
                f"{(model.get('maeRatio') or 0) * 100:.2f}% | "
                f"{model.get('spearmanHoldout')} | "
                f"{'是' if model.get('degenerate') else '否'} |"
            )
        break
    add("")
    add(
        "注：R² 在标签方差极小（例如 guide 刊例把 L3 合成玉贸易站固定成 132%）的房间上会失去意义，"
        "此时看 MAE/均值 更实际。"
    )
    add("")

    bond_rows = _bond_coefficient_rows(report)
    if bond_rows:
        add("## 显式羁绊基学到的系数")
        add("")
        add("读出层里的羁绊项是显式线性基，系数可以直接读出来（单位：分 / 每单位特征）：")
        add("")
        add("| 房间 | 目标 | 特征 | 系数 |")
        add("| --- | --- | --- | --- |")
        for row in bond_rows:
            add(
                f"| {row['room']} | {row['target']} | {row['feature']} | "
                f"{row['coefficient']:+.3f} |"
            )
        add("")
        add(
            "`faction_count:*` 是「本房间该阵营每多一人值多少分」，"
            "`bond:*` 是「一对有羁绊的干员同房值多少分」。"
            "注意阵营列之间相关性很高，岭回归会把权重分摊到同族标签上，"
            "绝对值只有排名意义，不宜逐条当作游戏内公式。"
        )
        add("")

    add("## 每个房间的差异")
    add("")
    for payload in report.get("cases") or []:
        runs = payload.get("runs") or {}
        hybrid = runs.get("hybrid")
        graph = runs.get("graph")
        if not graph or graph.get("failed"):
            continue
        if not hybrid or hybrid.get("failed"):
            continue
        add(f"### {payload['case']['label']}")
        add("")
        add("| 班次 | 房间 | 目标 | 混合 | 图卷积 | 差值 | 图卷积人选 |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        hybrid_rooms = {(item["shift"], item["roomId"]): item for item in hybrid["roomScores"]}
        for item in graph["roomScores"]:
            # 纯赤金平衡可能把周期重复多次（A#1、A#2...），只展示第一轮，避免表格刷屏。
            if item["shift"].endswith("#1") is False and "#" in item["shift"]:
                continue
            key = (item["shift"], item["roomId"])
            previous = hybrid_rooms.get(key)
            if previous is None:
                continue
            delta = item["score"] - previous["score"]
            add(
                f"| {item['shift']} | {item['roomId']} | {item['target'] or '-'} | "
                f"{previous['score']:.3f} | {item['score']:.3f} | {delta:+.3f} | "
                f"{', '.join(item['operators'])} |"
            )
        add("")
    return "\n".join(lines)


def write_benchmark_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "graph_vs_hybrid_benchmark.json"
    md_path = output_dir / "graph_vs_hybrid_benchmark.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
