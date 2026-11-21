"""M3b 探针：共享「房间 × 干员」亲和度矩阵 vs 每间房各拟合一套，读数质量对比。

对同一批房间规格、同一个标签口径（模拟器的精确房间分）、同一份样本预算，比较：

* ``per_room``：现状（``fit_room_value_model``，每个规格一套 47 维一元/二元权重）；
* ``affinity``：M3b（一条共享读出层，输入是房间节点嵌入 × 干员嵌入的双线性形式）。

要看的是**留出 R² / MAE**，不是训练误差：M3b 的假设是「房间条件化 + 池化」能在
样本少的规格上借到强度，代价是亲和度矩阵被强制落在房间基的张成空间里。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.graph_conv import (
    AffinityFitRoom,
    GraphEncoder,
    RoomCondition,
    RoomEncoder,
    RoomEncoderConfig,
    RoomQuery,
    fit_room_affinity_model,
    fit_room_value_model,
)
from arknights_schedule_generator.graph_features import build_operator_graph, build_room_graph
from arknights_schedule_generator.graph_optimizer import GraphScheduleOptimizer
from arknights_schedule_generator.optimizer import RoomSpec
from arknights_schedule_generator.roster import load_roster_xlsx

DATA = Path("data/cache")
ROSTER = Path("examples/fixtures/yituliu_full_roster_maxed.xlsx")

ROOMS = (
    ("TRADING", "O_GOLD"),
    ("TRADING", "O_DIAMOND"),
    ("MANUFACTURE", "F_GOLD"),
    ("MANUFACTURE", "F_EXP"),
    ("MANUFACTURE", "F_DIAMOND"),
    ("POWER", None),
)

LEVELS = (1, 2, 3)
DURATIONS = (8.0, 12.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128, help="每个规格的采样数")
    parser.add_argument("--ridge", type=float, default=4.0)
    parser.add_argument("--per-room-samples", type=int, default=288)
    args = parser.parse_args()

    game_data = GameData.load(DATA)
    roster = load_roster_xlsx(ROSTER)
    optimizer = GraphScheduleOptimizer(game_data, roster, shard_formula="rock")
    graph = build_operator_graph(game_data, roster)
    encoder = GraphEncoder(graph, optimizer.graph_config.encoder_config())
    embeddings = {
        node.name: vector for node, vector in zip(graph.nodes, encoder.embeddings())
    }
    room_encoder = RoomEncoder(build_room_graph(game_data), RoomEncoderConfig())
    print(
        f"room_dim={room_encoder.room_dim} embedding_dim={encoder.embedding_dim} "
        f"room_nodes={room_encoder.room_graph.size}"
    )

    fit_rooms: list[AffinityFitRoom] = []
    specs: list[tuple[RoomSpec, RoomCondition]] = []
    for room_type, target in ROOMS:
        for level in LEVELS:
            slots = optimizer._room_capacity(room_type, level)
            for duration in DURATIONS:
                spec = RoomSpec(
                    f"{room_type.lower()}_probe",
                    room_type,
                    target,
                    slots,
                    level,
                    slots,
                )
                variants = optimizer._best_variant_map(spec, duration)
                if not variants:
                    continue
                condition = RoomCondition(
                    room_type=room_type,
                    target=target,
                    room_level=level,
                    slots=slots,
                    duration_hours=duration,
                )
                bond_basis = optimizer._bond_basis(spec, duration, variants)

                def oracle(combo, _spec=spec, _variants=variants, _duration=duration):
                    operators = [_variants[name] for name in combo]
                    return optimizer._score_room_combo(_spec, operators, _duration)

                fit_rooms.append(
                    AffinityFitRoom(
                        condition=condition,
                        candidates=tuple(sorted(variants)),
                        capacity=slots,
                        oracle=oracle,
                        bond_basis=bond_basis,
                    )
                )
                specs.append((spec, condition))
    print(f"fit rooms={len(fit_rooms)}")

    started = time.perf_counter()
    affinity = fit_room_affinity_model(
        room_encoder,
        embeddings,
        fit_rooms,
        samples=args.samples,
        seed=11,
        ridge=args.ridge,
    )
    affinity_seconds = time.perf_counter() - started
    if affinity is None:
        print("affinity model: None")
        return 1
    print(
        f"affinity fit: {affinity_seconds:.1f}s parameters={affinity.parameter_count} "
        f"r2Train={affinity.fit_quality['r2Train']} r2Holdout={affinity.fit_quality['r2Holdout']} "
        f"spearman={affinity.fit_quality['spearmanHoldout']} mae={affinity.fit_quality['maeHoldout']}"
    )

    print()
    print("per-room baseline (fit_room_value_model, 同口径标签):")
    started = time.perf_counter()
    baseline: dict[str, dict] = {}
    for spec, condition in specs:
        variants = optimizer._best_variant_map(spec, 12.0 if condition.duration_hours == 12.0 else 8.0)
        if not variants:
            continue
        duration = condition.duration_hours

        def oracle(combo, _spec=spec, _variants=variants, _duration=duration):
            operators = [_variants[name] for name in combo]
            return optimizer._score_room_combo(_spec, operators, _duration)

        query = RoomQuery(
            room_type=condition.room_type,
            target=condition.target,
            room_level=condition.room_level,
            slots=condition.slots,
        )
        model = fit_room_value_model(
            encoder,
            query,
            list(variants),
            oracle,
            capacity=spec.capacity,
            samples=args.per_room_samples,
            seed=11,
            ridge=1.0,
            bond_basis=optimizer._bond_basis(spec, duration, variants),
        )
        if model is None:
            continue
        key = f"{condition.room_type}|{condition.target}|L{condition.room_level}|{int(duration)}h"
        baseline[key] = model.fit_quality
    baseline_seconds = time.perf_counter() - started
    print(f"per-room baseline fit: {baseline_seconds:.1f}s for {len(baseline)} specs")

    print()
    header = (
        f"{'room':34s} {'perRoom R2':>11s} {'affinity R2':>12s} "
        f"{'perRoom MAE':>12s} {'affinity MAE':>13s}"
    )
    print(header)
    per_room = affinity.fit_quality["perRoom"]
    for key in sorted(baseline):
        left = baseline[key]
        right = per_room.get(key, {"r2": float("nan"), "mae": float("nan")})
        print(
            f"{key:34s} {left['r2Holdout']:11.3f} {right['r2']:12.3f} "
            f"{left['maeHoldout']:12.3f} {right['mae']:13.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
