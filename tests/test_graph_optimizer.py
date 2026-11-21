"""图卷积优化器的回归测试。

覆盖四件事：

1. 静态图构建（阵营/羁绊/去重/确定性）；
2. 固定超参图卷积编码器（形状、确定性、数值稳定）；
3. 闭式读出层（岭回归求解正确性、拟合质量、排序能力）；
4. 优化器契约（班次/冲突/审计字典/版本号）与短名单质量。
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from arknights_schedule_generator.data import GameData
from arknights_schedule_generator.exporter import find_conflicts
from arknights_schedule_generator.graph_bonds import (
    BondBasis,
    build_bond_basis,
    in_room_count_tags,
)
from arknights_schedule_generator.graph_cache import CACHE_SCHEMA_VERSION, GraphModelCache
from arknights_schedule_generator.graph_conv import (
    AffinityFitRoom,
    EncoderConfig,
    GraphEncoder,
    RoomAffinityModel,
    RoomCondition,
    RoomEncoder,
    RoomEncoderConfig,
    RoomQuery,
    _solve_ridge,
    _spearman,
    fit_room_affinity_model,
    fit_room_value_model,
    numpy_available,
    rank_combinations,
)
from arknights_schedule_generator.graph_features import (
    FEATURE_NAMES,
    ROOM_CATALOG,
    ROOM_FEATURE_NAMES,
    build_operator_graph,
    build_room_graph,
)
from arknights_schedule_generator.graph_optimizer import (
    GRAPH_OPTIMIZER_MODEL_VERSION,
    GraphOptimizerConfig,
    GraphScheduleOptimizer,
)
from arknights_schedule_generator.models import RosterOperator
from arknights_schedule_generator.optimizer import RoomSpec, ScheduleOptimizer, parse_layout
from arknights_schedule_generator.roster import load_roster_xlsx

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = REPO_ROOT / "data" / "cache"
FIXTURE_ROSTER = REPO_ROOT / "examples" / "fixtures" / "yituliu_full_roster_maxed.xlsx"

TRADING_OPERATORS = [
    ("能天使", "char_exu", "trade_a", 35),
    ("德克萨斯", "char_texas", "trade_b", 30),
    ("拉普兰德", "char_lap", "trade_c", 28),
    ("空", "char_sora", "trade_d", 25),
    ("巫恋", "char_shamare", "trade_e", 22),
    ("龙舌兰", "char_tequila", "trade_f", 20),
    ("可露希尔", "char_croissant", "trade_g", 18),
    ("但书", "char_butushu", "trade_h", 12),
    ("摩根", "char_morgan", "trade_i", 26),
    ("维娜·维多利亚", "char_vina", "trade_j", 24),
    ("芬", "char_fang", "trade_k", 16),
    ("夜刀", "char_yato", "trade_l", 14),
]

MANUFACTURE_OPERATORS = [
    ("砾", "char_gravel", "manu_a", 30),
    ("斑点", "char_spot", "manu_b", 28),
    ("夜烟", "char_haze", "manu_c", 26),
    ("梅尔", "char_mayer", "manu_d", 24),
    ("温米", "char_wind", "manu_e", 22),
    ("苍苔", "char_moss", "manu_f", 20),
]


def write_graph_test_data(data_dir: Path) -> None:
    """一份带阵营标签和点名羁绊的最小数据集。"""
    rooms = {
        "CONTROL": {"phases": [{"maxStationedNum": 5}]},
        "TRADING": {"phases": [{"maxStationedNum": 3}]},
        "MANUFACTURE": {"phases": [{"maxStationedNum": 3}]},
        "POWER": {"phases": [{"maxStationedNum": 1}]},
        "DORMITORY": {"phases": [{"maxStationedNum": 5}]},
        "MEETING": {"phases": [{"maxStationedNum": 2}]},
        "HIRE": {"phases": [{"maxStationedNum": 1}]},
    }
    buffs: dict[str, dict[str, object]] = {}
    chars: dict[str, dict[str, object]] = {}
    characters: dict[str, dict[str, object]] = {}
    factions = {
        "char_texas": "siracusa",
        "char_lap": "siracusa",
        "char_vina": "victoria",
        "char_morgan": "glasgow",
        "char_exu": "laterano",
    }
    for name, char_id, buff_id, efficiency in TRADING_OPERATORS:
        description = f"进驻贸易站时，订单获取效率+{efficiency}%"
        if name == "拉普兰德":
            description = (
                f"进驻贸易站时，订单获取效率+{efficiency}%，"
                "当与德克萨斯在同一个贸易站时，订单获取效率额外+10%"
            )
        buffs[buff_id] = {
            "buffId": buff_id,
            "buffName": f"贸易技能{name}",
            "roomType": "TRADING",
            "description": description,
            "efficiency": efficiency,
            "targets": [],
        }
        chars[char_id] = {
            "charId": char_id,
            "buffChar": [
                {"buffData": [{"buffId": buff_id, "cond": {"phase": "PHASE_0", "level": 1}}]}
            ],
        }
        characters[char_id] = {
            "name": name,
            "rarity": "TIER_6",
            "nationId": factions.get(char_id),
            "phases": [{"maxLevel": 50, "evolveCost": None}],
        }
    for name, char_id, buff_id, efficiency in MANUFACTURE_OPERATORS:
        buffs[buff_id] = {
            "buffId": buff_id,
            "buffName": f"制造技能{name}",
            "roomType": "MANUFACTURE",
            "description": f"进驻制造站时，生产力+{efficiency}%",
            "efficiency": efficiency,
            "targets": ["F_GOLD"],
        }
        chars[char_id] = {
            "charId": char_id,
            "buffChar": [
                {"buffData": [{"buffId": buff_id, "cond": {"phase": "PHASE_0", "level": 1}}]}
            ],
        }
        characters[char_id] = {
            "name": name,
            "rarity": "TIER_5",
            "phases": [{"maxLevel": 50, "evolveCost": None}],
        }
    buffs["power_buff"] = {
        "buffId": "power_buff",
        "buffName": "备用能源",
        "roomType": "POWER",
        "description": "进驻发电站时，无人机充能速度+10%",
        "efficiency": 10,
        "targets": [],
    }
    chars["char_power"] = {
        "charId": "char_power",
        "buffChar": [
            {"buffData": [{"buffId": "power_buff", "cond": {"phase": "PHASE_0", "level": 1}}]}
        ],
    }
    characters["char_power"] = {
        "name": "雷蛇",
        "rarity": "TIER_5",
        "phases": [{"maxLevel": 50, "evolveCost": None}],
    }
    (data_dir / "building_data.json").write_text(
        json.dumps({"rooms": rooms, "buffs": buffs, "chars": chars}), encoding="utf-8"
    )
    (data_dir / "character_table.json").write_text(
        json.dumps(characters), encoding="utf-8"
    )
    (data_dir / "item_table.json").write_text(json.dumps({}), encoding="utf-8")
    (data_dir / "data_version.txt").write_text("test", encoding="utf-8")


def write_room_graph_test_data(data_dir: Path) -> None:
    """在最小数据集上补上房间图需要的两样东西：每级耗电、中枢减耗。"""
    write_graph_test_data(data_dir)
    path = data_dir / "building_data.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    electricity = {
        "CONTROL": 0,
        "TRADING": -10,
        "MANUFACTURE": -10,
        "POWER": 60,
        "DORMITORY": -10,
        "MEETING": -10,
        "HIRE": -20,
    }
    for room_type, record in payload["rooms"].items():
        for phase in record.get("phases") or []:
            phase["electricity"] = electricity.get(room_type, 0)
    payload["controlData"] = {"basicCostBuff": -5}
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_test_roster() -> list[RosterOperator]:
    operators: list[RosterOperator] = []
    for name, _char_id, _buff, _eff in [*TRADING_OPERATORS, *MANUFACTURE_OPERATORS]:
        operators.append(
            RosterOperator(
                name=name,
                recruited=True,
                rarity=6,
                level=90,
                elite=2,
                potential=6,
                skill_level=7,
                masteries=(3, 3, 3),
            )
        )
    operators.append(
        RosterOperator(name="雷蛇", recruited=True, rarity=5, level=80, elite=2)
    )
    return operators


class GraphFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.data_dir = Path(self._temp.name)
        write_graph_test_data(self.data_dir)
        self.game_data = GameData.load(self.data_dir)
        self.roster = build_test_roster()

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_graph_has_expected_nodes_and_typed_edges(self) -> None:
        graph = build_operator_graph(self.game_data, self.roster)
        self.assertEqual(graph.size, len(self.roster))
        report = graph.audit_report()
        self.assertEqual(report["featureWidth"], len(FEATURE_NAMES))
        self.assertGreater(report["edgeCountByRelation"]["faction"], 0)
        self.assertEqual(report["edgeCountByRelation"]["named"], 2)
        self.assertIn("德克萨斯 <-> 拉普兰德", report["namedDependencyPairs"])
        self.assertIn("siracusa", graph.faction_groups)

    def test_named_bond_neighbours_are_symmetric(self) -> None:
        graph = build_operator_graph(self.game_data, self.roster)
        forward = dict(graph.neighbors("拉普兰德", "named"))
        backward = dict(graph.neighbors("德克萨斯", "named"))
        self.assertIn("德克萨斯", forward)
        self.assertIn("拉普兰德", backward)
        self.assertAlmostEqual(forward["德克萨斯"], backward["拉普兰德"])

    def test_duplicate_roster_rows_are_merged(self) -> None:
        duplicated = [*self.roster, self.roster[0]]
        graph = build_operator_graph(self.game_data, duplicated)
        self.assertEqual(graph.size, len(self.roster))
        self.assertEqual(graph.meta["duplicateRosterNames"], {self.roster[0].name: 2})

    def test_graph_build_is_deterministic(self) -> None:
        first = build_operator_graph(self.game_data, self.roster)
        second = build_operator_graph(self.game_data, self.roster)
        self.assertEqual(first.meta["signature"], second.meta["signature"])
        self.assertEqual(
            [node.features for node in first.nodes],
            [node.features for node in second.nodes],
        )


class GraphEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.data_dir = Path(self._temp.name)
        write_graph_test_data(self.data_dir)
        self.game_data = GameData.load(self.data_dir)
        self.graph = build_operator_graph(self.game_data, build_test_roster())

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_embedding_shape(self) -> None:
        config = EncoderConfig(hidden_dim=8, layers=2)
        encoder = GraphEncoder(self.graph, config)
        embeddings = encoder.embeddings()
        self.assertEqual(len(embeddings), self.graph.size)
        expected = len(FEATURE_NAMES) + len(
            [name for name in FEATURE_NAMES if name.startswith("room:")]
        ) + 8 * 2
        self.assertEqual(encoder.embedding_dim, expected)
        self.assertTrue(all(len(vector) == encoder.embedding_dim for vector in embeddings))
        self.assertTrue(all(math.isfinite(value) for vector in embeddings for value in vector))

    def test_encoder_is_deterministic_across_instances(self) -> None:
        config = EncoderConfig(hidden_dim=6, layers=2, seed=1234)
        first = GraphEncoder(self.graph, config).embeddings()
        second = GraphEncoder(self.graph, config).embeddings()
        self.assertEqual(first, second)

    def test_seed_changes_embeddings(self) -> None:
        left = GraphEncoder(self.graph, EncoderConfig(hidden_dim=6, seed=1)).embeddings()
        right = GraphEncoder(self.graph, EncoderConfig(hidden_dim=6, seed=2)).embeddings()
        self.assertNotEqual(left, right)


class RidgeReadoutTest(unittest.TestCase):
    def test_solver_recovers_known_linear_model(self) -> None:
        weights = [1.5, -2.0, 0.5, 3.0]
        rows = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 1.0, 0.5, -1.0],
            [1.0, 2.0, 3.0, 0.0],
        ]
        targets = [sum(w * v for w, v in zip(weights, row)) for row in rows]
        solved = _solve_ridge(rows, targets, ridge=1e-9)
        for expected, actual in zip(weights, solved):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_spearman_is_one_for_monotone_prediction(self) -> None:
        actual = [3.0, 1.0, 2.0, 5.0, 4.0]
        predicted = [value * 2.0 + 7.0 for value in actual]
        self.assertAlmostEqual(_spearman(predicted, actual), 1.0, places=6)
        self.assertAlmostEqual(_spearman([-v for v in actual], actual), -1.0, places=6)


class RoomValueModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.data_dir = Path(self._temp.name)
        write_graph_test_data(self.data_dir)
        self.game_data = GameData.load(self.data_dir)
        self.graph = build_operator_graph(self.game_data, build_test_roster())
        self.encoder = GraphEncoder(self.graph, EncoderConfig(hidden_dim=10, layers=2))

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_model_recovers_an_additive_pairwise_oracle(self) -> None:
        names = [name for name, *_ in TRADING_OPERATORS]
        embeddings = {
            node.name: vector
            for node, vector in zip(self.graph.nodes, self.encoder.embeddings())
        }

        def oracle(combo: tuple[str, ...]) -> float:
            total = 0.0
            for index, name in enumerate(combo):
                total += sum(embeddings[name][:3]) * 2.0
                for other in combo[index + 1 :]:
                    total += embeddings[name][0] * embeddings[other][1] * 5.0
            return total

        model = fit_room_value_model(
            self.encoder,
            RoomQuery("TRADING", "O_GOLD", 3, 3),
            names,
            oracle,
            capacity=3,
            samples=220,
            seed=11,
            ridge=1e-4,
        )
        self.assertIsNotNone(model)
        assert model is not None
        self.assertFalse(model.fit_quality["degenerate"])
        self.assertGreater(model.fit_quality["r2Holdout"], 0.9)
        ranked = rank_combinations(
            model, embeddings, names, capacity=3, beam=8, top_k=5
        )
        self.assertEqual(len(ranked), 5)
        best_combo = ranked[0][1]
        best_actual = max(
            oracle(combo)
            for combo in _combinations_of(names, 3)
        )
        self.assertGreaterEqual(oracle(best_combo), best_actual - 1e-6)

    def test_constant_oracle_yields_degenerate_model(self) -> None:
        names = [name for name, *_ in MANUFACTURE_OPERATORS]
        model = fit_room_value_model(
            self.encoder,
            RoomQuery("MANUFACTURE", "F_GOLD", 3, 3),
            names,
            lambda combo: 42.0,
            capacity=3,
            samples=60,
            seed=3,
        )
        self.assertIsNotNone(model)
        assert model is not None
        self.assertTrue(model.fit_quality["degenerate"])
        self.assertAlmostEqual(model.score(names[:3], {
            node.name: vector
            for node, vector in zip(self.graph.nodes, self.encoder.embeddings())
        }), 0.0)


def _combinations_of(names, size):
    from itertools import combinations

    return combinations(names, size)


class GraphOptimizerContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.data_dir = Path(self._temp.name)
        write_graph_test_data(self.data_dir)
        self.game_data = GameData.load(self.data_dir)
        self.roster = build_test_roster()
        self.layout = parse_layout("243")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_model_version_differs_from_hybrid(self) -> None:
        self.assertNotEqual(
            GraphScheduleOptimizer.optimizer_model_version(),
            ScheduleOptimizer.optimizer_model_version(),
        )
        self.assertEqual(
            GraphScheduleOptimizer.optimizer_model_version(), GRAPH_OPTIMIZER_MODEL_VERSION
        )

    def test_optimize_returns_conflict_free_result_with_audit(self) -> None:
        optimizer = GraphScheduleOptimizer(self.game_data, self.roster, shard_formula="rock")
        result = optimizer.optimize(
            self.layout,
            mode="normal",
            shift_count=2,
            shift_hours=12,
            drone_policy="none",
        )
        self.assertEqual(len(result.shifts), 2)
        self.assertEqual([shift.duration_hours for shift in result.shifts], [12.0, 12.0])
        self.assertEqual([shift.name for shift in result.shifts], ["A", "B"])
        self.assertEqual(find_conflicts(result), [])
        self.assertIsNotNone(result.production_report)
        audit = result.diagnostic_insertion_search
        self.assertIsInstance(audit, dict)
        self.assertIn("summary", audit)
        self.assertIn("localQualityAudit", audit)
        self.assertIn("candidatePoolAudit", audit)
        validation = audit["cacheValidation"]
        self.assertEqual(validation["optimizerModelVersion"], GRAPH_OPTIMIZER_MODEL_VERSION)
        self.assertEqual(validation["maxGroups"], optimizer.diagnostic_insertion_group_limit())
        self.assertEqual(audit["summary"]["remainingImprovementCount"], 0)

    def test_room_combos_shortlist_reaches_exhaustive_best(self) -> None:
        hybrid = ScheduleOptimizer(self.game_data, self.roster, shard_formula="rock")
        graph = GraphScheduleOptimizer(
            self.game_data,
            self.roster,
            shard_formula="rock",
            graph_config=GraphOptimizerConfig(shortlist=24, beam=10),
        )
        spec = next(
            room
            for room in hybrid._room_specs(self.layout, "normal")
            if room.room_type == "TRADING"
        )
        exhaustive = hybrid._room_combos(spec, set(), 12.0)
        shortlist = graph._room_combos(spec, set(), 12.0)
        self.assertTrue(exhaustive)
        self.assertTrue(shortlist)
        self.assertGreaterEqual(shortlist[0].score, exhaustive[0].score - 1e-6)
        self.assertLessEqual(len(shortlist), len(exhaustive))

    def test_shortlist_is_much_smaller_than_exhaustive_enumeration(self) -> None:
        hybrid = ScheduleOptimizer(self.game_data, self.roster, shard_formula="rock")
        graph = GraphScheduleOptimizer(self.game_data, self.roster, shard_formula="rock")
        spec = next(
            room
            for room in hybrid._room_specs(self.layout, "normal")
            if room.room_type == "TRADING"
        )
        exhaustive = hybrid._room_combos(spec, set(), 12.0)
        shortlist = graph._room_combos(spec, set(), 12.0)
        self.assertGreater(len(exhaustive), 100)
        self.assertLess(len(shortlist), len(exhaustive) / 2)

    def test_multi_start_initial_assignment_is_at_least_as_good_as_each_start(self) -> None:
        """多起手必须不劣于任何单起手（它就是在同一个目标函数里取最大值）。"""
        layout = self.layout
        kwargs = dict(mode="normal", shift_count=2, shift_hours=12, drone_policy="none")
        single_scores = []
        for budget in (12, 0):
            optimizer = GraphScheduleOptimizer(
                self.game_data,
                self.roster,
                shard_formula="rock",
                graph_config=GraphOptimizerConfig(
                    use_cache=False, initial_starts=(budget,)
                ),
            )
            single_scores.append(
                optimizer.optimize(layout, **kwargs).score
            )
        multi = GraphScheduleOptimizer(
            self.game_data,
            self.roster,
            shard_formula="rock",
            graph_config=GraphOptimizerConfig(use_cache=False, initial_starts=(12, 0)),
        )
        multi_score = multi.optimize(layout, **kwargs).score
        self.assertGreaterEqual(multi_score + 1e-6, max(single_scores))
        self.assertEqual(multi.graph_stats["starts"]["budgets"], [12, 0])

    def test_disabling_candidate_closure_matches_parent_pool(self) -> None:
        hybrid = ScheduleOptimizer(self.game_data, self.roster, shard_formula="rock")
        graph = GraphScheduleOptimizer(
            self.game_data,
            self.roster,
            shard_formula="rock",
            graph_config=GraphOptimizerConfig(candidate_closure=False),
        )
        self.assertEqual(
            hybrid._combo_candidate_pool("TRADING", "O_GOLD", set()),
            graph._combo_candidate_pool("TRADING", "O_GOLD", set()),
        )


class RoomNodeGraphTest(unittest.TestCase):
    """M3b：房间节点 + 发电站/控制中枢的结构边。"""

    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.data_dir = Path(self._temp.name)
        write_room_graph_test_data(self.data_dir)
        self.game_data = GameData.load(self.data_dir)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_room_graph_covers_the_catalog_with_structural_edges(self) -> None:
        graph = build_room_graph(self.game_data)
        self.assertEqual(graph.size, len(ROOM_CATALOG))
        report = graph.audit_report()
        self.assertEqual(report["featureWidth"], len(ROOM_FEATURE_NAMES))
        self.assertEqual(set(report["edgeCountByRelation"]), {"power", "control"})
        self.assertGreater(report["edgeCountByRelation"]["power"], 0)
        self.assertGreater(report["edgeCountByRelation"]["control"], 0)
        # 发电站供电 → 每间耗电房都有一条 power 入边；控制中枢 → 每个房间一条 control 入边。
        trading = graph.node("TRADING", "O_GOLD")
        self.assertIsNotNone(trading)
        assert trading is not None
        relations = {relation for _node, relation, _weight in graph.neighbours("TRADING", "O_GOLD")}
        self.assertEqual(relations, {"power", "control"})

    def test_room_features_come_from_building_data(self) -> None:
        graph = build_room_graph(self.game_data)
        index = {name: position for position, name in enumerate(ROOM_FEATURE_NAMES)}
        power = graph.node("POWER", None)
        trading = graph.node("TRADING", "O_GOLD")
        assert power is not None and trading is not None
        # 发电站供电、耗电房耗电：符号必须相反（否则「谁给谁供电」就不是结构信息了）。
        self.assertGreater(power.features[index["powerPerSlot"]], 0)
        self.assertLess(trading.features[index["powerPerSlot"]], 0)
        self.assertEqual(trading.features[index["capacity"]], 3.0 / 5.0)
        self.assertEqual(trading.features[index["isResourceRoom"]], 1.0)
        self.assertEqual(power.features[index["isPowerRoom"]], 1.0)

    def test_room_graph_is_deterministic(self) -> None:
        first = build_room_graph(self.game_data)
        second = build_room_graph(self.game_data)
        self.assertEqual(first.meta["signature"], second.meta["signature"])
        self.assertEqual(
            [node.features for node in first.nodes],
            [node.features for node in second.nodes],
        )


class RoomAffinityReadoutTest(unittest.TestCase):
    """M3b：一条共享读出层同时给多间房打分（房间节点 × 干员嵌入的双线性读出）。"""

    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.data_dir = Path(self._temp.name)
        write_graph_test_data(self.data_dir)
        self.game_data = GameData.load(self.data_dir)
        self.roster = build_test_roster()
        self.graph = build_operator_graph(self.game_data, self.roster)
        self.encoder = GraphEncoder(self.graph, EncoderConfig(hidden_dim=10, layers=2))
        self.embeddings = {
            node.name: vector for node, vector in zip(self.graph.nodes, self.encoder.embeddings())
        }
        self.room_encoder = RoomEncoder(build_room_graph(self.game_data), RoomEncoderConfig())
        self.trading_names = [name for name, *_ in TRADING_OPERATORS]
        self.manufacture_names = [name for name, *_ in MANUFACTURE_OPERATORS]

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _two_room_oracle_model(self) -> RoomAffinityModel | None:
        """TRADING 想要 embedding[0] 大的干员，MANUFACTURE 想要 embedding[1] 大的。"""

        def trading_oracle(combo: tuple[str, ...]) -> float:
            return 50.0 * sum(self.embeddings[name][0] for name in combo)

        def manufacture_oracle(combo: tuple[str, ...]) -> float:
            return 50.0 * sum(self.embeddings[name][1] for name in combo)

        rooms = [
            AffinityFitRoom(
                condition=RoomCondition("TRADING", "O_GOLD", level, 1, 12.0),
                candidates=tuple(sorted(self.trading_names)),
                capacity=1,
                oracle=trading_oracle,
            )
            for level in (1, 2)
        ] + [
            AffinityFitRoom(
                condition=RoomCondition("MANUFACTURE", "F_GOLD", level, 1, 12.0),
                candidates=tuple(sorted(self.manufacture_names)),
                capacity=1,
                oracle=manufacture_oracle,
            )
            for level in (1, 2)
        ]
        return fit_room_affinity_model(
            self.room_encoder,
            self.embeddings,
            rooms,
            samples=48,
            seed=5,
            ridge=0.5,
        )

    def test_room_embedding_depends_on_room_and_query(self) -> None:
        base = RoomCondition("TRADING", "O_GOLD", 3, 3, 12.0)
        higher_level = RoomCondition("TRADING", "O_GOLD", 1, 3, 12.0)
        other_room = RoomCondition("MANUFACTURE", "F_GOLD", 3, 3, 12.0)
        other_duration = RoomCondition("TRADING", "O_GOLD", 3, 3, 8.0)
        vector = self.room_encoder.embedding(base)
        self.assertEqual(len(vector), self.room_encoder.room_dim)
        self.assertEqual(vector, self.room_encoder.embedding(base))
        self.assertNotEqual(vector, self.room_encoder.embedding(higher_level))
        self.assertNotEqual(vector, self.room_encoder.embedding(other_room))
        self.assertNotEqual(vector, self.room_encoder.embedding(other_duration))
        self.assertTrue(all(math.isfinite(value) for value in vector))
        # 结构边必须真的参与：TRADING 的房间嵌入要能被「发电站/中枢」这一项区分出来。
        self.assertGreater(abs(sum(vector[: len(ROOM_FEATURE_NAMES)])), 0.0)

    def test_shared_model_prefers_a_different_operator_per_room(self) -> None:
        model = self._two_room_oracle_model()
        self.assertIsNotNone(model)
        assert model is not None
        trading_scores = model.affinity_scores(
            RoomCondition("TRADING", "O_GOLD", 1, 1, 12.0), self.trading_names, self.embeddings
        )
        manufacture_scores = model.affinity_scores(
            RoomCondition("MANUFACTURE", "F_GOLD", 1, 1, 12.0),
            self.manufacture_names,
            self.embeddings,
        )
        trading_best = max(trading_scores, key=lambda name: trading_scores[name])
        manufacture_best = max(manufacture_scores, key=lambda name: manufacture_scores[name])
        expected_trading = max(
            self.trading_names, key=lambda name: self.embeddings[name][0]
        )
        expected_manufacture = max(
            self.manufacture_names, key=lambda name: self.embeddings[name][1]
        )
        # 同一张表、同一批干员嵌入，只因为房间不同就给出不同的第一名。
        self.assertEqual(trading_best, expected_trading)
        self.assertEqual(manufacture_best, expected_manufacture)

    def test_affinity_view_matches_the_materialised_linear_form(self) -> None:
        model = self._two_room_oracle_model()
        self.assertIsNotNone(model)
        assert model is not None
        condition = RoomCondition("TRADING", "O_GOLD", 3, 3, 12.0)
        view = model.view(condition)
        self.assertEqual(view.query.room_type, "TRADING")
        self.assertEqual(len(view.unary), model.embedding_dim + view.bond_unary_width)
        name = self.trading_names[0]
        vector = self.embeddings[name]
        direct = view.bias + sum(
            weight * value for weight, value in zip(view.unary[: model.embedding_dim], vector)
        )
        self.assertAlmostEqual(view.score([name], self.embeddings), direct, places=9)
        # 亲和度（一元项）与物化视图的一元项必须一致 —— 否则「同一张表」不成立。
        affinity = model.affinity_scores(condition, [name], self.embeddings)[name]
        self.assertAlmostEqual(direct - view.bias, affinity, places=9)
        ranked = rank_combinations(
            view, self.embeddings, self.trading_names, capacity=3, beam=6, top_k=3
        )
        self.assertEqual(len(ranked), 3)

    def test_affinity_model_round_trips_through_dict(self) -> None:
        model = self._two_room_oracle_model()
        self.assertIsNotNone(model)
        assert model is not None
        restored = RoomAffinityModel.from_dict(
            json.loads(json.dumps(model.to_dict())), self.room_encoder.room_graph
        )
        condition = RoomCondition("TRADING", "O_GOLD", 2, 1, 12.0)
        self.assertAlmostEqual(
            model.view(condition).bias, restored.view(condition).bias, places=9
        )
        self.assertEqual(
            model.affinity_scores(condition, self.trading_names, self.embeddings),
            restored.affinity_scores(condition, self.trading_names, self.embeddings),
        )

    def test_optimizer_affinity_readout_produces_conflict_free_schedule(self) -> None:
        optimizer = GraphScheduleOptimizer(
            self.game_data,
            self.roster,
            shard_formula="rock",
            graph_config=GraphOptimizerConfig(
                readout="affinity", use_cache=False, affinity_samples=32
            ),
        )
        result = optimizer.optimize(
            parse_layout("243"), mode="normal", shift_count=2, shift_hours=12, drone_policy="none"
        )
        self.assertEqual(find_conflicts(result), [])
        self.assertGreater(result.score, 0.0)
        stats = optimizer.graph_stats.get("affinityReadout") or {}
        self.assertEqual(stats.get("readout"), "room_operator_affinity")
        self.assertIn("roomGraph", optimizer.graph_stats)
        # 物化视图必须真的走了共享表，而不是悄悄退回每间房各拟合一套。
        spec = next(
            room
            for room in optimizer._room_specs(parse_layout("243"), "normal")
            if room.room_type == "TRADING"
        )
        view = optimizer._ensure_room_model(spec, 12.0)
        self.assertIsNotNone(view)
        assert view is not None
        self.assertNotIn("candidatePool", view.training)
        matrix = optimizer.affinity_matrix(
            [RoomCondition("TRADING", "O_GOLD", 3, 3, 12.0)] + [
                RoomCondition("MANUFACTURE", "F_GOLD", 3, 3, 12.0)
            ]
        )
        self.assertEqual(len(matrix), 2)
        self.assertTrue(matrix[0])
        self.assertNotEqual(matrix[0], matrix[1])

    def test_affinity_model_reuses_the_disk_cache(self) -> None:
        cache_dir = Path(self._temp.name) / "affinity_cache"
        config = GraphOptimizerConfig(
            readout="affinity", use_cache=True, cache_dir=str(cache_dir), affinity_samples=16
        )
        first = GraphScheduleOptimizer(
            self.game_data, self.roster, shard_formula="rock", graph_config=config
        )
        model = first.affinity_model()
        self.assertIsNotNone(model)
        self.assertGreater(first._cache.writes, 0)

        second = GraphScheduleOptimizer(
            self.game_data, self.roster, shard_formula="rock", graph_config=config
        )
        restored = second.affinity_model()
        self.assertIsNotNone(restored)
        self.assertGreater(second._cache.hits, 0)
        assert model is not None and restored is not None
        condition = RoomCondition("TRADING", "O_GOLD", 3, 3, 12.0)
        self.assertEqual(
            model.unary_weights(condition), restored.unary_weights(condition)
        )
        self.assertEqual(
            model.view(condition).pair[: model.embedding_dim],
            restored.view(condition).pair[: restored.embedding_dim],
        )

    def test_affinity_readout_falls_back_to_per_room_without_numpy(self) -> None:
        from unittest import mock

        from arknights_schedule_generator import graph_conv

        optimizer = GraphScheduleOptimizer(
            self.game_data,
            self.roster,
            shard_formula="rock",
            graph_config=GraphOptimizerConfig(readout="affinity", use_cache=False),
        )
        spec = next(
            room
            for room in optimizer._room_specs(parse_layout("243"), "normal")
            if room.room_type == "TRADING"
        )
        with mock.patch.object(graph_conv, "_np", None):
            self.assertFalse(optimizer.graph_config.uses_affinity_readout)
            self.assertIsNone(optimizer.affinity_model())
            model = optimizer._ensure_room_model(spec, 12.0)
        self.assertIsNotNone(model)
        assert model is not None
        # 每间房各拟合一套的老路径才会带 candidatePool 训练信息。
        self.assertIn("candidatePool", model.training)


class BondBasisTest(unittest.TestCase):
    """显式羁绊基：只在语义成立的房间/产物上开，且别名标签要合并。"""

    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.data_dir = Path(self._temp.name)
        write_graph_test_data(self.data_dir)
        self.game_data = GameData.load(self.data_dir)
        self.roster = build_test_roster()
        self.graph = build_operator_graph(self.game_data, self.roster)
        self.names = [name for name, *_ in TRADING_OPERATORS]

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_in_room_tags_are_room_and_target_aware(self) -> None:
        self.assertEqual(in_room_count_tags("TRADING", "O_GOLD"), ("glasgow", "laterano", "mh2"))
        self.assertEqual(in_room_count_tags("TRADING", "O_DIAMOND"), ())
        self.assertEqual(in_room_count_tags("MANUFACTURE", "F_EXP"), ("A1", "ussg"))
        # 黑钢/莱茵用的是全基建计数，不能当房间级一元项
        self.assertNotIn("blacksteel", in_room_count_tags("MANUFACTURE", "F_GOLD"))

    def test_basis_marks_faction_membership_and_named_pairs(self) -> None:
        basis = build_bond_basis(
            self.graph, room_type="TRADING", target="O_GOLD", candidate_names=self.names
        )
        self.assertIn("siracusa", basis.tags)
        siracusa_flag = basis.unary_features("德克萨斯")[basis.tags.index("siracusa")]
        self.assertEqual(siracusa_flag, 1.0)
        self.assertEqual(basis.unary_features("能天使")[basis.tags.index("siracusa")], 0.0)
        # 德克萨斯与拉普兰德互为点名羁绊
        forward = basis.pair_features("拉普兰德", "德克萨斯")
        self.assertEqual(forward[0], 1.0)  # 拉普兰德点名德克萨斯
        self.assertEqual(basis.pair_features("德克萨斯", "拉普兰德")[1], 1.0)

    def test_alias_tags_are_deduplicated(self) -> None:
        basis = build_bond_basis(
            self.graph, room_type="TRADING", target="O_GOLD", candidate_names=self.names
        )
        signatures = [frozenset(basis.members[tag]) for tag in basis.tags]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_basis_serialisation_round_trip(self) -> None:
        basis = build_bond_basis(
            self.graph, room_type="TRADING", target="O_GOLD", candidate_names=self.names
        )
        restored = BondBasis.from_dict(basis.to_dict())
        self.assertEqual(restored.tags, basis.tags)
        self.assertEqual(restored.named, basis.named)
        for name in self.names[:5]:
            self.assertEqual(restored.unary_features(name), basis.unary_features(name))


class NumpyEquivalenceTest(unittest.TestCase):
    """numpy 加速路径必须和纯 Python 路径给出同一个答案。"""

    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.data_dir = Path(self._temp.name)
        write_graph_test_data(self.data_dir)
        self.game_data = GameData.load(self.data_dir)
        self.graph = build_operator_graph(self.game_data, build_test_roster())

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_encoder_paths_agree(self) -> None:
        encoder = GraphEncoder(self.graph, EncoderConfig(hidden_dim=6, layers=2, seed=99))
        vectorised = encoder.embeddings()
        reference = encoder._embeddings_python()
        self.assertEqual(len(vectorised), len(reference))
        for left, right in zip(vectorised, reference):
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, b, places=9)

    def test_ridge_paths_agree(self) -> None:
        from unittest import mock

        from arknights_schedule_generator import graph_conv

        rows = [
            [1.0, 0.5, 0.0],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 1.0],
            [2.0, -1.0, 0.5],
            [0.5, 0.25, 1.5],
            [0.25, 2.0, -1.0],
        ]
        targets = [1.0, -2.0, 3.5, 0.5, 2.0, -1.25]
        vectorised = _solve_ridge(rows, targets, ridge=0.05)
        with mock.patch.object(graph_conv, "_np", None):
            reference = graph_conv._solve_ridge(rows, targets, ridge=0.05)
        self.assertEqual(len(vectorised), len(reference))
        for left, right in zip(vectorised, reference):
            self.assertAlmostEqual(left, right, places=6)

    def test_numpy_probe_reports_availability(self) -> None:
        self.assertIsInstance(numpy_available(), bool)


class GraphModelCacheTest(unittest.TestCase):
    """磁盘缓存：命中后必须给出完全一致的预测。"""

    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.cache_dir = Path(self._temp.name) / "cache"
        self.data_dir = Path(self._temp.name) / "data"
        self.data_dir.mkdir()
        write_graph_test_data(self.data_dir)
        self.game_data = GameData.load(self.data_dir)
        self.roster = build_test_roster()

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_round_trip_prediction_is_identical(self) -> None:
        cache = GraphModelCache(directory=self.cache_dir)
        self.assertTrue(cache.active)
        payload = {"hello": [1, 2, 3]}
        key = cache.key(payload)
        self.assertIsNone(cache.load(key))
        cache.save(key, {"value": 7.5})
        loaded = cache.load(key)
        self.assertEqual(loaded["value"], 7.5)
        self.assertEqual(loaded["schema"], CACHE_SCHEMA_VERSION)

    def test_optimizer_reuses_fitted_model_from_disk(self) -> None:
        config = GraphOptimizerConfig(use_cache=True, cache_dir=str(self.cache_dir), fit_samples=96)
        first = GraphScheduleOptimizer(
            self.game_data, self.roster, shard_formula="rock", graph_config=config
        )
        spec = next(
            room
            for room in first._room_specs(parse_layout("243"), "normal")
            if room.room_type == "TRADING"
        )
        first._room_combos(spec, set(), 12.0)
        self.assertGreater(first._cache.writes, 0)

        second = GraphScheduleOptimizer(
            self.game_data, self.roster, shard_formula="rock", graph_config=config
        )
        second._room_combos(spec, set(), 12.0)
        self.assertGreater(second._cache.hits, 0)
        self.assertEqual(second._cache.misses, 0)
        first_model = first._models[first._model_key(spec, 12.0)]
        second_model = second._models[second._model_key(spec, 12.0)]
        assert first_model is not None and second_model is not None
        self.assertEqual(first_model.unary, second_model.unary)
        self.assertEqual(first_model.pair, second_model.pair)
        names = ["能天使", "德克萨斯", "拉普兰德"]
        embeddings = second._embeddings()
        available = [name for name in names if name in embeddings]
        if len(available) == len(names):
            self.assertAlmostEqual(
                first_model.score(names, embeddings),
                second_model.score(names, embeddings),
                places=12,
            )


class RealDataGraphTest(unittest.TestCase):
    """用真实游戏数据跑的关键回归；缓存缺失时自动跳过。"""

    @classmethod
    def setUpClass(cls) -> None:
        if not DATA_CACHE.exists() or not FIXTURE_ROSTER.exists():
            raise unittest.SkipTest("需要 data/cache 与示例练度表")
        cls.game_data = GameData.load(DATA_CACHE)
        cls.roster = load_roster_xlsx(FIXTURE_ROSTER)

    def test_real_graph_exposes_faction_and_named_bonds(self) -> None:
        graph = build_operator_graph(self.game_data, self.roster)
        report = graph.audit_report()
        self.assertGreater(report["factionGroupCount"], 20)
        self.assertGreaterEqual(report["namedDependencyPairCount"], 20)
        self.assertGreater(report["edgeCountByRelation"]["combo"], 0)
        self.assertIn("德克萨斯", dict(graph.neighbors("拉普兰德", "named")))

    def test_real_graph_marks_self_contained_morale_nodes(self) -> None:
        """「自律」这类恢复自给自足的干员在图里是**特殊节点**（不接受任何恢复边）。"""
        graph = build_operator_graph(self.game_data, self.roster)
        report = graph.audit_report()

        self.assertIn("菲亚梅塔", report["specialMoraleNodes"])
        self.assertEqual(report["specialMoraleNodeCount"], len(report["specialMoraleNodes"]))
        self.assertEqual(graph.meta["specialMoraleNodes"], report["specialMoraleNodes"])
        # 标注不参与 GCN 特征宽度，避免无谓地让读出层缓存失效。
        self.assertEqual(report["featureWidth"], len(FEATURE_NAMES))

    def test_real_shortlist_still_fills_a_room_after_the_top_pool_is_excluded(self) -> None:
        """排除掉「无排除池」的头部之后，短名单不能把房间填成 1 人。

        技能的变体表是按无排除候选池建的，所以被排除之后池子里剩下的干员大多不在那张表里；
        v1 的 O_DIAMOND 常数标签把「房间只填 1 人」掩盖成满产，v2 之后它直接掉分
        （333 平衡搓玉 3×8 实测 −144 分）。这条锁住「退回全量枚举」的兜底：
        只要变体表里还剩 1 个人（`0 < len(names) < capacity`），就必须枚举出满员组合。
        """
        graph = GraphScheduleOptimizer(self.game_data, self.roster, shard_formula="rock")
        # 用 2 工位的合成玉贸易站，让全量枚举保持便宜（C(32,2)）。
        spec = RoomSpec("trading_1", "TRADING", "O_DIAMOND", 2, 2, 2)
        pool = [
            candidate.skill.operator_name
            for candidate in graph._combo_candidate_pool("TRADING", "O_DIAMOND", set())
        ]
        # 只留 1 个「变体表里有人」的干员，其余全部排除。
        kept = pool[-1]
        excluded = {name for name in pool if name != kept}

        available = graph._combo_candidate_pool("TRADING", "O_DIAMOND", excluded)
        self.assertGreaterEqual(len(available), spec.capacity)
        combos = graph._room_combos(spec, excluded, 12.0)
        self.assertTrue(combos)
        self.assertEqual({len(combo.operators) for combo in combos}, {spec.capacity})

    def test_real_optimizer_keeps_guide_mechanism_trio(self) -> None:
        """图短名单必须包含刊例表里 200% 那一行的三人组。"""
        hybrid = ScheduleOptimizer(self.game_data, self.roster, shard_formula="rock")
        graph = GraphScheduleOptimizer(self.game_data, self.roster, shard_formula="rock")
        spec = next(
            room
            for room in hybrid._room_specs(parse_layout("243"), "balanced-orundum")
            if room.room_type == "TRADING" and room.target == "O_GOLD"
        )
        exhaustive = hybrid._room_combos(spec, set(), 12.0)
        shortlist = graph._room_combos(spec, set(), 12.0)
        best_names = {skill.operator_name for skill in exhaustive[0].operators}
        shortlisted_pairs = [
            {skill.operator_name for skill in combo.operators} for combo in shortlist
        ]
        self.assertIn(best_names, shortlisted_pairs)


if __name__ == "__main__":
    unittest.main()
