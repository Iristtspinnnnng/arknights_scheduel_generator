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
    EncoderConfig,
    GraphEncoder,
    RoomQuery,
    _solve_ridge,
    _spearman,
    fit_room_value_model,
    numpy_available,
    rank_combinations,
)
from arknights_schedule_generator.graph_features import (
    FEATURE_NAMES,
    build_operator_graph,
)
from arknights_schedule_generator.graph_optimizer import (
    GRAPH_OPTIMIZER_MODEL_VERSION,
    GraphOptimizerConfig,
    GraphScheduleOptimizer,
)
from arknights_schedule_generator.models import RosterOperator
from arknights_schedule_generator.optimizer import ScheduleOptimizer, parse_layout
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
