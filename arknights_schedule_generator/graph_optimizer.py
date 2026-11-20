"""基于静态图卷积的排班优化器。

它和旧的混合搜索优化器（:class:`~arknights_schedule_generator.optimizer.ScheduleOptimizer`）
共用同一套收益口径、同一套约束、同一份导出契约，唯一的区别是**房间里放谁**这件事怎么决定：

* 旧混合搜索：把一个房间的候选池（上限 32 个技能变体）做全组合枚举 ——
  ``C(32,3)=4960`` 种三人组合，每一种都调用一次 ``ProductionSimulator.evaluate_room``，
  再按真实分数排序取前 64。一次 ``optimize()`` 里这个动作会发生几十到上百次。
* 本优化器：先用图卷积把每个干员编码成“自身能力 + 阵营/羁绊上下文”的向量，
  再用一个闭式拟合的读出层（一元 + 二元协同）给组合打分，只把**前 64 个**候选
  送进真正的模拟器。模拟器仍然是唯一的裁判，神经网络只负责“提名”。

因此：

* 收益口径不变（``ProductionReport.score`` 完全由 ``ProductionSimulator`` 计算）；
* 约束不变（继承父类的班次/宿舍/循环边界/容量逻辑）；
* 变的是候选生成方式 —— 从“穷举后排序”变成“图神经网络提名 + 精确复核”。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations as _combinations
from pathlib import Path
from typing import Any, Sequence

from .graph_bonds import BondBasis, build_bond_basis
from .graph_cache import GraphModelCache
from .graph_conv import (
    EncoderConfig,
    GraphEncoder,
    RoomQuery,
    RoomValueModel,
    fit_room_value_model,
    rank_combinations,
)
from .graph_features import StaticOperatorGraph, build_operator_graph
from .models import BaseSkill
from .optimizer import (
    COMBO_CANDIDATE_TOP_LIMIT,
    OPTIMIZER_MODEL_VERSION,
    Candidate,
    RoomCombo,
    RoomSpec,
    ScheduleOptimizer,
    candidate_variant_key,
    combo_closure_candidates,
    insertion_objective_score,
    insertion_search_summary,
    insertion_spec_key,
    local_quality_audit,
    repeat_shift_cycle,
    target_selection_score,
    unique_candidate_variants,
)
from .morale import MAX_CYCLE_HOURS
from .presets import normalize_mode
from .production import (
    DEFAULT_PURE_GOLD_TARGET_PER_DAY,
    DEFAULT_PURE_GOLD_TOLERANCE,
    ProductionSimulator,
    pure_gold_balance_quality,
)

__all__ = ["GraphScheduleOptimizer", "GraphOptimizerConfig", "GRAPH_OPTIMIZER_MODEL_VERSION"]

# 和父类区分开，避免推荐器的候选缓存在两种算法之间串用。
GRAPH_OPTIMIZER_MODEL_VERSION = 100 + OPTIMIZER_MODEL_VERSION

# 特征定义版本：改动节点特征 / 羁绊基 / 读出层结构时必须 +1，否则旧磁盘缓存会失效。
FEATURE_VERSION = 2

_ROOM_QUERY_ROOM_TYPES = {"TRADING", "MANUFACTURE", "POWER"}


@dataclass(frozen=True)
class GraphOptimizerConfig:
    """图优化器的超参。全部是固定的，不做梯度训练。"""

    hidden_dim: int = 8
    layers: int = 2
    seed: int = 20260624
    shortlist: int = 64
    """送进模拟器精确复核的候选组合数。父类的 beam 宽度同样是 64，所以这个值
    已经覆盖了父类实际会用到的全部候选。"""
    beam: int = 18
    """读出层做一元预筛后保留的干员数；组合枚举规模是 ``C(beam, capacity)``。"""
    lexicographic_prefix: int = 96
    """额外补进短名单的「枚举顺序前缀」组合数，用于和父类的同分排序对齐。"""
    initial_starts: tuple[int, ...] = (12, 0)
    """初始排班的多起手策略：每一项是一个羁绊闭包预算；用同一个目标函数择优。

    默认 (12, 0) = 「开闭包」和「关掉闭包」各生成一套初始排班，再选更好的。
    这套做法把「同一开关在不同场景符号相反」的问题变成取最大值，代价是
    多几轮 beam 搜索和每套一次 ``simulator.evaluate``。设为 ``(12,)`` 可回到单起手。"""
    bond_closure_budget: int = 12
    """羁绊闭包预算：最多为多少个「有 named 伙伴」的干员定向提名一个组合。"""
    coverage_extra: int = 0
    """覆盖保障预算：保证候选池里每个干员至少进入一个送审组合。

    默认 0（关闭）：实测它把一批低一元分的干员强行塞进短名单，会打乱同分组合的先后，
    在 243/不搓玉 上净损失约 9 分。羁绊归 ``bond_closure_budget`` 管，两者已拆开。"""
    mechanism_partner_pool: int = 12
    """机制超边补齐伙伴时，从一元排名前几名里取。"""
    mechanism_keep: int = 16
    """机制超边按精确分数保留前几个（每个房间规格只算一次，之后缓存）。"""
    use_bonds: bool = True
    """是否把阵营人数 / 点名羁绊写成显式线性特征（系数可直接读出）。"""
    bond_max_tags: int = 24
    """显式阵营基的标签上限，直接决定读出层多出来的参数个数。"""
    bond_ridge_scale: float = 4.0
    """显式羁绊基的收缩强度相对嵌入基的倍数（样本少、列少，需要更强正则）。"""
    use_cache: bool = True
    """是否把拟合好的读出层与机制超边落盘复用（跨进程）。"""
    cache_dir: str | None = "outputs/graph_model_cache"
    """缓存目录；None 表示只用内存缓存。"""
    fit_samples: int = 288
    """拟合读出层时采样的房间组合数。

    注意这里**不是**越大越好：读数层的角色是「提名」而不是「预测」。实测把样本从 288
    加到 1536，holdout R² 从 0.82 升到 0.88、MAE 从 2.4 降到 1.5，但最终排班分数反而
    系统性下降约 4%——因为模型越锐利，短名单越集中在同一批干员上，下游 beam 搜索失去
    多样性。若哪天给短名单加了显式多样性约束，可以再把这 个值调高。"""
    ridge: float = 1.0
    min_fit_samples: int = 24
    candidate_closure: bool = True
    """是否用图上的 ``named``/``combo`` 边做候选闭包，替代父类的硬编码干员名单。"""
    record_only: bool = False
    """只记录图模型的推荐而不真正改变搜索（用于诊断/对照）。"""
    prefer_score_within_tolerance: bool = True
    """纯赤金平衡的循环重复次数：在容差内选分数最高的，而不是第一个达标的。"""
    limit_insertion_search: bool = False
    """是否用图先验 + 预算限制插入搜索。默认关闭：实测限制预算虽然更快，但最终分数对
    插入顺序高度敏感（±3%），保留父类的完整插入搜索能让质量更稳定、可比性更好。"""
    insertion_group_budget: int = 48
    """每轮最多尝试多少个插入组（父类上限是 64，`DIAGNOSTIC_INSERTION_GROUP_LIMIT`）。"""
    insertion_max_passes: int = 4
    """插入搜索的最大轮数（父类最多 8 轮）。"""

    def encoder_config(self) -> EncoderConfig:
        return EncoderConfig(hidden_dim=self.hidden_dim, layers=self.layers, seed=self.seed)


class GraphScheduleOptimizer(ScheduleOptimizer):
    """用静态图卷积替代房间组合穷举的优化器。"""

    @staticmethod
    def optimizer_model_version() -> int:
        return GRAPH_OPTIMIZER_MODEL_VERSION

    def __init__(
        self,
        game_data,
        roster,
        *,
        allow_upgrades: bool = False,
        upgrade_cost_weight: float = 0.015,
        shard_formula: str = "rock",
        graph_config: GraphOptimizerConfig | None = None,
    ) -> None:
        super().__init__(
            game_data,
            roster,
            allow_upgrades=allow_upgrades,
            upgrade_cost_weight=upgrade_cost_weight,
            shard_formula=shard_formula,
        )
        self.graph_config = graph_config or GraphOptimizerConfig()
        self._graph: StaticOperatorGraph | None = None
        self._encoder: GraphEncoder | None = None
        self._models: dict[tuple[Any, ...], RoomValueModel | None] = {}
        self._unary_cache: dict[tuple[Any, ...], dict[str, float]] = {}
        self._pair_cache: dict[tuple[Any, ...], dict[tuple[str, str], float]] = {}
        self._variant_cache: dict[tuple[Any, ...], dict[str, tuple[BaseSkill, ...]]] = {}
        self._mention_cache: dict[str, frozenset[str]] = {}
        self._fatigue_cache: dict[tuple[Any, ...], dict[str, float]] = {}
        self._mechanism_cache: dict[tuple[Any, ...], list[tuple[str, ...]]] = {}
        self._embedding_by_name: dict[str, tuple[float, ...]] = {}
        self._graph_room_combos_cache: dict[tuple[Any, ...], tuple[RoomCombo, ...]] = {}
        self._bond_basis_cache: dict[tuple[Any, ...], BondBasis | None] = {}
        config = self.graph_config
        self._cache = GraphModelCache(
            directory=Path(config.cache_dir) if config.cache_dir else None,
            enabled=config.use_cache,
        )
        self.graph_stats: dict[str, Any] = {}
        # 多起手：同一份模型可以生成多种短名单策略，用**同一目标函数**挑初始排班。
        self._closure_override: int | None = None
        self._active_optimize_kwargs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 多起手：用同一个目标函数挑初始排班
    # ------------------------------------------------------------------
    def _build_shifts(
        self,
        layout,
        room_specs,
        *,
        shift_durations=None,
        shift_count=None,
        shift_hours=None,
        shift_times,
    ):
        """先各自生成一套初始排班，再按 ``optimize()`` 真正使用的目标函数择优。

        为什么要这么做：短名单策略（是否做羁绊闭包）会让**初始排班**在
        560.2 / 567.5 这种量级上摆动，而下游插入搜索只能部分补偿。
        开闭包在「平衡搓玉」上赚 14 分、在「不搓玉」上亏 9 分——符号取决于
        那次提名是否抢走了别的房间更需要的干员。
        逐房间贪心本来就没法给「抢走稀缺干员」定价，所以与其赌一个策略，
        不如两套都生成、用同一个 ``target_selection_score`` 打分选更好的那套。
        代价只是多几轮 beam 搜索 + 每套一次 ``simulator.evaluate``。
        """
        budgets = self.graph_config.initial_starts
        if len(budgets) <= 1:
            # 单起手时也要尊重指定的预算，否则 (0,) 会被误当成「用默认值」。
            self._closure_override = budgets[0] if budgets else None
            return super()._build_shifts(
                layout,
                room_specs,
                shift_durations=shift_durations,
                shift_count=shift_count,
                shift_hours=shift_hours,
                shift_times=shift_times,
            )

        best: tuple[float, list, list[str]] | None = None
        for budget in budgets:
            self._closure_override = budget
            self._graph_room_combos_cache.clear()
            shifts, warnings = super()._build_shifts(
                layout,
                room_specs,
                shift_durations=shift_durations,
                shift_count=shift_count,
                shift_hours=shift_hours,
                shift_times=shift_times,
            )
            score = self._start_score(shifts)
            if best is None or score > best[0]:
                best = (score, shifts, warnings)
        self._closure_override = None
        assert best is not None
        self.graph_stats.setdefault("starts", {})
        self.graph_stats["starts"]["budgets"] = list(budgets)
        return best[1], best[2]

    def _start_score(self, shifts) -> float:
        """用 ``optimize()`` 实际排序用的目标给候选初始排班打分。"""
        kwargs = self._active_optimize_kwargs
        simulator = ProductionSimulator(
            self.game_data,
            shard_formula=self.shard_formula,
            drone_policy=kwargs.get("drone_policy", "none"),
            calibration_profile="guide",
            reference_expected_daily=kwargs.get("drone_reference_daily"),
            pure_gold_target=kwargs.get("pure_gold_target", DEFAULT_PURE_GOLD_TARGET_PER_DAY),
            pure_gold_tolerance=kwargs.get("pure_gold_tolerance", DEFAULT_PURE_GOLD_TOLERANCE),
        )
        report = simulator.evaluate(shifts)
        mode = normalize_mode(kwargs.get("mode", "normal"))
        return target_selection_score(
            mode,
            report,
            min_lmd_gross=kwargs.get("min_lmd_gross", 0.0),
            min_exp=kwargs.get("min_exp", 0.0),
            min_orundum=kwargs.get("min_orundum", 0.0),
        )

    # ------------------------------------------------------------------
    # 对外契约：标上自己的模型版本号
    # ------------------------------------------------------------------
    def optimize(self, layout, **kwargs):
        """继承父类的全部流程，只把审计里的模型版本号换成图优化器的。"""
        self._active_optimize_kwargs = dict(kwargs)
        result = super().optimize(layout, **kwargs)
        insertion_search = result.diagnostic_insertion_search
        if isinstance(insertion_search, dict):
            insertion_search.setdefault("cacheValidation", {})
            insertion_search["cacheValidation"].update(
                {
                    "optimizerModelVersion": GRAPH_OPTIMIZER_MODEL_VERSION,
                    "maxGroups": self.diagnostic_insertion_group_limit(),
                    "jointProductionCandidateLimit": self.joint_production_candidate_limit(),
                    "optimizer": "graph",
                }
            )
            insertion_search["optimizer"] = "graph"
            if self.graph_stats:
                insertion_search["graphModel"] = self.graph_stats
        return result

    # ------------------------------------------------------------------
    # 静态图上下文
    # ------------------------------------------------------------------
    def graph(self) -> StaticOperatorGraph:
        if self._graph is None:
            self._graph = build_operator_graph(
                self.game_data,
                self.roster,
                allow_upgrades=self.allow_upgrades,
            )
        return self._graph

    def encoder(self) -> GraphEncoder:
        if self._encoder is None:
            self._encoder = GraphEncoder(self.graph(), self.graph_config.encoder_config())
            embeddings = self._encoder.embeddings()
            self._embedding_by_name = {
                node.name: embeddings[index]
                for index, node in enumerate(self._encoder.graph.nodes)
            }
            self.graph_stats.setdefault("graph", {})
            self.graph_stats["graph"] = {
                "signature": self.graph().meta.get("signature"),
                "nodeCount": self.graph().size,
                "edgeCount": len(self.graph().edges),
                "embeddingDim": self._encoder.embedding_dim,
            }
        return self._encoder

    def _embeddings(self) -> dict[str, tuple[float, ...]]:
        self.encoder()
        return self._embedding_by_name

    # ------------------------------------------------------------------
    # 房间价值模型
    # ------------------------------------------------------------------
    def _model_key(self, spec: RoomSpec, duration_hours: float) -> tuple[Any, ...]:
        return (
            *self._cache_context_key(),
            "graph_room_model",
            spec.room_type,
            spec.target,
            spec.room_level,
            spec.slots,
            round(float(duration_hours), 6),
        )

    def _pool_key(self, spec: RoomSpec, duration_hours: float) -> tuple[Any, ...]:
        return (
            *self._cache_context_key(),
            "graph_room_pool",
            spec.room_type,
            spec.target,
            spec.room_level,
            spec.slots,
            round(float(duration_hours), 6),
        )

    def _variant_options(self, spec: RoomSpec, duration_hours: float) -> dict[str, tuple[BaseSkill, ...]]:
        """每个干员在该房间里的技能变体，按候选池顺序（``score_skill`` 降序）。

        父类把同一个干员的多个技能变体当作不同的组合元素全量穷举；图模型只对
        “干员”建模，所以变体必须用规则挑。挑法见 :meth:`_pick_variant`：
        优先选“文本里点名了同房队友”的那个变体，这正是羁绊图的边在起作用。
        """
        key = self._pool_key(spec, duration_hours)
        cached = self._variant_cache.get(key)
        if cached is not None:
            return cached
        pool = self._combo_candidate_pool(spec.room_type, spec.target, set())
        options: dict[str, list[BaseSkill]] = {}
        seen_buffs: dict[str, set[str]] = {}
        for candidate in pool:
            skill = candidate.skill
            name = skill.operator_name
            bucket = seen_buffs.setdefault(name, set())
            if skill.buff_id in bucket:
                continue
            bucket.add(skill.buff_id)
            options.setdefault(name, []).append(skill)
        result = {name: tuple(skills) for name, skills in options.items()}
        self._variant_cache[key] = result
        return result

    def _best_variant_map(self, spec: RoomSpec, duration_hours: float) -> dict[str, BaseSkill]:
        return {
            name: skills[0]
            for name, skills in self._variant_options(spec, duration_hours).items()
            if skills
        }

    def _skill_mentions(self, skill: BaseSkill) -> frozenset[str]:
        cached = self._mention_cache.get(skill.buff_id)
        if cached is not None:
            return cached
        text = f"{skill.buff_name} {skill.description}"
        roster_names = set(self.roster_by_name)
        mentions = frozenset(
            name
            for name in roster_names
            if len(name) >= 2 and name != skill.operator_name and name in text
        )
        self._mention_cache[skill.buff_id] = mentions
        return mentions

    def _pick_variant(
        self,
        options: Sequence[BaseSkill],
        combo_names: Sequence[str],
    ) -> BaseSkill:
        """在多个技能变体里挑一个。

        优先级：文本点名了同房队友的变体 > ``score_skill`` 更高的变体。
        前者就是羁绊图 ``named`` 边的用途——旧优化器靠全量枚举“碰巧”发现
        「相伴 + 能天使」这类联动，这里用图关系直接定向选择。
        """
        if len(options) == 1:
            return options[0]
        partner_names = set(combo_names) - {options[0].operator_name}
        best = options[0]
        best_key = (-1, 0.0)
        for skill in options:
            mentions = self._skill_mentions(skill)
            linked = 2 if mentions & partner_names else (1 if mentions else 0)
            key = (linked, float(skill.parsed_score or 0.0))
            if key > best_key:
                best_key = key
                best = skill
        return best

    def _fatigue_extra(self, spec: RoomSpec, duration_hours: float) -> dict[str, float]:
        """心情/疲劳的可加修正项（每个干员一项）。

        ``ProductionSimulator`` 把 ``-fatigueRisk * 2`` 计入房间分数，而
        ``fatigueRisk = max(0, 加班 + Σ每小时心情消耗 × 时长)``。对同房间干员来说
        这部分是**可加**的，所以不需要神经网络去“学”，直接精确算出来加到一元项上；
        读出层只拟合产能，避免把心情和产能混在同一个低维投影里。
        切换技能变体会改变心情消耗，所以这里按干员取其默认变体。
        """
        if spec.room_type not in {"TRADING", "MANUFACTURE"}:
            return {}
        from .skill_rules import evaluate_room_effect

        key = (*self._pool_key(spec, duration_hours), "fatigue")
        cached = self._fatigue_cache.get(key)
        if cached is not None:
            return cached
        result: dict[str, float] = {}
        for name, skills in self._variant_options(spec, duration_hours).items():
            if not skills:
                continue
            effect = evaluate_room_effect([skills[0]], spec.target)
            delta = float(effect.fatigue_delta_per_hour or 0.0)
            result[name] = -2.0 * float(duration_hours) * delta
        self._fatigue_cache[key] = result
        return result

    def _ensure_room_model(
        self, spec: RoomSpec, duration_hours: float
    ) -> RoomValueModel | None:
        key = self._model_key(spec, duration_hours)
        if key in self._models:
            return self._models[key]
        config = self.graph_config
        encoder = self.encoder()
        variants = self._best_variant_map(spec, duration_hours)
        if not variants:
            self._models[key] = None
            return None
        fatigue_extra = self._fatigue_extra(spec, duration_hours)
        query = RoomQuery(
            room_type=spec.room_type,
            target=spec.target,
            room_level=spec.room_level,
            slots=spec.slots if spec.slots is not None else spec.capacity,
        )
        bond_basis = self._bond_basis(spec, duration_hours, variants)

        cache_key = self._cache.key(
            {
                "kind": "room_value_model",
                "graph": self.graph().meta.get("signature"),
                "query": query.to_dict(),
                "capacity": spec.capacity,
                "samples": config.fit_samples,
                "ridge": config.ridge,
                "seed": config.seed,
                "bonds": None if bond_basis is None else bond_basis.to_dict(),
                "hiddenDim": config.hidden_dim,
                "layers": config.layers,
                "featureVersion": FEATURE_VERSION,
            }
        )
        cached = self._cache.load(cache_key)
        if cached and cached.get("model"):
            try:
                model = RoomValueModel.from_dict(cached["model"])
                self._models[key] = model
                self._record_runtime_count("graphModel.cacheHit")
                self._store_model_stats(spec, model)
                return model
            except (KeyError, TypeError, ValueError):
                pass
        self._record_runtime_count("graphModel.cacheMiss")

        def oracle(names: tuple[str, ...]) -> float:
            operators = [variants[name] for name in names if name in variants]
            if not operators:
                return 0.0
            raw = self._score_room_combo(spec, operators, duration_hours)
            return raw - sum(fatigue_extra.get(name, 0.0) for name in names)

        model = fit_room_value_model(
            encoder,
            query,
            list(variants),
            oracle,
            capacity=spec.capacity,
            samples=config.fit_samples,
            seed=config.seed,
            ridge=config.ridge,
            min_combo_size=1,
            bond_basis=bond_basis,
            bond_ridge_scale=config.bond_ridge_scale,
        )
        if model is not None and model.training.get("samples", 0) < config.min_fit_samples:
            model = None
        self._models[key] = model
        self._store_model_stats(spec, model)
        if model is not None:
            self._cache.save(cache_key, {"model": model.to_dict()})
        return model

    def _store_model_stats(self, spec: RoomSpec, model: RoomValueModel | None) -> None:
        self.graph_stats.setdefault("models", {})
        self.graph_stats["models"][
            f"{spec.room_type}:{spec.target}:{spec.room_level}:{spec.slots}"
        ] = None if model is None else model.describe()

    def _bond_basis(
        self,
        spec: RoomSpec,
        duration_hours: float,
        variants: dict[str, BaseSkill] | None = None,
    ) -> BondBasis | None:
        """显式羁绊基：阵营人数与点名羁绊指示函数。"""
        if not self.graph_config.use_bonds:
            return None
        key = (*self._pool_key(spec, duration_hours), "bonds")
        if key in self._bond_basis_cache:
            return self._bond_basis_cache[key]
        names = variants if variants is not None else self._best_variant_map(spec, duration_hours)
        basis = build_bond_basis(
            self.graph(),
            room_type=spec.room_type,
            candidate_names=list(names),
            target=spec.target,
            max_tags=self.graph_config.bond_max_tags,
        )
        self._bond_basis_cache[key] = basis
        return basis

    # ------------------------------------------------------------------
    # 候选池：用图上的羁绊边做闭包
    # ------------------------------------------------------------------
    def _graph_partner_names(self, skill: BaseSkill) -> set[str]:
        graph = self.graph()
        name = skill.operator_name
        partners: set[str] = set()
        for relation in ("named", "combo"):
            for partner, _weight in graph.neighbors(name, relation):
                partners.add(partner)
        return partners

    def _combo_candidate_pool(
        self, room_type: str, target: str | None, excluded: set[str]
    ) -> list[Candidate]:
        if not self.graph_config.candidate_closure:
            return super()._combo_candidate_pool(room_type, target, excluded)
        cache_key = (
            *self._cache_context_key(),
            "graph_combo_candidate_pool",
            room_type,
            target,
            tuple(sorted(excluded)),
        )
        cached = self._combo_candidate_pool_cache.get(cache_key)
        if cached is not None:
            self._record_runtime_count("graphComboCandidatePool.cacheHit")
            return list(cached)
        self._record_runtime_count("graphComboCandidatePool.cacheMiss")
        candidates = self._combo_candidates(room_type, target, excluded)
        if room_type == "POWER":
            result = candidates[:16]
            self._combo_candidate_pool_cache[cache_key] = tuple(result)
            return result

        top_candidates = unique_candidate_variants(candidates[:COMBO_CANDIDATE_TOP_LIMIT])
        # 仍然保留父类的机制性闭包（它编码了“特殊订单/组合技”这类硬约束），
        # 再叠加图上的 named/combo 邻居，让羁绊关系由数据结构而不是名单决定。
        protected_candidates = combo_closure_candidates(candidates, room_type, target)
        graph_names: set[str] = set()
        for candidate in top_candidates:
            if candidate.skill.room_type != room_type:
                continue
            graph_names.update(self._graph_partner_names(candidate.skill))

        pool = unique_candidate_variants([*top_candidates, *protected_candidates])
        pool_keys = {candidate_variant_key(candidate) for candidate in pool}
        selected = list(top_candidates)
        selected_keys = {candidate_variant_key(candidate) for candidate in selected}
        for candidate in protected_candidates:
            key = candidate_variant_key(candidate)
            if key in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(key)
            if len(selected) >= self._pool_limit():
                break
        if graph_names and len(selected) < self._pool_limit():
            best_by_name: dict[str, Candidate] = {}
            for candidate in candidates:
                name = candidate.skill.operator_name
                if name not in graph_names:
                    continue
                previous = best_by_name.get(name)
                if previous is None or candidate.score > previous.score:
                    best_by_name[name] = candidate
            for name, candidate in best_by_name.items():
                key = candidate_variant_key(candidate)
                if key in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.add(key)
                if len(selected) >= self._pool_limit():
                    break
        if len(selected) < self._pool_limit():
            for candidate in sorted(pool, key=lambda item: item.score, reverse=True):
                key = candidate_variant_key(candidate)
                if key in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.add(key)
                if len(selected) >= self._pool_limit():
                    break
        result = selected[: self._pool_limit()]
        self._combo_candidate_pool_cache[cache_key] = tuple(result)
        return result

    def _pool_limit(self) -> int:
        from .optimizer import COMBO_CANDIDATE_POOL_LIMIT

        return COMBO_CANDIDATE_POOL_LIMIT

    # ------------------------------------------------------------------
    # 核心：用图模型提名组合，模拟器复核
    # ------------------------------------------------------------------
    def _room_combos(
        self, spec: RoomSpec, excluded: set[str], duration_hours: float
    ) -> list[RoomCombo]:
        if self.graph_config.record_only:
            return super()._room_combos(spec, excluded, duration_hours)
        cache_key = (
            *self._cache_context_key(),
            "graph_room_combos",
            spec.room_id,
            spec.room_type,
            spec.target,
            spec.capacity,
            spec.room_level,
            spec.slots,
            round(duration_hours, 6),
            tuple(sorted(excluded)),
        )
        cached = self._graph_room_combos_cache.get(cache_key)
        if cached is not None:
            self._record_runtime_count("graphRoomCombos.cacheHit")
            return list(cached)
        self._record_runtime_count("graphRoomCombos.cacheMiss")

        candidates = self._combo_candidate_pool(spec.room_type, spec.target, excluded)
        if not candidates:
            self._graph_room_combos_cache[cache_key] = ()
            return []

        model = self._ensure_room_model(spec, duration_hours)
        if model is None:
            # 数据不足时退回父类：这类房间本来就不是瓶颈。
            result = super()._room_combos(spec, excluded, duration_hours)
            self._graph_room_combos_cache[cache_key] = tuple(result)
            return result

        variants = self._best_variant_map(spec, duration_hours)
        variant_options = self._variant_options(spec, duration_hours)
        known_names = set(variants)
        candidates_names = [candidate.skill.operator_name for candidate in candidates]
        names: list[str] = []
        seen_names: set[str] = set()
        for candidate in candidates:
            name = candidate.skill.operator_name
            if name in seen_names or name not in known_names:
                continue
            seen_names.add(name)
            names.append(name)
        if not names:
            # 变体表里一个人都没有：短名单没有信息优势，交给调用方的逐人贪心兜底
            # （``_assign_room`` 会按 ``_candidates`` 把房间填满）。
            self._graph_room_combos_cache[cache_key] = ()
            return []
        if len(names) < spec.capacity:
            # 技能变体表（``_variant_options``）是按**无排除**的候选池建的，所以排除一批
            # 干员之后，剩下的池子里可能有人的变体根本不在那张表里。这时短名单会把
            # 整间房填成 1~2 人——v1 的 O_DIAMOND 常数标签把这个缺陷完全掩盖了
            # （空房间也满产），v2 之后它直接掉分（333 平衡搓玉 3×8 实测 −144）。
            # 宁可退回全量枚举，也不要把房间填残。
            exhausted = super()._room_combos(spec, excluded, duration_hours)
            self._graph_room_combos_cache[cache_key] = tuple(exhausted)
            self._record_runtime_count("graphRoomCombos.exhaustiveFallback")
            return exhausted

        # 池内下标：父类同分时的先后顺序就是它。
        rank_of_name: dict[str, int] = {}
        for index, name in enumerate(candidates_names):
            rank_of_name.setdefault(name, index)

        unary_cache = self._unary_cache.setdefault(self._pool_key(spec, duration_hours), {})
        pair_cache = self._pair_cache.setdefault(self._pool_key(spec, duration_hours), {})
        ranked = rank_combinations(
            model,
            self._embeddings(),
            names,
            capacity=min(spec.capacity, len(names)),
            beam=self.graph_config.beam,
            top_k=self.graph_config.shortlist,
            unary_cache=unary_cache,
            pair_cache=pair_cache,
            unary_adjust=self._fatigue_extra(spec, duration_hours),
        )
        shortlisted = [combo_names for _predicted, combo_names in ranked]
        name_set = set(names)
        graph = self.graph()
        covered: set[str] = set()
        for combo in shortlisted:
            covered.update(combo)
        # 羁绊闭包：只针对「在池子里真有 named 伙伴」的干员，沿边把必须同房才生效的
        # 伙伴拼成一个组合，再用一元预测最强的干员补满。
        # 旧优化器要靠全量枚举“碰巧”撞上「相伴 + 能天使」这类联动，这里定向提名。
        closure_budget = (
            self.graph_config.bond_closure_budget
            if self._closure_override is None
            else self._closure_override
        )
        for name in names:
            if closure_budget <= 0:
                break
            partners = [
                partner
                for partner, _weight in graph.neighbors(name, "named")
                if partner in name_set and partner != name
            ]
            if not partners:
                continue
            combo_tuple = self._assemble_combo(
                [name, *partners[: max(0, spec.capacity - 1)]],
                names,
                unary_cache,
                spec.capacity,
            )
            if combo_tuple is None:
                continue
            closure_budget -= 1
            covered.update(combo_tuple)
            if combo_tuple not in shortlisted:
                shortlisted.append(combo_tuple)
        # 覆盖保障：让池子里每个干员至少进入一个送审组合。
        # 默认关闭——实测它会把一批低一元分的干员强行塞进短名单，打乱同分组合的先后，
        # 在 243/不搓玉 上要了 9 分（−11.55 → −2.39）。需要绝对安全时再打开。
        coverage_budget = self.graph_config.coverage_extra
        for name in names:
            if coverage_budget <= 0:
                break
            if name in covered:
                continue
            combo_tuple = self._assemble_combo([name], names, unary_cache, spec.capacity)
            if combo_tuple is None:
                continue
            coverage_budget -= 1
            covered.update(combo_tuple)
            if combo_tuple not in shortlisted:
                shortlisted.append(combo_tuple)
        # 机制超边：guide 刊例表里“必须同时上场才触发”的干员集合。
        # 这些是阶跃型增益（命中查表 → 固定大数值），线性读出层预测不了阶跃，
        # 所以把表本身当作静态先验，按房间规格预先精确打分、只保留最强的若干个。
        added_mechanisms = 0
        for mechanism in self._mechanism_combos(spec, duration_hours):
            if any(name in excluded for name in mechanism):
                continue
            if mechanism not in shortlisted:
                shortlisted.append(mechanism)
            added_mechanisms += 1
        # 平手兜底：父类是「先 combinations(pool, k) 再稳定排序」，所以同分组合中
        # 它优先看到的是**枚举顺序最靠前**的那批。而合成玉/赤金/经验这些房间经常大片同分
        # （例：L3 合成玉贸易室 4750 个组合全是 282.0），谁进前 64 完全由这个顺序决定。
        # 短名单只覆盖了父类枚举空间的一小部分，同分先后就会和父类不一致，
        # 进而把下游 beam/插入搜索带到另一条支路。这里补枚举前缀让同分区间对齐。
        prefix_budget = self.graph_config.lexicographic_prefix
        if prefix_budget > 0 and len(names) >= spec.capacity:
            ordered_names = sorted(names, key=lambda name: rank_of_name.get(name, 1 << 30))
            for combo in _combinations(ordered_names, spec.capacity):
                if prefix_budget <= 0:
                    break
                prefix_budget -= 1
                combo_tuple = tuple(sorted(combo))
                if combo_tuple not in shortlisted:
                    shortlisted.append(combo_tuple)

        self._record_runtime_count("graphShortlist.count", len(shortlisted))
        self._record_runtime_count("graphShortlist.mechanismCandidates", added_mechanisms)
        combos: list[RoomCombo] = []
        seen: set[tuple[tuple[str, str, tuple[str, ...]], ...]] = set()
        for combo_names in shortlisted:
            operators: list[BaseSkill] = []
            for name in combo_names:
                options = variant_options.get(name)
                if not options:
                    operators = []
                    break
                operators.append(self._pick_variant(options, combo_names))
            if not operators:
                continue
            key = tuple(sorted(candidate_variant_key(Candidate(s, 0.0)) for s in operators))
            if key in seen:
                continue
            seen.add(key)
            score = self._score_room_combo(spec, operators, duration_hours)
            combos.append(RoomCombo(operators=tuple(operators), score=score))
        self._record_runtime_count("graphShortlist.exactEvaluations", len(combos))
        # 同分组合的先后顺序会一路放大到最终排班（例如 L3 合成玉贸易站几乎全是 282.0，
        # 谁进这三格纯看排序）。父类的顺序是「按池下标字典序」——因为它是先
        # ``combinations(pool, k)`` 再稳定排序。这里用同样的平手规则，
        # 让短名单在分数相同的组合上与父类保持一致，避免下游搜索走到完全不同的支路。
        rank_of = {name: index for index, name in enumerate(candidates_names)}
        combos.sort(
            key=lambda combo: tuple(
                sorted(rank_of.get(skill.operator_name, len(rank_of)) for skill in combo.operators)
            )
        )
        combos.sort(
            key=lambda combo: (
                combo.score,
                sum(skill.unlocked for skill in combo.operators),
                -sum(
                    float(skill.upgrade.cost_score if skill.upgrade else 0.0)
                    for skill in combo.operators
                ),
            ),
            reverse=True,
        )
        self._graph_room_combos_cache[cache_key] = tuple(combos)
        return combos

    def _assemble_combo(
        self,
        pinned: Sequence[str],
        names: Sequence[str],
        unary_cache: dict[str, float],
        capacity: int,
    ) -> tuple[str, ...] | None:
        """把 ``pinned`` 里的干员固定下来，再用一元预测最强的干员补满到 ``capacity``。"""
        chosen = list(dict.fromkeys(pinned))
        if len(chosen) > capacity:
            return None
        if len(chosen) < capacity:
            fill_order = sorted(
                (other for other in names if other not in chosen),
                key=lambda other: (-unary_cache.get(other, 0.0), other),
            )
            chosen.extend(fill_order[: capacity - len(chosen)])
        if len(chosen) != capacity:
            return None
        return tuple(sorted(chosen))

    def _mechanism_combos(
        self,
        spec: RoomSpec,
        duration_hours: float,
    ) -> list[tuple[str, ...]]:
        """guide 刊例表里的“成组触发”干员集合 + 最强伙伴补齐，按房间规格缓存。

        这些机制是阶跃函数（三人同时上场才命中 200% 那一行），
        线性读出层没有能力预测阶跃，所以把表本身当作静态先验注入。
        每次 ``optimize()`` 里同一个规格会被查询几十次，所以这里按规格缓存，
        只用全量候选池算一次。
        """
        key = (*self._pool_key(spec, duration_hours), "mechanisms")
        cached = self._mechanism_cache.get(key)
        if cached is not None:
            return cached
        capacity = max(1, spec.capacity)
        variants = self._best_variant_map(spec, duration_hours)
        unary_cache = self._unary_cache.setdefault(self._pool_key(spec, duration_hours), {})
        cache_key = self._cache.key(
            {
                "kind": "mechanism_combos",
                "graph": self.graph().meta.get("signature"),
                "roomType": spec.room_type,
                "target": spec.target,
                "level": spec.room_level,
                "slots": spec.slots,
                "capacity": capacity,
                "duration": round(float(duration_hours), 6),
                "partnerPool": self.graph_config.mechanism_partner_pool,
                "keep": self.graph_config.mechanism_keep,
                "variants": sorted(
                    f"{name}:{skill.buff_id}" for name, skill in variants.items()
                ),
                "featureVersion": FEATURE_VERSION,
            }
        )
        stored = self._cache.load(cache_key)
        if stored and stored.get("combos") is not None:
            result = [tuple(combo) for combo in stored["combos"]]
            self._mechanism_cache[key] = result
            self._record_runtime_count("graphMechanism.cacheHit")
            return result
        ordered_by_unary = sorted(
            variants, key=lambda name: (-unary_cache.get(name, 0.0), name)
        )
        scored: list[tuple[float, tuple[str, ...]]] = []
        seen: set[tuple[str, ...]] = set()
        for required in self._mechanism_required_sets(spec):
            if not required or len(required) > capacity:
                continue
            if any(name not in variants for name in required):
                continue
            partners = [
                name
                for name in ordered_by_unary[: self.graph_config.mechanism_partner_pool]
                if name not in required
            ]
            for fill in _combinations(partners, capacity - len(required)):
                combo = tuple(sorted((*required, *fill)))
                if combo in seen:
                    continue
                seen.add(combo)
                operators = [variants[name] for name in combo]
                scored.append((self._score_room_combo(spec, operators, duration_hours), combo))
        scored.sort(key=lambda item: (-item[0], item[1]))
        result = [combo for _score, combo in scored[: self.graph_config.mechanism_keep]]
        self._mechanism_cache[key] = result
        self._record_runtime_count("graphMechanism.cacheMiss")
        self._cache.save(cache_key, {"combos": [list(combo) for combo in result]})
        return result

    def _mechanism_required_sets(self, spec: RoomSpec) -> list[tuple[str, ...]]:
        from .production import (
            GUIDE_LMD_TRADE_LOOKUPS,
            GUIDE_MANUFACTURE_LOOKUPS,
        )

        key = (spec.room_type, spec.target, spec.room_level, spec.slots)
        cached = self._mechanism_cache.get(key)
        if cached is not None:
            return cached
        sets: list[tuple[str, ...]] = []
        if spec.room_type == "TRADING" and spec.target == "O_GOLD":
            for row in GUIDE_LMD_TRADE_LOOKUPS:
                if int(row.get("level", 0)) not in {0, spec.room_level}:
                    continue
                required = tuple(sorted(row.get("required") or ()))
                if required:
                    sets.append(required)
        elif spec.room_type == "MANUFACTURE":
            for row in GUIDE_MANUFACTURE_LOOKUPS:
                if row.get("target") != spec.target:
                    continue
                required = tuple(sorted(row.get("required") or ()))
                if required:
                    sets.append(required)
        self._mechanism_cache[key] = sets
        return sets

    # ------------------------------------------------------------------
    # 插入搜索：按图先验排序 + 预算
    # ------------------------------------------------------------------
    def _order_insertion_groups(self, groups) -> list:
        """按“图价值模型看好程度”给插入组排序。

        旧优化器按构造顺序搜索全部插入组，每轮都整组重排，实测 534 次尝试里只有 9 次
        真正提分。这里用静态图的节点嵌入给组里的干员打分：在目标房间里
        一元价值高的干员优先尝试，把预算花在更可能提分的方向上。
        """
        embeddings = self._embeddings()
        models = self._models
        unaries = self._unary_cache

        def group_priority(group) -> tuple[float, str]:
            total = 0.0
            for spec in group.specs:
                best = -1e9
                for key, model in models.items():
                    if model is None or key[4] != spec.room_type:
                        continue
                    if spec.target is not None and key[5] != spec.target:
                        continue
                    vector = embeddings.get(spec.operator_name)
                    if vector is None:
                        continue
                    value = sum(w * v for w, v in zip(model.unary, vector))
                    best = max(best, value)
                if best > -1e9:
                    total += best
                else:
                    total += 0.0
            return (total, group.id)

        return sorted(groups, key=group_priority, reverse=True)

    def _improve_shifts_with_insertions(
        self,
        layout,
        shifts,
        simulator,
        *,
        mode: str = "normal",
    ):
        """和父类同构，但只搜索图中排序最高的一组插入组。

        为什么需要这个预算：``_best_insertion_attempt`` 每次都要跑完整的
        ``simulator.evaluate``（含无人机分配），实测占 ``optimize()`` 的 54%~88%。
        而真正提分的尝试不到 2%。所以这里先按图先验排序，再在预算内做多轮，
        剩下的组记为 ``not_searched_graph_budget``，不会静默丢弃。
        """
        config = self.graph_config
        if not config.limit_insertion_search:
            return super()._improve_shifts_with_insertions(
                layout, shifts, simulator, mode=mode
            )

        all_groups = self._diagnostic_shift_insertion_groups()
        groups = self._available_shift_insertion_groups(all_groups)
        ordered = self._order_insertion_groups(groups)
        budget = max(1, min(len(ordered), config.insertion_group_budget))
        searched = ordered[:budget]
        deferred = ordered[budget:]

        audit: dict[str, Any] = {
            "source": "graph_prioritized_insertion_search",
            "optimizerModelVersion": GRAPH_OPTIMIZER_MODEL_VERSION,
            "maxGroups": self.diagnostic_insertion_group_limit(),
            "jointProductionCandidateLimit": self.joint_production_candidate_limit(),
            "operatorAnchorPreferenceCount": len(self.operator_anchor_preference),
            "availableGroupCount": len(groups),
            "searchBudget": {
                "groupBudget": budget,
                "searchedGroupCount": len(searched),
                "deferredGroupCount": len(deferred),
                "maxPasses": config.insertion_max_passes,
            },
            "accepted": [],
            "skipped": [
                *self._unsearched_faction_dependency_records(),
                *self._unavailable_shift_insertion_records(all_groups, groups),
                *self._group_limit_insertion_records(all_groups, groups),
                *[
                    {
                        "groupId": group.id,
                        "specKey": insertion_spec_key(group.specs),
                        "status": "not_searched_graph_budget",
                        "operators": [spec.operator_name for spec in group.specs],
                    }
                    for group in deferred
                ],
            ],
        }
        if not groups:
            audit["summary"] = insertion_search_summary(audit)
            audit["localQualityAudit"] = local_quality_audit(audit)
            return shifts, audit

        best_shifts = shifts
        best_report = simulator.evaluate(best_shifts)
        best_score = insertion_objective_score(mode, best_report)
        max_passes = max(1, min(config.insertion_max_passes, budget))
        for pass_index in range(max_passes):
            accepted_this_pass = 0
            for group in searched:
                result, candidate, candidate_report = self._best_insertion_attempt(
                    layout,
                    best_shifts,
                    simulator,
                    group,
                    best_score,
                    best_report,
                    mode=mode,
                )
                if result["status"] != "improves" or candidate is None:
                    continue
                score = float(result.get("candidateScore") or best_score)
                if score <= best_score + 0.001:
                    continue
                accepted = dict(result)
                accepted["status"] = "accepted"
                accepted["pass"] = pass_index + 1
                audit["accepted"].append(accepted)
                best_shifts = candidate
                best_report = (
                    candidate_report
                    if candidate_report is not None
                    else simulator.evaluate(best_shifts)
                )
                best_score = score
                accepted_this_pass += 1
            if accepted_this_pass == 0:
                break

        accepted_keys = {
            (item.get("groupId"), item.get("specKey")) for item in audit["accepted"]
        }
        skipped: list[dict[str, Any]] = list(audit.get("skipped") or [])
        # 诊断复核只覆盖真正搜索过的组，避免为“没搜的组”再全量跑一遍。
        for group in searched:
            group_key = (group.id, insertion_spec_key(group.specs))
            if group_key in accepted_keys:
                continue
            result, _, _ = self._best_insertion_attempt(
                layout,
                best_shifts,
                simulator,
                group,
                best_score,
                best_report,
                mode=mode,
            )
            if result["status"] == "improves":
                result = {**result, "status": "remaining_improvement_after_pass_limit"}
            skipped.append(result)
        audit["skipped"] = skipped
        audit["summary"] = insertion_search_summary(audit)
        audit["localQualityAudit"] = local_quality_audit(audit)
        return best_shifts, audit

    # ------------------------------------------------------------------
    # 纯赤金平衡：只接受“不掉分”的重复次数
    # ------------------------------------------------------------------
    def _apply_pure_gold_drone_cycle_repeats(
        self,
        shifts,
        simulator,
        *,
        current_report,
        drone_policy: str,
        pure_gold_target: float,
        pure_gold_tolerance: float,
        max_drone_cycle_repeats: int,
    ):
        """在赤金平衡容差内选择**综合分最高**的循环重复次数。

        父类的规则是“第一个进入容差区间的重复次数就采用”，不比较分数。
        这会在容差边界上造成巨大方差：本仓库实测 342/极限搓玉 场景里，
        重复 2 次只把赤金日差从 -1.58 改善到 -0.11，却把综合分从 388.7 拉到 273.2
        （-30%），并让班次从 3 个变 6 个、整体耗时翻倍。

        新规则：把所有重复次数都算一遍，在**满足容差**的候选中取分数最高者；
        若没有任何候选满足容差，则沿用父类的“误差最小、分数次之”规则。
        可以通过 ``graph_config.prefer_score_within_tolerance`` 关掉，回到父类行为。
        """
        if not self.graph_config.prefer_score_within_tolerance:
            return super()._apply_pure_gold_drone_cycle_repeats(
                shifts,
                simulator,
                current_report=current_report,
                drone_policy=drone_policy,
                pure_gold_target=pure_gold_target,
                pure_gold_tolerance=pure_gold_tolerance,
                max_drone_cycle_repeats=max_drone_cycle_repeats,
            )

        max_repeats = max(1, int(max_drone_cycle_repeats))
        base_quality = pure_gold_balance_quality(
            current_report.dailyExpected,
            target=pure_gold_target,
            tolerance=pure_gold_tolerance,
        )
        policy: dict[str, Any] = {
            "version": 1,
            "scope": "drone_allocation_only",
            "selectionRule": "max_score_within_tolerance",
            "optimizerModelVersion": GRAPH_OPTIMIZER_MODEL_VERSION,
            "targetPerDay": round(float(pure_gold_target), 3),
            "tolerancePerDay": round(float(pure_gold_tolerance), 3),
            "externalPureGoldAssumptionPerDay": round(-float(pure_gold_target), 3),
            "externalPureGoldAssumptionSource": (
                "Conservative stable-income assumption: current daily/weekly missions and "
                "permanent shops are not modeled as reliable Pure Gold income."
            ),
            "maxDroneCycleRepeats": max_repeats,
            "baseShiftCount": len(shifts),
            "initialPureGoldDelta": base_quality["pureGoldDelta"],
            "initialDeltaFromTarget": base_quality["deltaFromTarget"],
        }
        if simulator.drone_policy != "auto" or str(drone_policy).lower().replace("_", "-") != "auto":
            policy.update(
                {
                    "status": "not_applicable_drone_policy",
                    "reason": "Pure Gold balancing is applied only by --drone-policy auto.",
                    "repeatCount": 1,
                    "cycleRepeated": False,
                    "finalPureGoldDelta": base_quality["pureGoldDelta"],
                    "finalDeltaFromTarget": base_quality["deltaFromTarget"],
                    "finalAbsDeltaFromTarget": base_quality["absDeltaFromTarget"],
                    "withinTolerance": base_quality["withinTolerance"],
                }
            )
            return shifts, current_report, policy

        evaluated: list[dict[str, Any]] = []
        best_within: tuple[float, int, list, Any] | None = None
        best_any: tuple[float, float, int, list, Any] | None = None
        # 心情周期硬门槛要求整个循环不超过 7 天（`morale.MAX_CYCLE_HOURS`）。
        # 父类靠「第一个达标的重复次数就 break」很少撞上，而这里会扫完所有重复次数，
        # 所以必须自己把超限的候选挡掉，否则会产出过不了门槛的排班。
        base_cycle_hours = sum(float(shift.duration_hours) for shift in shifts)
        for repeat_count in range(1, max_repeats + 1):
            if base_cycle_hours * repeat_count > MAX_CYCLE_HOURS + 1e-6:
                policy["stoppedByMaxCycleHours"] = repeat_count
                break
            candidate_shifts = (
                shifts if repeat_count == 1 else repeat_shift_cycle(shifts, repeat_count)
            )
            candidate_report = (
                current_report if repeat_count == 1 else simulator.evaluate(candidate_shifts)
            )
            quality = pure_gold_balance_quality(
                candidate_report.dailyExpected,
                target=pure_gold_target,
                tolerance=pure_gold_tolerance,
            )
            error = float(quality["absDeltaFromTarget"])
            evaluated.append(
                {
                    "repeatCount": repeat_count,
                    "shiftCount": len(candidate_shifts),
                    "pureGoldDelta": quality["pureGoldDelta"],
                    "deltaFromTarget": quality["deltaFromTarget"],
                    "absDeltaFromTarget": quality["absDeltaFromTarget"],
                    "withinTolerance": quality["withinTolerance"],
                    "score": round(candidate_report.score, 3),
                    "droneUsed": round(candidate_report.dailyExpected.droneUsed, 3),
                    "droneCount": round(candidate_report.dailyExpected.droneCount, 3),
                }
            )
            if quality["withinTolerance"]:
                candidate_key = (
                    float(candidate_report.score),
                    -error,
                    repeat_count,
                )
                if best_within is None or candidate_key > best_within[0]:
                    best_within = (
                        candidate_key,
                        repeat_count,
                        candidate_shifts,
                        candidate_report,
                    )
            any_key = (-error, float(candidate_report.score), -repeat_count)
            if best_any is None or any_key > best_any[0]:
                best_any = (
                    any_key,
                    error,
                    repeat_count,
                    candidate_shifts,
                    candidate_report,
                )

        if best_within is not None:
            _, best_repeat, best_shifts, best_report = best_within
        else:
            assert best_any is not None
            _, _error, best_repeat, best_shifts, best_report = best_any

        final_quality = pure_gold_balance_quality(
            best_report.dailyExpected,
            target=pure_gold_target,
            tolerance=pure_gold_tolerance,
        )
        within = bool(final_quality["withinTolerance"])
        initial_error = float(base_quality["absDeltaFromTarget"])
        policy.update(
            {
                "status": "within_tolerance" if within else "unable_to_balance_within_repeat_limit",
                "repeatCount": best_repeat,
                "cycleRepeated": best_repeat > 1,
                "finalShiftCount": len(best_shifts),
                "finalPureGoldDelta": final_quality["pureGoldDelta"],
                "finalDeltaFromTarget": final_quality["deltaFromTarget"],
                "finalAbsDeltaFromTarget": final_quality["absDeltaFromTarget"],
                "withinTolerance": within,
                "improvedByRepeating": float(final_quality["absDeltaFromTarget"]) < initial_error - 0.001,
                "candidateRepeats": evaluated,
            }
        )
        return best_shifts, best_report, policy

    # ------------------------------------------------------------------
    # 运行画像
    # ------------------------------------------------------------------
    def _runtime_profile(self) -> dict[str, Any]:
        profile = super()._runtime_profile()
        if self.graph_stats:
            profile["graph"] = {**self.graph_stats, "cache": self._cache.profile()}
        elif self._cache.active:
            profile["graph"] = {"cache": self._cache.profile()}
        return profile

    def candidate_pool_audit(self, shifts) -> dict[str, Any]:
        return self._candidate_pool_audit(shifts)
