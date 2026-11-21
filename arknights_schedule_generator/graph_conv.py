"""固定超参图卷积（GCN）+ 闭式读出层。

设计取舍
--------
旧优化器用“全组合枚举 + 逐个调用生产模拟器”来决定一个房间该放谁：
一个房间 32 个候选干员就要评估 ``C(32,3)=4960`` 次模拟器。本模块用一张静态
干员关系图把这个过程换成一次前向传播 + 一个小的线性读出层。

三件事让它可以既“是神经网络”又保持轻量、确定、无外部依赖：

1. **图卷积的权重是固定的**（这里是确定性随机投影，seed 固定）。
   所以前向传播完全可复现，不需要训练循环、不需要 GPU、也不引入 torch/numpy 依赖。
   这等价于一个 Gaussian random feature 网络；真正被“学”的只有读出层。
2. **读出层用闭式岭回归求解**（正规方程 + 高斯消元），
   标签来自现有 ``ProductionSimulator``——也就是说新模型蒸馏旧模型的评分口径，
   而不是重新定义收益。求解一次只需毫秒级。
3. **聚合按关系类型分开**：阵营（faction）用超节点均值，显式羁绊（named）与
   组合机制（combo）用稀疏边。阵营完全子图不做 O(n²) 展开，避免 94 人的
   ``rhodes`` 把整张图变成全连接。

读出层的形式是“一元 + 二元”的集合函数：

.. math::

    \\hat{y}(S) = b + \\sum_{i \\in S} w_u \\cdot e_i
                + \\sum_{i < j \\in S} w_p \\cdot (e_i \\odot e_j)

这正是房内协同（羁绊/增益）的自然表达：单人贡献是一元的，
“两人同房间才生效”的增益是二元的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations as _combinations
from typing import Any, Callable, Iterable, Mapping, Sequence

from .graph_bonds import BondBasis
from .graph_features import (
    FEATURE_NAMES,
    RELATIONS,
    ROOM_FEATURE_NAMES,
    ROOM_RELATIONS,
    ROOM_TYPES,
    StaticOperatorGraph,
    StaticRoomGraph,
)

try:  # numpy 是可选加速路径；工具本身不把它当依赖。
    import numpy as _np
except ImportError:  # pragma: no cover - 环境相关
    _np = None

__all__ = [
    "EncoderConfig",
    "GraphEncoder",
    "RoomValueModel",
    "RoomQuery",
    "RoomCondition",
    "RoomEncoderConfig",
    "RoomEncoder",
    "RoomAffinityModel",
    "AffinityFitRoom",
    "ROOM_QUERY_FEATURE_NAMES",
    "fit_room_value_model",
    "fit_room_affinity_model",
    "rank_combinations",
    "numpy_available",
]


def numpy_available() -> bool:
    """numpy 加速路径是否可用（不影响功能，只影响速度）。"""
    return _np is not None


# 房间类型 → 用于房间查询向量的下标（顺序必须与 ROOM_TYPES 一致）。
_ROOM_INDEX = {room_type: index for index, room_type in enumerate(ROOM_TYPES)}
_TARGET_INDEX = {
    target: index
    for index, target in enumerate(("O_GOLD", "O_DIAMOND", "F_GOLD", "F_EXP", "F_DIAMOND"))
}

# 需要做“饱和非线性”增广的特征：产能类效率天然会被订单上限/仓位截断，
# 线性读出层需要平方项才能表达这种凹性。
_NONLINEAR_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"room:{room_type}" for room_type in ROOM_TYPES
)


@dataclass(frozen=True)
class EncoderConfig:
    """图卷积的固定超参。没有一项是通过训练得到的。"""

    hidden_dim: int = 24
    layers: int = 2
    self_weight: float = 0.65
    relation_weights: Mapping[str, float] = field(
        default_factory=lambda: {"faction": 0.55, "named": 1.30, "combo": 0.45}
    )
    activation_slope: float = 0.15
    seed: int = 20260624
    nonzeros_per_channel: int = 3

    def resolve_relation_weight(self, relation: str) -> float:
        return float(self.relation_weights.get(relation, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hiddenDim": self.hidden_dim,
            "layers": self.layers,
            "selfWeight": self.self_weight,
            "relationWeights": {k: float(v) for k, v in self.relation_weights.items()},
            "activationSlope": self.activation_slope,
            "seed": self.seed,
            "nonzerosPerChannel": self.nonzeros_per_channel,
        }


class _RandomStream:
    """确定性伪随机流（SplitMix64），保证跨平台、跨 Python 版本一致。"""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def next_unit(self) -> float:
        return (self.next_u64() >> 11) / float(1 << 53)

    def next_signed(self) -> float:
        return self.next_unit() * 2.0 - 1.0

    def next_int(self, bound: int) -> int:
        if bound <= 0:
            return 0
        return self.next_u64() % bound


@dataclass(frozen=True)
class _SparseProjection:
    """稀疏随机投影：每个输出通道只连 ``nonzeros`` 个输入通道。"""

    indices: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[float, ...], ...]

    def apply(self, vector: Sequence[float]) -> list[float]:
        out: list[float] = []
        append = out.append
        for idx_row, weight_row in zip(self.indices, self.weights):
            total = 0.0
            for idx, weight in zip(idx_row, weight_row):
                total += vector[idx] * weight
            append(total)
        return out

    def apply_sum(self, vectors: Iterable[Sequence[float]]) -> list[float]:
        """对若干向量先分别投影再求和；等价于对和投影（线性），但可省一次中间分配。"""
        out: list[float] | None = None
        for vector in vectors:
            projected = self.apply(vector)
            if out is None:
                out = projected
            else:
                for index, value in enumerate(projected):
                    out[index] += value
        return out if out is not None else []


def _build_sparse_projection(
    stream: _RandomStream,
    dim_in: int,
    dim_out: int,
    nonzeros: int,
    scale: float,
) -> _SparseProjection:
    nonzeros = max(1, min(nonzeros, dim_in))
    indices: list[tuple[int, ...]] = []
    weights: list[tuple[float, ...]] = []
    gain = scale / math.sqrt(nonzeros)
    for _ in range(dim_out):
        idx_row = tuple(stream.next_int(dim_in) for _ in range(nonzeros))
        weight_row = tuple(stream.next_signed() * gain for _ in range(nonzeros))
        indices.append(idx_row)
        weights.append(weight_row)
    return _SparseProjection(tuple(indices), tuple(weights))


class GraphEncoder:
    """在 :class:`StaticOperatorGraph` 上做固定超参的消息传递。

    传播一层的形状是::

        h'[i] = act( ws * W_self h[i]
                     + sum_r wr * W_r ( A_r h )[i] )

    其中 ``A_faction`` 用阵营超节点均值实现（完全子图的等价 O(n) 形式），
    ``A_named`` / ``A_combo`` 用行归一化的稀疏边。

    每层输出拼接后得到 ``(layers + 1) * hidden_dim`` 维的节点嵌入，
    这是“节点自身特征 + 多层邻域上下文”的定长表示。
    """

    def __init__(
        self,
        graph: StaticOperatorGraph,
        config: EncoderConfig | None = None,
    ) -> None:
        self.graph = graph
        self.config = config or EncoderConfig()
        self._feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
        self._nonlinear_indices = tuple(
            self._feature_index[name]
            for name in _NONLINEAR_FEATURE_NAMES
            if name in self._feature_index
        )
        self._input_dim = len(FEATURE_NAMES) + len(self._nonlinear_indices)
        self._embeddings: tuple[tuple[float, ...], ...] | None = None
        self._build_projections()

    # ------------------------------------------------------------------
    @property
    def embedding_dim(self) -> int:
        """嵌入维度 = 原始特征 + 每层隐藏表示（第一层拼接的是输入特征本身）。"""
        return self._input_dim + self.config.layers * self.config.hidden_dim

    def _build_projections(self) -> None:
        config = self.config
        stream = _RandomStream(config.seed)
        dim_h = config.hidden_dim
        self._self_projections: list[_SparseProjection] = []
        self._relation_projections: dict[str, list[_SparseProjection]] = {
            relation: [] for relation in RELATIONS
        }
        for layer in range(config.layers):
            dim_in = self._input_dim if layer == 0 else dim_h
            self._self_projections.append(
                _build_sparse_projection(
                    stream,
                    dim_in,
                    dim_h,
                    config.nonzeros_per_channel,
                    scale=1.0 / math.sqrt(dim_h),
                )
            )
            for relation in RELATIONS:
                self._relation_projections[relation].append(
                    _build_sparse_projection(
                        stream,
                        dim_in,
                        dim_h,
                        config.nonzeros_per_channel,
                        scale=1.0 / math.sqrt(dim_h),
                    )
                )

    # ------------------------------------------------------------------
    def _input_matrix(self) -> list[list[float]]:
        rows: list[list[float]] = []
        for node in self.graph.nodes:
            base = list(node.features)
            extra = [base[index] ** 2 for index in self._nonlinear_indices]
            rows.append(base + extra)
        return rows

    def _faction_membership(self) -> list[list[int]]:
        """每个节点所属的阵营组下标（仅使用参与建边的组）。"""
        groups = list(self.graph.faction_groups.items())
        group_index = {tag: index for index, (tag, _) in enumerate(groups)}
        self._faction_groups = [members for _, members in groups]
        membership: list[list[int]] = [[] for _ in range(self.graph.size)]
        for tag_index, (_, members) in enumerate(groups):
            for name in members:
                node_index = self.graph.index_by_name.get(name)
                if node_index is not None:
                    membership[node_index].append(tag_index)
        self._faction_group_index = group_index
        return membership

    def _relation_adjacency(self) -> dict[str, list[list[tuple[int, float]]]]:
        """行归一化邻接表（不含 faction：它走超节点均值）。"""
        raw: dict[str, list[list[tuple[int, float]]]] = {
            relation: [[] for _ in range(self.graph.size)]
            for relation in RELATIONS
            if relation != "faction"
        }
        totals: dict[str, list[float]] = {
            relation: [0.0] * self.graph.size for relation in raw
        }
        for edge in self.graph.edges:
            bucket = raw.get(edge.relation)
            if bucket is None:
                continue
            bucket[edge.src].append((edge.dst, edge.weight))
            totals[edge.relation][edge.src] += edge.weight
        normalized: dict[str, list[list[tuple[int, float]]]] = {}
        for relation, bucket in raw.items():
            norm_bucket: list[list[tuple[int, float]]] = []
            for node_index, neighbours in enumerate(bucket):
                total = totals[relation][node_index]
                if total <= 0:
                    norm_bucket.append([])
                    continue
                norm_bucket.append([(dst, weight / total) for dst, weight in neighbours])
            normalized[relation] = norm_bucket
        return normalized

    # ------------------------------------------------------------------
    def embeddings(self) -> list[tuple[float, ...]]:
        """所有节点的定长嵌入（原始特征 ⊕ 每层隐表示）。

        numpy 可用时走批量矩阵路径（同一套权重，数值等价），
        否则退回到纯 Python 稀疏传播。
        """
        if self._embeddings is not None:
            return list(self._embeddings)
        if not self.graph.size:
            self._embeddings = ()
            return []
        if _np is not None:
            self._embeddings = self._embeddings_numpy()
        else:
            self._embeddings = self._embeddings_python()
        return list(self._embeddings)

    def _dense_projection(self, projection: _SparseProjection, dim_in: int) -> Any:
        """把稀疏随机投影物化成稠密矩阵（numpy 路径用，一次构建多次复用）。"""
        assert _np is not None
        matrix = _np.zeros((dim_in, len(projection.indices)), dtype=float)
        for output_channel, (idx_row, weight_row) in enumerate(
            zip(projection.indices, projection.weights)
        ):
            for idx, weight in zip(idx_row, weight_row):
                matrix[idx, output_channel] += weight
        return matrix

    def _numpy_operators(self) -> dict[str, Any]:
        cached = getattr(self, "_operators_numpy", None)
        if cached is not None:
            return cached
        assert _np is not None
        size = self.graph.size
        membership = self._faction_membership()
        adjacency = self._relation_adjacency()

        # faction：阵营超节点 → 节点均值，写成一张稠密矩阵
        faction_matrix = _np.zeros((size, size), dtype=float)
        group_members: list[list[int]] = []
        for members in getattr(self, "_faction_groups", []):
            indices = [
                self.graph.index_by_name[name]
                for name in members
                if name in self.graph.index_by_name
            ]
            group_members.append(indices)
        tag_of_node = membership
        for node_index, tags in enumerate(tag_of_node):
            if not tags:
                continue
            scale = 1.0 / len(tags)
            for tag_index in tags:
                indices = group_members[tag_index]
                if not indices:
                    continue
                weight = scale / len(indices)
                for other in indices:
                    faction_matrix[node_index, other] += weight

        matrices: dict[str, Any] = {"faction": faction_matrix}
        for relation in RELATIONS:
            if relation == "faction":
                continue
            matrix = _np.zeros((size, size), dtype=float)
            for node_index, neighbours in enumerate(adjacency.get(relation) or []):
                for dst, weight in neighbours:
                    matrix[node_index, dst] += weight
            matrices[relation] = matrix

        operators = {
            "relations": matrices,
            "self": [
                self._dense_projection(projection, self._input_dim if layer == 0 else self.config.hidden_dim)
                for layer, projection in enumerate(self._self_projections)
            ],
            "relation": {
                relation: [
                    self._dense_projection(
                        self._relation_projections[relation][layer],
                        self._input_dim if layer == 0 else self.config.hidden_dim,
                    )
                    for layer in range(self.config.layers)
                ]
                for relation in RELATIONS
            },
        }
        self._operators_numpy = operators
        return operators

    def _embeddings_numpy(self) -> tuple[tuple[float, ...], ...]:
        assert _np is not None
        config = self.config
        operators = self._numpy_operators()
        height = _np.asarray(self._input_matrix(), dtype=float)
        stack: list[Any] = [height]
        for layer in range(config.layers):
            self_weight = operators["self"][layer]
            activated = (height @ self_weight) * config.self_weight
            for relation in RELATIONS:
                weight = config.resolve_relation_weight(relation)
                if not weight:
                    continue
                aggregated = operators["relations"][relation] @ height
                activated = activated + (aggregated @ operators["relation"][relation][layer]) * weight
            height = _np.where(activated >= 0, activated, activated * config.activation_slope)
            stack.append(height)
        combined = _np.concatenate(stack, axis=1)
        return tuple(tuple(float(value) for value in row) for row in combined)

    def _embeddings_python(self) -> tuple[tuple[float, ...], ...]:
        config = self.config
        dim_h = config.hidden_dim
        size = self.graph.size
        height = self._input_matrix() if size else []
        membership = self._faction_membership()
        adjacency = self._relation_adjacency()
        faction_weight = config.resolve_relation_weight("faction")

        layer_stack: list[list[tuple[float, ...]]] = [
            [tuple(row) for row in height]
        ]

        for layer in range(config.layers):
            self_projection = self._self_projections[layer]
            relation_projections = {
                relation: self._relation_projections[relation][layer]
                for relation in RELATIONS
            }
            dim_in = len(height[0]) if height else self._input_dim

            # faction：先算每个阵营组的均值，再对节点取所属组的均值
            group_means: list[list[float]] = []
            for members in getattr(self, "_faction_groups", []):
                acc = [0.0] * dim_in
                used = 0
                for name in members:
                    node_index = self.graph.index_by_name.get(name)
                    if node_index is None:
                        continue
                    vector = height[node_index]
                    for channel in range(dim_in):
                        acc[channel] += vector[channel]
                    used += 1
                if used:
                    group_means.append([value / used for value in acc])
                else:
                    group_means.append([0.0] * dim_in)

            faction_projected: list[list[float]] = []
            if group_means:
                for members in membership:
                    if not members:
                        faction_projected.append([0.0] * dim_in)
                        continue
                    acc = [0.0] * dim_in
                    for tag_index in members:
                        mean = group_means[tag_index]
                        for channel in range(dim_in):
                            acc[channel] += mean[channel]
                    scale = 1.0 / len(members)
                    faction_projected.append([value * scale for value in acc])
            else:
                faction_projected = [[0.0] * dim_in for _ in range(size)]

            relation_aggregates: dict[str, list[list[float]]] = {}
            for relation in RELATIONS:
                if relation == "faction":
                    continue
                bucket = adjacency.get(relation) or []
                aggregated: list[list[float]] = []
                for node_index in range(size):
                    acc = [0.0] * dim_in
                    for dst, weight in bucket[node_index]:
                        vector = height[dst]
                        for channel in range(dim_in):
                            acc[channel] += vector[channel] * weight
                    aggregated.append(acc)
                relation_aggregates[relation] = aggregated

            next_height: list[tuple[float, ...]] = []
            slope = config.activation_slope
            for node_index in range(size):
                activated = self_projection.apply(height[node_index])
                for channel in range(dim_h):
                    activated[channel] *= config.self_weight
                if faction_weight:
                    projected = relation_projections["faction"].apply(
                        faction_projected[node_index]
                    )
                    for channel in range(dim_h):
                        activated[channel] += projected[channel] * faction_weight
                for relation in RELATIONS:
                    if relation == "faction":
                        continue
                    weight = config.resolve_relation_weight(relation)
                    if not weight:
                        continue
                    projected = relation_projections[relation].apply(
                        relation_aggregates[relation][node_index]
                    )
                    for channel in range(dim_h):
                        activated[channel] += projected[channel] * weight
                next_height.append(
                    tuple(
                        value if value >= 0 else value * slope
                        for value in activated
                    )
                )
            height = [list(row) for row in next_height]
            layer_stack.append(next_height)

        return tuple(
            tuple(value for row in layer_stack for value in row[node_index])
            for node_index in range(size)
        )

    def embedding(self, name: str) -> tuple[float, ...] | None:
        index = self.graph.index_by_name.get(name)
        if index is None:
            return None
        return self.embeddings()[index]

    def profile(self) -> dict[str, Any]:
        return {
            "encoder": self.config.to_dict(),
            "nodeCount": self.graph.size,
            "edgeCount": len(self.graph.edges),
            "embeddingDim": self.embedding_dim,
            "inputDim": self._input_dim,
            "graphSignature": self.graph.meta.get("signature"),
        }


@dataclass(frozen=True)
class RoomQuery:
    """房间侧的查询条件。用于把读出层绑定到“这个房间/这个产物”。"""

    room_type: str
    target: str | None = None
    room_level: int = 3
    slots: int = 3

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.room_type, self.target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roomType": self.room_type,
            "target": self.target,
            "roomLevel": self.room_level,
            "slots": self.slots,
        }


def _extra_features(query: RoomQuery, count: int) -> list[float]:
    level_norm = max(0.0, min(1.0, query.room_level / 3.0))
    slots_norm = max(0.0, min(1.0, query.slots / 5.0))
    return [
        float(count),
        float(count * count),
        count * level_norm,
        count * slots_norm,
    ]


@dataclass
class RoomValueModel:
    """房间组合价值模型：一元项 + 二元协同项的集合函数。

    读出层的基分成两部分：

    * **图嵌入基**：``(layers+1)`` 层图卷积得到的 47 维向量，表达结构上下文；
    * **显式羁绊基**（``bond_basis``）：阵营人数与点名羁绊指示函数，
      让「每多一名 glasgow 干员值多少分」「一对有羁绊的干员同房值多少分」
      变成可以直接读出来的系数，而不是埋在嵌入里。
    """

    query: RoomQuery
    embedding_dim: int
    unary: list[float]
    pair: list[float]
    extra: list[float]
    bias: float
    bond_basis: BondBasis | None = None
    fit_quality: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    @property
    def query_key(self) -> tuple[str, str | None]:
        return self.query.key

    @property
    def bond_unary_width(self) -> int:
        return 0 if self.bond_basis is None else self.bond_basis.unary_width

    @property
    def bond_pair_width(self) -> int:
        return 0 if self.bond_basis is None else self.bond_basis.pair_width

    # ------------------------------------------------------------------
    def extra_features(self, count: int) -> list[float]:
        return _extra_features(self.query, count)

    def unary_bond(self, name: str) -> tuple[float, ...]:
        if self.bond_basis is None:
            return ()
        return self.bond_basis.unary_features(name)

    def pair_bond(self, left: str, right: str) -> tuple[float, ...]:
        if self.bond_basis is None:
            return ()
        return self.bond_basis.pair_features(left, right)

    @property
    def unary_bond_weights(self) -> Sequence[float]:
        dim = self.embedding_dim
        return self.unary[dim : dim + self.bond_unary_width]

    @property
    def pair_bond_weights(self) -> Sequence[float]:
        return self.pair[self.embedding_dim : self.embedding_dim + self.bond_pair_width]

    def _features(self, names: Sequence[str], vectors: Sequence[Sequence[float]]) -> list[float]:
        dim = self.embedding_dim
        unary = [0.0] * dim
        for vector in vectors:
            for channel in range(dim):
                unary[channel] += vector[channel]
        pair = [0.0] * dim
        count = len(vectors)
        for left in range(count):
            left_vector = vectors[left]
            for right in range(left + 1, count):
                right_vector = vectors[right]
                for channel in range(dim):
                    pair[channel] += left_vector[channel] * right_vector[channel]
        features = [*unary]
        if self.bond_basis is not None:
            bond_unary = [0.0] * self.bond_unary_width
            for name in names:
                for index, value in enumerate(self.bond_basis.unary_features(name)):
                    bond_unary[index] += value
            features.extend(bond_unary)
        features.extend(pair)
        if self.bond_basis is not None:
            bond_pair = [0.0] * self.bond_pair_width
            for left in range(count):
                for right in range(left + 1, count):
                    for index, value in enumerate(
                        self.bond_basis.pair_features(names[left], names[right])
                    ):
                        bond_pair[index] += value
            features.extend(bond_pair)
        features.extend(self.extra_features(count))
        return features

    def score(self, names: Sequence[str], embeddings: Mapping[str, Sequence[float]]) -> float:
        vectors = [embeddings[name] for name in names if name in embeddings]
        if len(vectors) != len(names):
            raise KeyError(f"missing embedding for one of {list(names)}")
        return self.score_vectors(vectors, names=names)

    def score_vectors(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        names: Sequence[str] | None = None,
    ) -> float:
        if not vectors:
            return self.bias
        if self.bond_basis is not None and names is None:
            raise ValueError("this model has an explicit bond basis; pass names=... too")
        features = self._features(names or [""] * len(vectors), vectors)
        total = self.bias
        for weight, value in zip([*self.unary, *self.pair, *self.extra], features):
            total += weight * value
        return total

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "roomType": self.query_key[0],
            "target": self.query_key[1],
            "embeddingDim": self.embedding_dim,
            "parameterCount": len(self.unary) + len(self.pair) + len(self.extra) + 1,
            "bondBasisWidth": self.bond_unary_width + self.bond_pair_width,
            **self.fit_quality,
            "training": self.training,
        }
        if self.bond_basis is not None:
            payload["bonds"] = self.bond_basis.describe(self.unary, self.pair)
        return payload

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """序列化（用于磁盘缓存）。数值不截断，保证复用后预测完全一致。"""
        return {
            "query": self.query.to_dict(),
            "embeddingDim": self.embedding_dim,
            "unary": list(self.unary),
            "pair": list(self.pair),
            "extra": list(self.extra),
            "bias": self.bias,
            "bondBasis": None if self.bond_basis is None else self.bond_basis.to_dict(),
            "fitQuality": dict(self.fit_quality),
            "training": dict(self.training),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoomValueModel":
        query_payload = payload["query"]
        basis_payload = payload.get("bondBasis")
        return cls(
            query=RoomQuery(
                room_type=query_payload["roomType"],
                target=query_payload.get("target"),
                room_level=int(query_payload.get("roomLevel", 3)),
                slots=int(query_payload.get("slots", 3)),
            ),
            embedding_dim=int(payload["embeddingDim"]),
            unary=[float(value) for value in payload["unary"]],
            pair=[float(value) for value in payload["pair"]],
            extra=[float(value) for value in payload["extra"]],
            bias=float(payload.get("bias", 0.0)),
            bond_basis=None if basis_payload is None else BondBasis.from_dict(basis_payload),
            fit_quality=dict(payload.get("fitQuality") or {}),
            training=dict(payload.get("training") or {}),
        )


def _solve_ridge(rows: Sequence[Sequence[float]], targets: Sequence[float], ridge: float) -> list[float]:
    """正规方程求解；有 numpy 走向量化解，否则用纯 Python 高斯-约当消元。"""
    # 用 ``len(rows)`` 而不是 ``if rows``：共享亲和度矩阵会直接传 numpy 数组进来，
    # 数组的真值判断是歧义的。
    n = len(rows[0]) if len(rows) else 0
    if n == 0:
        return []
    if _np is not None:
        matrix = _np.asarray(rows, dtype=float)
        vector = _np.asarray(targets, dtype=float)
        gram = matrix.T @ matrix
        gram[_np.diag_indices_from(gram)] += ridge
        rhs = matrix.T @ vector
        try:
            solution = _np.linalg.solve(gram, rhs)
        except _np.linalg.LinAlgError:  # 奇异时退回最小二乘
            solution = _np.linalg.lstsq(gram, rhs, rcond=None)[0]
        return [float(value) for value in solution]

    augmented = [[0.0] * (n + 1) for _ in range(n)]
    for row, target in zip(rows, targets):
        for i in range(n):
            value = row[i]
            if value == 0.0:
                continue
            row_i = augmented[i]
            for j in range(n):
                row_i[j] += value * row[j]
            row_i[n] += value * target
    for i in range(n):
        augmented[i][i] += ridge
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            continue
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_row = augmented[column]
        pivot_value = pivot_row[column]
        for j in range(column, n + 1):
            pivot_row[j] /= pivot_value
        for r in range(n):
            if r == column:
                continue
            factor = augmented[r][column]
            if factor == 0.0:
                continue
            row_r = augmented[r]
            for j in range(column, n + 1):
                row_r[j] -= factor * pivot_row[j]
    return [augmented[i][n] for i in range(n)]


def _spearman(predicted: Sequence[float], actual: Sequence[float]) -> float:
    n = len(predicted)
    if n < 3:
        return 0.0

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda index: values[index])
        result = [0.0] * n
        position = 0
        while position < n:
            end = position
            while end + 1 < n and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    rank_p = ranks(predicted)
    rank_a = ranks(actual)
    mean_p = sum(rank_p) / n
    mean_a = sum(rank_a) / n
    numerator = sum((rank_p[i] - mean_p) * (rank_a[i] - mean_a) for i in range(n))
    denominator = math.sqrt(
        sum((rank_p[i] - mean_p) ** 2 for i in range(n))
        * sum((rank_a[i] - mean_a) ** 2 for i in range(n))
    )
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _r2(predicted: Sequence[float], actual: Sequence[float]) -> float:
    mean = sum(actual) / max(1, len(actual))
    ss_tot = sum((value - mean) ** 2 for value in actual)
    if ss_tot <= 0:
        return 0.0
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 1.0 - ss_res / ss_tot


def _mae(predicted: Sequence[float], actual: Sequence[float]) -> float:
    if not actual:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def fit_room_value_model(
    encoder: GraphEncoder,
    query: RoomQuery,
    candidates: Sequence[str],
    oracle: Callable[[tuple[str, ...]], float],
    *,
    capacity: int,
    samples: int = 320,
    seed: int = 7,
    ridge: float = 1.0,
    holdout_ratio: float = 0.25,
    min_combo_size: int = 1,
    bond_basis: BondBasis | None = None,
    bond_ridge_scale: float = 4.0,
) -> RoomValueModel | None:
    """用生产模拟器当“老师”，闭式拟合房间组合价值模型。

    ``oracle`` 接收一个干员名字元组，返回该组合在模拟器下的房间分数。
    采样策略刻意混合 1 人 / 2 人 / 满编组合：一元项需要单人样本，
    二元项需要成对样本，满编样本给出真实目标。

    ``bond_basis`` 给出显式羁绊基（阵营人数 + 点名羁绊指示），
    它们作为额外线性特征参与闭式求解，系数可直接读出。
    """
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) < 1:
        return None
    embeddings = encoder.embeddings()
    embedding_by_name = {
        node.name: embeddings[index] for index, node in enumerate(encoder.graph.nodes)
    }
    available = [name for name in unique_candidates if name in embedding_by_name]
    if not available:
        return None
    if len(embedding_by_name[available[0]]) != encoder.embedding_dim:
        raise AssertionError(
            "embedding width does not match encoder.embedding_dim; "
            "features and readout would silently mis-align."
        )

    capacity = max(1, min(capacity, len(available)))
    stream = _RandomStream(seed)

    def sample_combo() -> tuple[str, ...]:
        upper = min(capacity, len(available))
        lower = min(min_combo_size, upper)
        size = lower + stream.next_int(upper - lower + 1)
        picked: list[str] = []
        seen: set[str] = set()
        guard = 0
        while len(picked) < size and guard < size * 12:
            guard += 1
            name = available[stream.next_int(len(available))]
            if name in seen:
                continue
            seen.add(name)
            picked.append(name)
        return tuple(sorted(picked))

    combos: list[tuple[str, ...]] = []
    seen_combos: set[tuple[str, ...]] = set()
    attempts = 0
    budget = samples * 6
    while len(combos) < samples and attempts < budget:
        attempts += 1
        combo = sample_combo()
        if not combo or combo in seen_combos:
            continue
        if len(combo) != len(set(combo)):
            continue
        seen_combos.add(combo)
        combos.append(combo)

    if len(combos) < 8:
        return None

    dim = encoder.embedding_dim
    extra_width = len(_extra_features(query, 1))
    bond_unary_width = 0 if bond_basis is None else bond_basis.unary_width
    bond_pair_width = 0 if bond_basis is None else bond_basis.pair_width
    width = dim * 2 + bond_unary_width + bond_pair_width + extra_width

    labels: list[float] = []
    rows: list[list[float]] = []
    for combo in combos:
        label = float(oracle(combo))
        labels.append(label)
        vectors = [embedding_by_name[name] for name in combo]
        row = [0.0] * width
        for vector in vectors:
            for channel in range(dim):
                row[channel] += vector[channel]
        offset = dim
        if bond_basis is not None:
            for name in combo:
                for index, value in enumerate(bond_basis.unary_features(name)):
                    row[offset + index] += value
            offset += bond_unary_width
        for left in range(len(vectors)):
            left_vector = vectors[left]
            for right in range(left + 1, len(vectors)):
                right_vector = vectors[right]
                for channel in range(dim):
                    row[offset + channel] += left_vector[channel] * right_vector[channel]
        offset += dim
        if bond_basis is not None:
            for left in range(len(combo)):
                for right in range(left + 1, len(combo)):
                    for index, value in enumerate(
                        bond_basis.pair_features(combo[left], combo[right])
                    ):
                        row[offset + index] += value
            offset += bond_pair_width
        for index, value in enumerate(_extra_features(query, len(vectors))):
            row[offset + index] = value
        rows.append(row)

    # 按标签尺度归一化，否则岭罚项和真实分数不在一个量纲上。
    scale = max(1e-6, sum(abs(value) for value in labels) / len(labels))
    normalized = [value / scale for value in labels]
    spread = max(normalized) - min(normalized)
    degenerate = spread < 1e-3

    order = list(range(len(rows)))
    stream_shuffle = _RandomStream(seed + 1)
    for index in range(len(order) - 1, 0, -1):
        swap = stream_shuffle.next_int(index + 1)
        order[index], order[swap] = order[swap], order[index]
    holdout = max(1, int(len(order) * holdout_ratio))
    test_index = set(order[:holdout])
    train_rows = [rows[index] for index in range(len(rows)) if index not in test_index]
    train_labels = [normalized[index] for index in range(len(rows)) if index not in test_index]
    if degenerate:
        # 该房间规格对所有组合给出同一个分数（例如 guide 刊例把 L3 合成玉贸易站
        # 固定成 132%）。这时任何组合都等价，拟合权重只会变成噪声，直接给常数模型。
        coeffs = [0.0] * width
    else:
        # 显式羁绊基和嵌入基共享同一个岭罚项，但羁绊基只有十几列、样本又少，
        # 需要更强的收缩。按列缩放把它的有效惩罚放大 ``bond_ridge_scale`` 倍：
        # 列乘 c 后拟合系数约乘 1/c，|w| 变大 ⇒ 罚得更重。
        scaled_rows = train_rows
        if bond_basis is not None and bond_ridge_scale > 1.0:
            scale_column = 1.0 / math.sqrt(bond_ridge_scale)
            scaled_rows = [
                [
                    *row[: dim + bond_unary_width],
                    *(value * scale_column for value in row[dim + bond_unary_width : dim + bond_unary_width + bond_pair_width]),
                    *row[dim + bond_unary_width + bond_pair_width :],
                ]
                for row in train_rows
            ]
        coeffs = _solve_ridge(scaled_rows, train_labels, ridge / max(1, len(train_rows)))
        if bond_basis is not None and bond_ridge_scale > 1.0:
            scale_column = 1.0 / math.sqrt(bond_ridge_scale)
            # 列乘 c 后拟合系数约乘 1/c，还原时再乘回 c。
            cols = range(dim + bond_unary_width, dim + bond_unary_width + bond_pair_width)
            for column in cols:
                coeffs[column] *= scale_column

    test_rows = [rows[index] for index in range(len(rows)) if index in test_index]
    test_actual = [normalized[index] for index in range(len(rows)) if index in test_index]
    train_pred, test_pred = project_predictions(train_rows, test_rows, coeffs)
    train_actual = train_labels

    model = RoomValueModel(
        query=query,
        embedding_dim=dim,
        unary=coeffs[: dim + bond_unary_width],
        pair=coeffs[dim + bond_unary_width : dim * 2 + bond_unary_width + bond_pair_width],
        extra=coeffs[dim * 2 + bond_unary_width + bond_pair_width :],
        bias=0.0,
        bond_basis=bond_basis,
        fit_quality={
            "r2Train": 1.0 if degenerate else round(_r2(train_pred, train_actual), 4),
            "r2Holdout": 1.0 if degenerate else round(_r2(test_pred, test_actual), 4),
            "spearmanHoldout": 0.0 if degenerate else round(_spearman(test_pred, test_actual), 4),
            "maeHoldout": round(_mae(test_pred, test_actual) * scale, 4),
            "maeRatio": round(_mae(test_pred, test_actual), 6),
            "labelScale": round(scale, 4),
            "labelSpread": round(spread * scale, 6),
            "degenerate": degenerate,
            "solver": "numpy" if _np is not None else "python",
        },
        training={
            "samples": len(rows),
            "trainSamples": len(train_rows),
            "holdoutSamples": len(test_rows),
            "capacity": capacity,
            "candidatePool": len(available),
            "parameterCount": width + 1,
            "bondUnaryWidth": bond_unary_width,
            "bondPairWidth": bond_pair_width,
            "ridge": ridge,
            "seed": seed,
        },
    )
    # 反归一化：让读出层直接输出模拟器量纲的房间分数。
    model.unary = [value * scale for value in model.unary]
    model.pair = [value * scale for value in model.pair]
    model.extra = [value * scale for value in model.extra]
    return model


def project_predictions(
    train_rows: Sequence[Sequence[float]],
    test_rows: Sequence[Sequence[float]],
    coeffs: Sequence[float],
) -> tuple[list[float], list[float]]:
    """用同一份系数给训练集和留出集打分（numpy 可用时批量算）。"""
    if _np is not None and coeffs:
        weights = _np.asarray(coeffs, dtype=float)
        train_pred = (
            _np.asarray(train_rows, dtype=float) @ weights if train_rows else _np.zeros(0)
        )
        test_pred = (
            _np.asarray(test_rows, dtype=float) @ weights if test_rows else _np.zeros(0)
        )
        return [float(v) for v in train_pred], [float(v) for v in test_pred]
    return (
        [sum(w * v for w, v in zip(coeffs, row)) for row in train_rows],
        [sum(w * v for w, v in zip(coeffs, row)) for row in test_rows],
    )


def rank_combinations(
    model: RoomValueModel,
    embeddings: Mapping[str, Sequence[float]],
    candidates: Sequence[str],
    *,
    capacity: int,
    beam: int = 18,
    top_k: int = 12,
    unary_cache: dict[str, float] | None = None,
    pair_cache: dict[tuple[str, str], float] | None = None,
    unary_adjust: Mapping[str, float] | None = None,
) -> list[tuple[float, tuple[str, ...]]]:
    """用读出层给候选组合排序。

    关键性能细节：一元项按“干员”缓存、二元项按“干员对”缓存，
    因此枚举组合时只是在查表求和（O(k²) 次查表），而不是每次都做
    ``O(k²·d)`` 的点积。同一个房间规格会被反复查询（旧算法里 91 次），
    缓存命中后一次排序是亚毫秒级，这正是能替代全量枚举的原因。

    ``unary_adjust`` 是“可精确计算、且对组合可加”的修正项，目前用于心情/疲劳：
    ``fatigueRisk`` 对同房间干员是可加的，所以把 ``-2 * duration * 每小时心情消耗``
    直接加到一元项上，让读出层只负责拟合产能，不必从噪声里学这种线性项。
    """
    capacity = max(1, capacity)
    dim = model.embedding_dim
    unary_cache = {} if unary_cache is None else unary_cache
    pair_cache = {} if pair_cache is None else pair_cache
    adjust = dict(unary_adjust or {})
    unary_weights = model.unary
    bond_unary_weights = model.unary_bond_weights
    pair_bond_weights = model.pair_bond_weights

    def unary_value(name: str) -> float:
        value = unary_cache.get(name)
        if value is None:
            vector = embeddings[name]
            total = 0.0
            for index in range(dim):
                total += unary_weights[index] * vector[index]
            for index, flag in enumerate(model.unary_bond(name)):
                if flag:
                    total += bond_unary_weights[index] * flag
            value = total
            unary_cache[name] = value
        return value + adjust.get(name, 0.0)

    scored_names: list[tuple[float, str]] = []
    for name in candidates:
        if name not in embeddings:
            continue
        scored_names.append((unary_value(name), name))
    scored_names.sort(key=lambda item: (-item[0], item[1]))
    pool = [name for _, name in scored_names[: max(capacity, beam)]]
    if len(pool) < capacity:
        capacity = len(pool)
    if capacity < 1:
        return []

    def pair_value(left: str, right: str) -> float:
        key = (left, right) if left <= right else (right, left)
        value = pair_cache.get(key)
        if value is None:
            left_vector = embeddings[left]
            right_vector = embeddings[right]
            total = 0.0
            for index in range(dim):
                total += model.pair[index] * left_vector[index] * right_vector[index]
            if bond_unary_weights:
                for index, flag in enumerate(model.pair_bond(left, right)):
                    if flag:
                        total += pair_bond_weights[index] * flag
            value = total
            pair_cache[key] = value
        return value

    if _np is not None:
        return _rank_combinations_numpy(
            pool, capacity, top_k, unary_value, pair_value, adjust
        )

    scored: list[tuple[float, tuple[str, ...]]] = []
    chosen: list[str] = []

    def total() -> float:
        value = 0.0
        for name in chosen:
            value += unary_value(name)
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                value += pair_value(chosen[i], chosen[j])
        return value

    def recurse(start: int) -> None:
        if len(chosen) == capacity:
            scored.append((total(), tuple(chosen)))
            return
        remaining = capacity - len(chosen)
        for index in range(start, len(pool) - remaining + 1):
            chosen.append(pool[index])
            recurse(index + 1)
            chosen.pop()

    recurse(0)
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[: max(1, top_k)]


def _rank_combinations_numpy(
    pool: Sequence[str],
    capacity: int,
    top_k: int,
    unary_value: Callable[[str], float],
    pair_value: Callable[[str, str], float],
    adjust: Mapping[str, float],
) -> list[tuple[float, tuple[str, ...]]]:
    """numpy 批量版：把一元分和二元分装成数组，一次算出全部组合分。"""
    assert _np is not None
    size = len(pool)
    unary = _np.asarray([unary_value(name) for name in pool], dtype=float)
    pair = _np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(i + 1, size):
            value = pair_value(pool[i], pool[j])
            pair[i, j] = value
            pair[j, i] = value
    index_combos = list(_combinations(range(size), capacity))
    if not index_combos:
        return []
    indices = _np.asarray(index_combos, dtype=_np.intp)
    scores = unary[indices].sum(axis=1)
    for left in range(capacity):
        for right in range(left + 1, capacity):
            scores = scores + pair[indices[:, left], indices[:, right]]
    order = _np.argsort(-scores, kind="stable")
    results: list[tuple[float, tuple[str, ...]]] = []
    for position in order[: max(1, top_k)]:
        combo_indices = indices[position]
        results.append(
            (
                float(scores[position]),
                tuple(pool[int(index)] for index in combo_indices),
            )
        )
    return results


# ---------------------------------------------------------------------------
# M3b：房间节点 + 「房间 × 干员」亲和度矩阵（一条共享读出层）
# ---------------------------------------------------------------------------
#
# 旧形态（``RoomValueModel``）是**每间房各拟合一套一元/二元权重**：拆开看的话，
# 它其实是一张 ``房间 × 干员`` 表，但每间房是独立求解的，房间之间不共享任何东西，
# 也没法给「没拟合过的房间规格」打分。
#
# M3b 的形态是：房间先编码成节点嵌入 ``r``（房型/产物 → 节点特征，等级/工位/时长 →
# 查询尾巴，再叠一层发电站/控制中枢的消息传递），读出层变成**双线性形式**
#
# .. math::
#
#     \\hat{y}(S \\mid r) = b \\cdot r
#         + \\sum_{i \\in S} (A r) \\cdot e_i
#         + \\sum_{i < j \\in S} w_p \\cdot (e_i \\odot e_j)
#         + \\sum_j (C r)_j \\cdot \\mathrm{extra}_j(S)
#
# 其中 ``A`` 就是**亲和度矩阵**：``A[c][k]`` 读作「房间特征 k 让干员嵌入通道 c 值多少分」。
# 于是同一张表能给任意「房间 × 干员」对打分（``affinity_matrix``），
# 也能把任意房间规格“物化”成一个 :class:`RoomValueModel`（``view``），
# 所以下游的排序/短名单代码一行都不用改。
#
# 两条刻意的取舍（都是为了不让参数爆炸到无法估计）：
#
# 1. **二元协同项 ``w_p`` 不做房间条件**。房间对「两人同房」的影响已经由
#    显式羁绊基（``BondBasis``，按房间/产物门控，见 ``graph_bonds``）承担；
#    再把 47 维二元权重也乘上房间基会多出 1410 个参数，样本撑不住。
# 2. **显式羁绊系数仍按房间单独拟合**（在共享模型的残差上做一次小岭回归），
#    所以「每多一名 glasgow 值多少分」这类可解释系数没有丢。

#: 房间**查询**属性（等级 / 工位 / 时长）的基；与房间节点的静态特征拼成房间嵌入的输入。
ROOM_QUERY_FEATURE_NAMES: tuple[str, ...] = (
    "level",
    "slots",
    "duration",
    "levelSlots",
    "levelDuration",
    "slotsDuration",
)

ROOM_CONDITION_DIM = len(ROOM_FEATURE_NAMES) + len(ROOM_QUERY_FEATURE_NAMES)

#: 房间节点之间的关系权重（固定超参，和学习无关）。中枢加成比供电更“贴近产能”，给更高权重。
ROOM_RELATION_WEIGHTS: dict[str, float] = {"power": 0.55, "control": 0.45}


@dataclass(frozen=True)
class RoomCondition:
    """一次房间查询：房间节点身份 + 等级/工位/时长。

    ``duration_hours`` 必须进来：订单上限、时长带来的饱和都是电梯度的一部分，
    共享模型要在同一条表里跨时长泛化，就不能把时长写在模型外面。
    """

    room_type: str
    target: str | None = None
    room_level: int = 3
    slots: int = 3
    duration_hours: float = 12.0

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.room_type, self.target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roomType": self.room_type,
            "target": self.target,
            "roomLevel": int(self.room_level),
            "slots": int(self.slots),
            "durationHours": round(float(self.duration_hours), 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoomCondition":
        return cls(
            room_type=str(payload["roomType"]),
            target=payload.get("target"),
            room_level=int(payload.get("roomLevel", 3)),
            slots=int(payload.get("slots", 3)),
            duration_hours=float(payload.get("durationHours", 12.0)),
        )

    def query_features(self) -> tuple[float, ...]:
        level = max(0.0, min(1.0, self.room_level / 3.0))
        slots = max(0.0, min(1.0, self.slots / 5.0))
        duration = max(0.0, min(1.0, self.duration_hours / 24.0))
        return (
            level,
            slots,
            duration,
            level * slots,
            level * duration,
            slots * duration,
        )


@dataclass(frozen=True)
class RoomEncoderConfig:
    """房间编码器的固定超参（与 ``EncoderConfig`` 同构，但规模小得多）。"""

    hidden_dim: int = 8
    layers: int = 1
    self_weight: float = 0.65
    activation_slope: float = 0.15
    seed: int = 20260701
    nonzeros_per_channel: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "hiddenDim": self.hidden_dim,
            "layers": self.layers,
            "selfWeight": self.self_weight,
            "activationSlope": self.activation_slope,
            "seed": self.seed,
            "nonzerosPerChannel": self.nonzeros_per_channel,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoomEncoderConfig":
        return cls(
            hidden_dim=int(payload.get("hiddenDim", 8)),
            layers=int(payload.get("layers", 1)),
            self_weight=float(payload.get("selfWeight", 0.65)),
            activation_slope=float(payload.get("activationSlope", 0.15)),
            seed=int(payload.get("seed", 20260701)),
            nonzeros_per_channel=int(payload.get("nonzerosPerChannel", 3)),
        )


class RoomEncoder:
    """把 :class:`RoomCondition` 编码成定长嵌入。

    ``h' = act( ws·W_self·h + Σ_r wr·W_r·(A_r h) )``，与干员侧的图卷积同一套规则，
    只是邻居是**房间节点**：``power`` 边来自发电站、``control`` 边来自控制中枢。
    邻居的嵌入是静态的，所以消息传递项可以**预先算成每个节点的常数偏移**，
    查询时只剩一次稀疏投影（亚微秒级）。
    """

    def __init__(
        self, room_graph: StaticRoomGraph, config: RoomEncoderConfig | None = None
    ) -> None:
        self.room_graph = room_graph
        self.config = config or RoomEncoderConfig()
        self._feature_index = {name: index for index, name in enumerate(ROOM_FEATURE_NAMES)}
        self._build_projections()
        self._neighbour_offset: list[list[list[float]]] = self._build_neighbour_offsets()
        self._static: list[tuple[float, ...]] = [node.features for node in room_graph.nodes]

    # ------------------------------------------------------------------
    @property
    def room_dim(self) -> int:
        return ROOM_CONDITION_DIM + self.config.layers * self.config.hidden_dim

    def _build_projections(self) -> None:
        config = self.config
        stream = _RandomStream(config.seed)
        dim_h = config.hidden_dim
        self._self_projections: list[_SparseProjection] = []
        self._relation_projections: dict[str, list[_SparseProjection]] = {
            relation: [] for relation in ROOM_RELATIONS
        }
        for layer in range(config.layers):
            dim_in = ROOM_CONDITION_DIM if layer == 0 else dim_h
            self._self_projections.append(
                _build_sparse_projection(
                    stream, dim_in, dim_h, config.nonzeros_per_channel, 1.0 / math.sqrt(dim_h)
                )
            )
            for relation in ROOM_RELATIONS:
                self._relation_projections[relation].append(
                    _build_sparse_projection(
                        stream, dim_in, dim_h, config.nonzeros_per_channel, 1.0 / math.sqrt(dim_h)
                    )
                )

    def _build_neighbour_offsets(self) -> list[list[list[float]]]:
        """``[layer][node] -> 常数偏移向量``（邻居消息传递的结果，与查询无关）。

        干员的房间节点邻居是**原型**（POWER / CONTROL），不带查询尾巴，
        所以它们的嵌入是固定值；因此这一项对每个节点是常数，可以完全预计算。
        """
        config = self.config
        dim_h = config.hidden_dim
        size = self.room_graph.size
        # 原型节点的高度：第一层输入是静态特征（查询尾巴补零），之后是隐表示。
        heights: list[list[list[float]]] = [
            [list(node.features) + [0.0] * len(ROOM_QUERY_FEATURE_NAMES) for node in self.room_graph.nodes]
        ]
        offsets: list[list[list[float]]] = []
        for layer in range(config.layers):
            height = heights[layer]
            aggregated: dict[str, list[list[float]]] = {
                relation: [[0.0] * len(height[0]) for _ in range(size)]
                for relation in ROOM_RELATIONS
            }
            for edge in self.room_graph.edges:
                if edge.relation not in aggregated:
                    continue
                bucket = aggregated[edge.relation][edge.dst]
                source = height[edge.src]
                for channel in range(len(source)):
                    bucket[channel] += source[channel] * edge.weight
            layer_offsets: list[list[float]] = []
            for node_index in range(size):
                total = [0.0] * dim_h
                for relation in ROOM_RELATIONS:
                    weight = ROOM_RELATION_WEIGHTS.get(relation, 0.0)
                    if not weight:
                        continue
                    projected = self._relation_projections[relation][layer].apply(
                        aggregated[relation][node_index]
                    )
                    for channel in range(dim_h):
                        total[channel] += projected[channel] * weight
                layer_offsets.append(total)
            offsets.append(layer_offsets)
            # 下一层的原型高度需要本层的隐表示（查询尾巴仍然为零，因为原型没有查询）。
            next_height: list[list[float]] = []
            for node_index in range(size):
                activated = self._self_projections[layer].apply(height[node_index])
                for channel in range(dim_h):
                    activated[channel] = activated[channel] * config.self_weight
                    activated[channel] += offsets[layer][node_index][channel]
                next_height.append(
                    [
                        value if value >= 0 else value * config.activation_slope
                        for value in activated
                    ]
                )
            heights.append(next_height)
        return offsets

    # ------------------------------------------------------------------
    def embedding(self, condition: RoomCondition) -> tuple[float, ...]:
        """房间节点嵌入 = 输入基（房间节点特征 ⊕ 查询尾巴）⊕ 每层隐表示。"""
        node_index = self.room_graph.index(condition.room_type, condition.target)
        if node_index is None:
            base = [0.0] * ROOM_CONDITION_DIM
        else:
            base = list(self._static[node_index]) + [0.0] * len(ROOM_QUERY_FEATURE_NAMES)
        tail = condition.query_features()
        offset = len(ROOM_FEATURE_NAMES)
        for index, value in enumerate(tail):
            base[offset + index] = value
        stack: list[float] = list(base)
        height: list[float] = base
        slope = self.config.activation_slope
        dim_h = self.config.hidden_dim
        for layer in range(self.config.layers):
            activated = self._self_projections[layer].apply(height)
            node_offset = (
                self._neighbour_offset[layer][node_index]
                if node_index is not None
                else [0.0] * dim_h
            )
            for channel in range(dim_h):
                value = activated[channel] * self.config.self_weight + node_offset[channel]
                stack.append(value if value >= 0 else value * slope)
            height = stack[-dim_h:]
        return tuple(stack)

    def profile(self) -> dict[str, Any]:
        return {
            "encoder": self.config.to_dict(),
            "roomDim": self.room_dim,
            "roomNodeCount": self.room_graph.size,
            "roomEdgeCount": len(self.room_graph.edges),
            "roomGraphSignature": self.room_graph.meta.get("signature"),
        }


@dataclass(frozen=True)
class AffinityFitRoom:
    """共享拟合时的一个房间规格：候选干员 + 精确打分器（模拟器）+ 羁绊基。"""

    condition: RoomCondition
    candidates: tuple[str, ...]
    capacity: int
    oracle: Callable[[tuple[str, ...]], float]
    bond_basis: BondBasis | None = None


#: 共享读出层的附加项宽度（与 ``_extra_features`` 对齐）。
_AFFINITY_EXTRA_WIDTH = 4


@dataclass
class RoomAffinityModel:
    """一条**共享**的「房间 × 干员」读出层（M3b）。

    参数布局：

    * ``unary[room_dim][embedding_dim]`` —— 亲和度矩阵 ``A``：房间特征 → 一元权重；
    * ``pair[embedding_dim]`` —— 房间无关的二元协同权重；
    * ``extra[room_dim][4]`` —— 房间特征 → 数量/等级/工位的附加项权重；
    * ``bias[room_dim]`` —— 房间特征 → 截距；
    * ``bonds[房间键]`` —— 每个房间各自拟合的显式羁绊系数（阵营人数 / 点名羁绊）；
    * ``scales[房间键]`` —— 每个房间的标签尺度（房间分数量级差得很远，见下面的拟合注释）。
    """

    encoder: RoomEncoder
    embedding_dim: int
    unary: list[list[float]]
    pair: list[float]
    extra: list[list[float]]
    bias: list[float]
    bonds: dict[str, dict[str, Any]]
    scales: dict[str, float] = field(default_factory=dict)
    default_scale: float = 1.0
    fit_quality: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    @property
    def room_dim(self) -> int:
        return self.encoder.room_dim

    def scale_for(self, condition: RoomCondition) -> float:
        """该房间的标签尺度；没见过的房间用已知房间的均值（线性外推不如均值稳）。"""
        return float(
            self.scales.get(_room_key(condition.room_type, condition.target), self.default_scale)
        )

    @property
    def parameter_count(self) -> int:
        bond_parameters = sum(
            len(entry["unary"]) + len(entry["pair"]) for entry in self.bonds.values()
        )
        return (
            self.room_dim * self.embedding_dim
            + self.embedding_dim
            + self.room_dim * _AFFINITY_EXTRA_WIDTH
            + self.room_dim
            + bond_parameters
            + len(self.scales)
        )

    # ------------------------------------------------------------------
    def unary_weights(self, condition: RoomCondition) -> list[float]:
        """亲和度矩阵在该房间上的一列：``w_u = scale(房间) · A r``。"""
        vector = self.encoder.embedding(condition)
        scale = self.scale_for(condition)
        weights = [0.0] * self.embedding_dim
        for channel in range(self.embedding_dim):
            row = self.unary[channel]
            total = 0.0
            for index, value in enumerate(vector):
                total += row[index] * value
            weights[channel] = total * scale
        return weights

    def affinity_scores(
        self,
        condition: RoomCondition,
        names: Sequence[str],
        embeddings: Mapping[str, Sequence[float]],
    ) -> dict[str, float]:
        """任意「房间 × 干员」对的亲和度（一元预测分）。

        这是 M3b 相对「每间房各拟合一套」多出来的东西：同一张表可以**跨房间**
        给任意干员打分，不需要该房间曾经拟合过。
        """
        weights = self.unary_weights(condition)
        bond_entry = self.bonds.get(_room_key(condition.room_type, condition.target))
        bond_weights = bond_entry["unary"] if bond_entry else ()
        bond_basis = self._bond_basis(bond_entry)
        scores: dict[str, float] = {}
        for name in names:
            vector = embeddings.get(name)
            if vector is None:
                continue
            if len(vector) != self.embedding_dim:
                raise ValueError(
                    f"embedding width {len(vector)} != affinity model width "
                    f"{self.embedding_dim}; pass the encoder the model was fitted with."
                )
            total = 0.0
            for index, value in enumerate(vector):
                total += weights[index] * value
            if bond_basis is not None:
                for index, flag in enumerate(bond_basis.unary_features(name)):
                    if flag:
                        total += bond_weights[index] * flag
            scores[name] = total
        return scores

    def affinity_matrix(
        self,
        conditions: Sequence[RoomCondition],
        names: Sequence[str],
        embeddings: Mapping[str, Sequence[float]],
    ) -> list[dict[str, float]]:
        """``房间 × 干员`` 的亲和度矩阵（每行一个房间）。"""
        return [self.affinity_scores(condition, names, embeddings) for condition in conditions]

    # ------------------------------------------------------------------
    @staticmethod
    def _bond_basis(entry: Mapping[str, Any] | None) -> BondBasis | None:
        if not entry:
            return None
        payload = entry.get("basis")
        return None if payload is None else BondBasis.from_dict(payload)

    def view(self, condition: RoomCondition) -> RoomValueModel:
        """把共享表**物化**成该房间规格的 :class:`RoomValueModel`。

        这样下游的 ``rank_combinations`` / 短名单 / 精确复核一行都不用改：
        亲和度矩阵只是换了一种估计一元/二元权重的方式。
        """
        vector = self.encoder.embedding(condition)
        scale = self.scale_for(condition)
        query = RoomQuery(
            room_type=condition.room_type,
            target=condition.target,
            room_level=condition.room_level,
            slots=condition.slots,
        )
        bond_entry = self.bonds.get(_room_key(condition.room_type, condition.target))
        bond_basis = self._bond_basis(bond_entry)
        unary = self.unary_weights(condition)
        if bond_basis is not None and bond_entry is not None:
            unary = [*unary, *[float(value) for value in bond_entry["unary"]]]
        pair = [value * scale for value in self.pair]
        if bond_basis is not None and bond_entry is not None:
            pair = [*pair, *[float(value) for value in bond_entry["pair"]]]
        extra: list[float] = []
        for index in range(_AFFINITY_EXTRA_WIDTH):
            row = self.extra[index]
            total = 0.0
            for position, value in enumerate(vector):
                total += row[position] * value
            extra.append(total * scale)
        bias = 0.0
        for position, value in enumerate(vector):
            bias += self.bias[position] * value
        bias *= scale
        return RoomValueModel(
            query=query,
            embedding_dim=self.embedding_dim,
            unary=unary,
            pair=pair,
            extra=extra,
            bias=bias,
            bond_basis=bond_basis,
            fit_quality=dict(self.fit_quality),
            training=dict(self.training),
        )

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "readout": "room_operator_affinity",
            "roomDim": self.room_dim,
            "embeddingDim": self.embedding_dim,
            "parameterCount": self.parameter_count,
            "rooms": sorted(self.bonds),
            "encoder": self.encoder.profile(),
            **self.fit_quality,
            "training": self.training,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "embeddingDim": self.embedding_dim,
            "encoder": self.encoder.config.to_dict(),
            "unary": [list(row) for row in self.unary],
            "pair": list(self.pair),
            "extra": [list(row) for row in self.extra],
            "bias": list(self.bias),
            "scales": {key: float(value) for key, value in self.scales.items()},
            "defaultScale": float(self.default_scale),
            "bonds": {
                key: {
                    "basis": None if entry.get("basis") is None else dict(entry["basis"]),
                    "unary": [float(value) for value in entry["unary"]],
                    "pair": [float(value) for value in entry["pair"]],
                }
                for key, entry in self.bonds.items()
            },
            "fitQuality": dict(self.fit_quality),
            "training": dict(self.training),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], room_graph: StaticRoomGraph
    ) -> "RoomAffinityModel":
        encoder = RoomEncoder(
            room_graph, RoomEncoderConfig.from_dict(payload.get("encoder") or {})
        )
        return cls(
            encoder=encoder,
            embedding_dim=int(payload["embeddingDim"]),
            unary=[[float(value) for value in row] for row in payload["unary"]],
            pair=[float(value) for value in payload["pair"]],
            extra=[[float(value) for value in row] for row in payload["extra"]],
            bias=[float(value) for value in payload["bias"]],
            scales={
                str(key): float(value)
                for key, value in (payload.get("scales") or {}).items()
            },
            default_scale=float(payload.get("defaultScale", 1.0)),
            bonds={
                str(key): {
                    "basis": entry.get("basis"),
                    "unary": [float(value) for value in entry["unary"]],
                    "pair": [float(value) for value in entry["pair"]],
                }
                for key, entry in (payload.get("bonds") or {}).items()
            },
            fit_quality=dict(payload.get("fitQuality") or {}),
            training=dict(payload.get("training") or {}),
        )


def _room_key(room_type: str, target: str | None) -> str:
    return f"{room_type}|{'' if target is None else target}"


def _affinity_row_layout(embedding_dim: int, room_dim: int) -> dict[str, tuple[int, int]]:
    """设计矩阵的分段布局（``[start, end)``）。"""
    unary_end = embedding_dim * room_dim
    pair_end = unary_end + embedding_dim
    extra_end = pair_end + _AFFINITY_EXTRA_WIDTH * room_dim
    bias_end = extra_end + room_dim
    return {
        "unary": (0, unary_end),
        "pair": (unary_end, pair_end),
        "extra": (pair_end, extra_end),
        "bias": (extra_end, bias_end),
    }


def _affinity_design_row(
    room_vector: Sequence[float],
    vectors: Sequence[Sequence[float]],
    extras: Sequence[float],
    embedding_dim: int,
    room_dim: int,
) -> list[float]:
    """构造一行设计矩阵：``[kron(r, Σe) | Σ e_i⊙e_j | kron(r, extras) | r]``。"""
    unary_width = embedding_dim * room_dim
    row = [0.0] * (unary_width + embedding_dim + _AFFINITY_EXTRA_WIDTH * room_dim + room_dim)
    summed = [0.0] * embedding_dim
    for vector in vectors:
        for channel in range(embedding_dim):
            summed[channel] += vector[channel]
    for channel in range(embedding_dim):
        base = channel * room_dim
        value = summed[channel]
        for position in range(room_dim):
            row[base + position] = value * room_vector[position]
    pair_offset = unary_width
    for left in range(len(vectors)):
        left_vector = vectors[left]
        for right in range(left + 1, len(vectors)):
            right_vector = vectors[right]
            for channel in range(embedding_dim):
                row[pair_offset + channel] += left_vector[channel] * right_vector[channel]
    extra_offset = unary_width + embedding_dim
    for index in range(_AFFINITY_EXTRA_WIDTH):
        base = extra_offset + index * room_dim
        value = extras[index] if index < len(extras) else 0.0
        for position in range(room_dim):
            row[base + position] = value * room_vector[position]
    bias_offset = extra_offset + _AFFINITY_EXTRA_WIDTH * room_dim
    for position in range(room_dim):
        row[bias_offset + position] = room_vector[position]
    return row


def _fill_affinity_design_row(
    matrix: Any,
    row_index: int,
    room_vector: Sequence[float],
    vectors: Sequence[Sequence[float]],
    extras: Sequence[float],
    embedding_dim: int,
    room_dim: int,
    layout: Mapping[str, tuple[int, int]],
) -> None:
    """把一行直接写进 numpy 矩阵（池化拟合 9000 行 × 1600 列时，纯 Python 列表会吃掉几百 MB）。"""
    assert _np is not None
    r = _np.asarray(room_vector, dtype=float)
    unary_start, unary_end = layout["unary"]
    summed = _np.zeros(embedding_dim, dtype=float)
    pairwise = _np.zeros(embedding_dim, dtype=float)
    for left in range(len(vectors)):
        left_vector = _np.asarray(vectors[left], dtype=float)
        summed += left_vector
        for right in range(left + 1, len(vectors)):
            pairwise += left_vector * _np.asarray(vectors[right], dtype=float)
    matrix[row_index, unary_start:unary_end] = (summed[:, None] * r[None, :]).ravel()
    pair_start, pair_end = layout["pair"]
    matrix[row_index, pair_start:pair_end] = pairwise
    extra_start, _extra_end = layout["extra"]
    for index in range(_AFFINITY_EXTRA_WIDTH):
        value = extras[index] if index < len(extras) else 0.0
        base = extra_start + index * room_dim
        matrix[row_index, base : base + room_dim] = value * r
    bias_start, bias_end = layout["bias"]
    matrix[row_index, bias_start:bias_end] = r


def _sample_affinity_combos(
    available: Sequence[str], capacity: int, samples: int, seed: int
) -> list[tuple[str, ...]]:
    """采样组合，**一半预算给满编**（与 ``fit_room_value_model`` 的均匀采样不同）。

    为什么偏袒满编：决定短名单提名的是满编组合，一元/二元项只是拟合用的中间基。
    均匀采样时 3 工位的房间只有 1/3 的样本是满编（256 样本 ⇒ 85 个满编），
    高分区覆盖太稀，模型学出来的排序在真正要用的那一段不可靠。
    实测这一点比「把样本翻倍」便宜得多（见 §3 第 6 步 M3b 的实测表）。
    """
    stream = _RandomStream(seed)
    upper = max(1, min(capacity, len(available)))
    full_quota = max(4, min(samples, samples // 2))
    combos: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    budget = max(8, samples) * 6
    while len(combos) < samples and attempts < budget:
        attempts += 1
        size = upper if len(combos) < full_quota else 1 + stream.next_int(upper)
        picked: list[str] = []
        picked_set: set[str] = set()
        guard = 0
        while len(picked) < size and guard < size * 12:
            guard += 1
            name = available[stream.next_int(len(available))]
            if name in picked_set:
                continue
            picked_set.add(name)
            picked.append(name)
        combination = tuple(sorted(picked))
        if not combination or combination in seen:
            continue
        seen.add(combination)
        combos.append(combination)
    return combos


def _solve_ridge_design(
    design: Any,
    row_indices: Sequence[int],
    targets: Sequence[float],
    ridge: float,
) -> list[float]:
    """对（可能很大的）设计矩阵按行子集求岭回归，**分块累积正规方程**。

    共享亲和度矩阵的一次池化拟合是「36 个房间 × 256 样本 × 1607 列」量级：
    直接把训练行切出来做 ``X.T @ X`` 会临时复制 ~170MB。分块累积把峰值压到
    一个 2048 行的块（25MB）+ 正规方程矩阵（20MB）。
    """
    assert _np is not None
    width = int(design.shape[1])
    gram = _np.zeros((width, width), dtype=float)
    rhs = _np.zeros(width, dtype=float)
    target_array = _np.asarray(targets, dtype=float)
    block = 2048
    for start in range(0, len(row_indices), block):
        chunk = list(row_indices[start : start + block])
        rows = _np.asarray(design[chunk], dtype=float)
        gram += rows.T @ rows
        rhs += rows.T @ target_array[chunk]
    gram[_np.diag_indices_from(gram)] += ridge
    try:
        solution = _np.linalg.solve(gram, rhs)
    except _np.linalg.LinAlgError:  # 奇异时退回最小二乘
        solution = _np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return [float(value) for value in solution]


def fit_room_affinity_model(
    room_encoder: RoomEncoder,
    embeddings: Mapping[str, Sequence[float]],
    rooms: Sequence[AffinityFitRoom],
    *,
    samples: int = 128,
    seed: int = 11,
    ridge: float = 4.0,
    bond_ridge: float = 1.0,
    holdout_ratio: float = 0.25,
) -> RoomAffinityModel | None:
    """一条闭式读数层，池化拟合**所有**房间规格（M3b）。

    流程：按房间采样组合 → 用模拟器打分 → 拼成 ``kron(房间嵌入, 干员特征)`` 的设计矩阵
    → 一次岭回归解出亲和度矩阵 → 再用每个房间的**残差**单独拟合它的显式羁绊系数。

    返回 ``None`` 表示样本/候选不足（调用方应退回每间房各自拟合的旧路径）。
    """
    prepared: list[tuple[AffinityFitRoom, list[str], list[tuple[str, ...]]]] = []
    for index, room in enumerate(rooms):
        names = sorted({name for name in room.candidates if name in embeddings})
        if len(names) < 1:
            continue
        combos = _sample_affinity_combos(
            names, max(1, min(room.capacity, len(names))), samples, seed + 7919 * index
        )
        if len(combos) < 4:
            continue
        prepared.append((room, names, combos))
    if len(prepared) < 2:
        return None

    room_dim = room_encoder.room_dim
    embedding_dim = len(embeddings[prepared[0][1][0]])
    if any(len(vector) != embedding_dim for vector in embeddings.values()):
        raise AssertionError(
            "embedding width mismatch; affinity readout would silently mis-align."
        )

    width = (
        embedding_dim * room_dim + embedding_dim + _AFFINITY_EXTRA_WIDTH * room_dim + room_dim
    )
    layout = _affinity_row_layout(embedding_dim, room_dim)
    total_rows = sum(len(combos) for _room, _names, combos in prepared)
    use_numpy = _np is not None
    design: Any = _np.zeros((total_rows, width), dtype=float) if use_numpy else []
    labels: list[float] = []
    owner: list[int] = []
    cursor = 0
    for room_index, (room, _names, combos) in enumerate(prepared):
        room_vector = room_encoder.embedding(room.condition)
        query = RoomQuery(
            room_type=room.condition.room_type,
            target=room.condition.target,
            room_level=room.condition.room_level,
            slots=room.condition.slots,
        )
        for combo in combos:
            vectors = [embeddings[name] for name in combo]
            extras = _extra_features(query, len(combo))
            if use_numpy:
                _fill_affinity_design_row(
                    design,
                    cursor,
                    room_vector,
                    vectors,
                    extras,
                    embedding_dim,
                    room_dim,
                    layout,
                )
            else:
                design.append(
                    _affinity_design_row(
                        room_vector, vectors, extras, embedding_dim, room_dim
                    )
                )
            labels.append(float(room.oracle(combo)))
            owner.append(room_index)
            cursor += 1
    if len(labels) < 8:
        return None

    # **逐房间标签尺度**：房间分数量级差得很远（合成玉贸易站 ~280，龙门币贸易站 ~55），
    # 用同一个全局尺度归一化会让小分数房间的信号只占方差的几个百分点，被岭罚项直接压平
    # （实测 TRADING/O_GOLD 的逐房间 R² 会掉到 −0.8）。
    # 所以每个房间按自己的 ``mean(|y|)`` 归一化，共享模型学的是「形状」，
    # 量纲由每个房间的一个标量尺度还原（``scales``）；没见过的房间用已知房间的均值兜底。
    room_scale: dict[int, float] = {}
    for room_index in range(len(prepared)):
        values = [abs(labels[index]) for index in range(len(labels)) if owner[index] == room_index]
        room_scale[room_index] = max(1e-6, sum(values) / max(1, len(values)))
    normalized = [labels[index] / room_scale[owner[index]] for index in range(len(labels))]
    spread = max(normalized) - min(normalized)

    order = list(range(len(labels)))
    stream_shuffle = _RandomStream(seed + 1)
    for index in range(len(order) - 1, 0, -1):
        swap = stream_shuffle.next_int(index + 1)
        order[index], order[swap] = order[swap], order[index]
    holdout = max(1, int(len(order) * holdout_ratio))
    test_index = set(order[:holdout])
    train_indices = [index for index in order if index not in test_index]
    test_indices = [index for index in order if index in test_index]
    train_labels = [normalized[index] for index in train_indices]
    if spread < 1e-3 or not train_indices:
        coeffs = [0.0] * width
        prediction = [0.0] * len(labels)
    else:
        if use_numpy:
            coeffs = _solve_ridge_design(
                design,
                train_indices,
                [normalized[index] for index in range(len(normalized))],
                ridge / max(1, len(train_indices)),
            )
            prediction = [float(value) for value in design @ _np.asarray(coeffs, dtype=float)]
        else:
            train_matrix = [design[index] for index in train_indices]
            coeffs = _solve_ridge(
                train_matrix, train_labels, ridge / max(1, len(train_indices))
            )
            prediction = [
                sum(weight * value for weight, value in zip(coeffs, row)) for row in design
            ]

    unary_width = embedding_dim * room_dim
    unary: list[list[float]] = [
        [coeffs[channel * room_dim + position] for position in range(room_dim)]
        for channel in range(embedding_dim)
    ]
    pair = [coeffs[unary_width + channel] for channel in range(embedding_dim)]
    extra_offset = unary_width + embedding_dim
    extra: list[list[float]] = [
        [
            coeffs[extra_offset + index * room_dim + position]
            for position in range(room_dim)
        ]
        for index in range(_AFFINITY_EXTRA_WIDTH)
    ]
    bias_offset = extra_offset + _AFFINITY_EXTRA_WIDTH * room_dim
    bias = [coeffs[bias_offset + position] for position in range(room_dim)]
    scales = {
        _room_key(prepared[room_index][0].condition.room_type, prepared[room_index][0].condition.target): room_scale[room_index]
        for room_index in range(len(prepared))
    }
    default_scale = sum(scales.values()) / max(1, len(scales))

    def shared_prediction(index: int, scale: float) -> float:
        return prediction[index] * scale

    def bond_row(basis: BondBasis, combo: Sequence[str]) -> list[float]:
        row = [0.0] * (basis.unary_width + basis.pair_width)
        for position, name in enumerate(combo):
            for column, flag in enumerate(basis.unary_features(name)):
                row[column] += flag
            for other in combo[position + 1 :]:
                for column, flag in enumerate(basis.pair_features(name, other)):
                    row[basis.unary_width + column] += flag
        return row

    def bond_contribution(
        basis: BondBasis,
        combo: Sequence[str],
        unary_weights: Sequence[float],
        pair_weights: Sequence[float],
    ) -> float:
        total = 0.0
        for position, name in enumerate(combo):
            for column, flag in enumerate(basis.unary_features(name)):
                if flag:
                    total += unary_weights[column] * flag
            for other in combo[position + 1 :]:
                for column, flag in enumerate(basis.pair_features(name, other)):
                    if flag:
                        total += pair_weights[column] * flag
        return total

    # ---- 显式羁绊系数：在共享模型的残差上按房间各做一次小岭回归 ----------
    bonds: dict[str, dict[str, Any]] = {}
    per_room_quality: dict[str, Any] = {}
    for room_index, (room, _names, combos) in enumerate(prepared):
        row_indices = [index for index, value in enumerate(owner) if value == room_index]
        if not row_indices:
            continue
        key = _room_key(room.condition.room_type, room.condition.target)
        basis = room.bond_basis
        entry: dict[str, Any] = {
            "basis": None if basis is None else basis.to_dict(),
            "unary": [],
            "pair": [],
        }
        if basis is not None:
            fit_rows = [
                (index, combos[position])
                for position, index in enumerate(row_indices)
                if index not in test_index
            ]
            if len(fit_rows) >= 4:
                residual = [
                    labels[index] - shared_prediction(index, room_scale[room_index])
                    for index, _combo in fit_rows
                ]
                bond_scale = max(
                    1e-6, sum(abs(value) for value in residual) / len(residual)
                )
                solved = _solve_ridge(
                    [bond_row(basis, combo) for _index, combo in fit_rows],
                    [value / bond_scale for value in residual],
                    bond_ridge / max(1, len(fit_rows)),
                )
                entry["unary"] = [
                    solved[column] * bond_scale for column in range(basis.unary_width)
                ]
                entry["pair"] = [
                    solved[basis.unary_width + column] * bond_scale
                    for column in range(basis.pair_width)
                ]
        if basis is not None and not entry["unary"]:
            # 羁绊基存在但样本不足没解出来：必须把基一起清掉，否则视图里
            # 「基有宽度、系数为空」会让 ``rank_combinations`` 越界。
            entry["basis"] = None
        bonds[key] = entry

        # 逐房间读数质量：用最终模型（共享 + 羁绊）在该房间自己的**留出行**上算误差。
        eval_rows = [
            (index, combos[position])
            for position, index in enumerate(row_indices)
            if index in test_index
        ] or [(index, combos[position]) for position, index in enumerate(row_indices)]
        predictions: list[float] = []
        actual: list[float] = []
        for index, combo in eval_rows:
            value = shared_prediction(index, room_scale[room_index])
            if basis is not None and entry["unary"]:
                value += bond_contribution(basis, combo, entry["unary"], entry["pair"])
            predictions.append(value)
            actual.append(labels[index])
        per_room_quality[
            f"{room.condition.room_type}|{room.condition.target}"
            f"|L{room.condition.room_level}|{int(room.condition.duration_hours)}h"
        ] = {
            "room": key,
            "samples": len(eval_rows),
            "r2": round(_r2(predictions, actual), 4),
            "mae": round(_mae(predictions, actual), 4),
            "level": room.condition.room_level,
            "durationHours": room.condition.duration_hours,
        }

    test_actual = [normalized[index] for index in test_indices]
    train_pred = [prediction[index] for index in train_indices]
    test_pred = [prediction[index] for index in test_indices]
    test_raw = [
        shared_prediction(index, room_scale[owner[index]]) for index in sorted(test_index)
    ]
    test_raw_actual = [labels[index] for index in sorted(test_index)]
    model = RoomAffinityModel(
        encoder=room_encoder,
        embedding_dim=embedding_dim,
        unary=unary,
        pair=pair,
        extra=extra,
        bias=bias,
        bonds=bonds,
        scales=scales,
        default_scale=default_scale,
        fit_quality={
            "r2Train": round(_r2(train_pred, train_labels), 4),
            "r2Holdout": round(_r2(test_pred, test_actual), 4),
            "spearmanHoldout": round(_spearman(test_pred, test_actual), 4),
            "maeHoldout": round(_mae(test_raw, test_raw_actual), 4),
            "labelScale": round(default_scale, 4),
            "degenerate": spread < 1e-3,
            "perRoom": per_room_quality,
            "solver": "numpy" if _np is not None else "python",
        },
        training={
            "samples": len(labels),
            "trainSamples": len(train_indices),
            "holdoutSamples": len(test_indices),
            "rooms": len(prepared),
            "samplesPerRoom": samples,
            "roomDim": room_dim,
            "embeddingDim": embedding_dim,
            "parameterCount": (
                embedding_dim * room_dim
                + embedding_dim
                + _AFFINITY_EXTRA_WIDTH * room_dim
                + room_dim
            ),
            "ridge": ridge,
            "seed": seed,
        },
    )
    return model


