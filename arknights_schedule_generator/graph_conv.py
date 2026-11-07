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
from typing import Any, Callable, Iterable, Mapping, Sequence

from .graph_features import FEATURE_NAMES, RELATIONS, ROOM_TYPES, StaticOperatorGraph

__all__ = [
    "EncoderConfig",
    "GraphEncoder",
    "RoomValueModel",
    "RoomQuery",
    "fit_room_value_model",
    "rank_combinations",
]


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
        if self._embeddings is not None:
            return list(self._embeddings)

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

        self._embeddings = tuple(
            tuple(value for row in layer_stack for value in row[node_index])
            for node_index in range(size)
        )
        return list(self._embeddings)

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

    只对 ``(room_type, target)`` 分组训练；房间等级/槽位通过额外标量特征进入。
    """

    query: RoomQuery
    embedding_dim: int
    unary: list[float]
    pair: list[float]
    extra: list[float]
    bias: float
    fit_quality: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    @property
    def query_key(self) -> tuple[str, str | None]:
        return self.query.key

    # ------------------------------------------------------------------
    def extra_features(self, count: int) -> list[float]:
        return _extra_features(self.query, count)

    def _features(self, vectors: Sequence[Sequence[float]]) -> list[float]:
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
        return [*unary, *pair, *self.extra_features(count)]

    def score_vectors(self, vectors: Sequence[Sequence[float]]) -> float:
        if not vectors:
            return self.bias
        features = self._features(vectors)
        total = self.bias
        dim = self.embedding_dim
        offset = 0
        for weight in self.unary:
            total += weight * features[offset]
            offset += 1
        for weight in self.pair:
            total += weight * features[offset]
            offset += 1
        for weight in self.extra:
            total += weight * features[offset]
            offset += 1
        return total

    def score(
        self,
        names: Sequence[str],
        embeddings: Mapping[str, Sequence[float]],
    ) -> float:
        vectors = [embeddings[name] for name in names if name in embeddings]
        if len(vectors) != len(names):
            raise KeyError(f"missing embedding for one of {list(names)}")
        return self.score_vectors(vectors)

    def describe(self) -> dict[str, Any]:
        return {
            "roomType": self.query_key[0],
            "target": self.query_key[1],
            "embeddingDim": self.embedding_dim,
            "parameterCount": len(self.unary) + len(self.pair) + len(self.extra) + 1,
            **self.fit_quality,
            "training": self.training,
        }


def _solve_ridge(rows: Sequence[Sequence[float]], targets: Sequence[float], ridge: float) -> list[float]:
    """正规方程 + 高斯-约当消元；纯 Python，参数规模 (<300) 下毫秒级。"""
    n = len(rows[0]) if rows else 0
    if n == 0:
        return []
    matrix = [[0.0] * (n + 1) for _ in range(n)]
    for row, target in zip(rows, targets):
        for i in range(n):
            value = row[i]
            if value == 0.0:
                continue
            row_i = matrix[i]
            for j in range(n):
                row_i[j] += value * row[j]
            row_i[n] += value * target
    for i in range(n):
        matrix[i][i] += ridge
    # 高斯-约当消元（带部分主元）
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(matrix[r][column]))
        if abs(matrix[pivot][column]) < 1e-12:
            continue
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        pivot_row = matrix[column]
        pivot_value = pivot_row[column]
        for j in range(column, n + 1):
            pivot_row[j] /= pivot_value
        for r in range(n):
            if r == column:
                continue
            factor = matrix[r][column]
            if factor == 0.0:
                continue
            row_r = matrix[r]
            for j in range(column, n + 1):
                row_r[j] -= factor * pivot_row[j]
    return [matrix[i][n] for i in range(n)]


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
) -> RoomValueModel | None:
    """用生产模拟器当“老师”，闭式拟合房间组合价值模型。

    ``oracle`` 接收一个干员名字元组，返回该组合在模拟器下的房间分数。
    采样策略刻意混合 1 人 / 2 人 / 满编组合：一元项需要单人样本，
    二元项需要成对样本，满编样本给出真实目标。
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

    labels: list[float] = []
    rows: list[list[float]] = []
    dim = encoder.embedding_dim
    extra_width = len(_extra_features(query, 1))
    for combo in combos:
        label = float(oracle(combo))
        labels.append(label)
        vectors = [embedding_by_name[name] for name in combo]
        row = [0.0] * (dim * 2 + extra_width)
        for vector in vectors:
            for channel in range(dim):
                row[channel] += vector[channel]
        offset = dim
        for left in range(len(vectors)):
            left_vector = vectors[left]
            for right in range(left + 1, len(vectors)):
                right_vector = vectors[right]
                for channel in range(dim):
                    row[offset + channel] += left_vector[channel] * right_vector[channel]
        offset += dim
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
        coeffs = [0.0] * (dim * 2 + extra_width)
    else:
        coeffs = _solve_ridge(train_rows, train_labels, ridge / max(1, len(train_rows)))

    def predict(row: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(coeffs, row))

    train_pred = [predict(row) for row in train_rows]
    train_actual = train_labels
    test_rows = [rows[index] for index in range(len(rows)) if index in test_index]
    test_actual = [normalized[index] for index in range(len(rows)) if index in test_index]
    test_pred = [predict(row) for row in test_rows]

    def r2(predicted: Sequence[float], actual: Sequence[float]) -> float:
        mean = sum(actual) / max(1, len(actual))
        ss_tot = sum((value - mean) ** 2 for value in actual)
        if ss_tot <= 0:
            return 0.0
        ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
        return 1.0 - ss_res / ss_tot

    def mae(predicted: Sequence[float], actual: Sequence[float]) -> float:
        if not actual:
            return 0.0
        return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)

    model = RoomValueModel(
        query=query,
        embedding_dim=dim,
        unary=coeffs[:dim],
        pair=coeffs[dim : dim * 2],
        extra=coeffs[dim * 2 :],
        bias=0.0,
        fit_quality={
            "r2Train": 1.0 if degenerate else round(r2(train_pred, train_actual), 4),
            "r2Holdout": 1.0 if degenerate else round(r2(test_pred, test_actual), 4),
            "spearmanHoldout": 0.0 if degenerate else round(_spearman(test_pred, test_actual), 4),
            "maeHoldout": round(mae(test_pred, test_actual) * scale, 4),
            "maeRatio": round(mae(test_pred, test_actual), 6),
            "labelScale": round(scale, 4),
            "labelSpread": round(spread * scale, 6),
            "degenerate": degenerate,
        },
        training={
            "samples": len(rows),
            "trainSamples": len(train_rows),
            "holdoutSamples": len(test_rows),
            "capacity": capacity,
            "candidatePool": len(available),
            "ridge": ridge,
            "seed": seed,
        },
    )
    # 反归一化：让读出层直接输出模拟器量纲的房间分数。
    model.unary = [value * scale for value in model.unary]
    model.pair = [value * scale for value in model.pair]
    model.extra = [value * scale for value in model.extra]
    return model


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

    def unary_value(name: str) -> float:
        value = unary_cache.get(name)
        if value is None:
            vector = embeddings[name]
            value = sum(weight * channel for weight, channel in zip(model.unary[:dim], vector))
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
            value = 0.0
            for index in range(dim):
                value += model.pair[index] * left_vector[index] * right_vector[index]
            pair_cache[key] = value
        return value

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

