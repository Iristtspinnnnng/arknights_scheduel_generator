"""静态干员关系图：把游戏数据里散落的“羁绊 / 增益 / 舱室”信息整理成一张可学习的有向异构图。

这个模块只依赖 ``GameData`` 与练度表，不参与搜索，也不调用生产模拟器。
它的职责是把“旧优化器里散落在十几处 if 判断和硬编码名单中的协同信息”
显式化为节点特征 + 类型化边，供 :mod:`graph_conv` 做消息传递。

图结构
------
* 节点：每个已招募干员一个节点（按名字排序，索引稳定）。
* 边（有向，双向存储）：
  - ``faction``：同 nationId / groupId / teamId（即阵营/势力，一图流里的“羁绊”主体）。
    权重按组规模衰减，避免“罗德岛 94 人”这种大阵营淹没真正的小团体。
  - ``named``：技能文本里点名了另一位干员的**显式羁绊**（如 德克萨斯↔拉普兰德、维什戴尔→伊内丝）。
  - ``combo``：技能文本命中组合机制关键词（``当与`` / ``同一个贸易站`` / ``归零`` 等），
    表示“两人同房间才有联动”，边只在该技能所属房间类型内生效。

所有边权都由数据本身推导，没有随机数，也没有硬编码干员名单——旧优化器里的
``("Tequila", "Butushu", "Proviso", "Shamare")`` 之类名单在这里被替换成
“关系类型”，具体干员由技能文本解析得到。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .data import GameData
from .models import BaseSkill, RosterOperator
from .skill_rules import self_contained_recovery_rate

__all__ = [
    "ROOM_TYPES",
    "TARGETS",
    "RELATIONS",
    "FEATURE_NAMES",
    "ROOM_CATALOG",
    "ROOM_FEATURE_NAMES",
    "ROOM_RELATIONS",
    "OperatorNode",
    "GraphEdge",
    "RoomNode",
    "RoomEdge",
    "StaticOperatorGraph",
    "StaticRoomGraph",
    "build_operator_graph",
    "build_room_graph",
    "graph_signature",
    "room_graph_signature",
]

# 参与优化的工作房间类型（宿舍单独处理：只影响心情，不影响产能）。
ROOM_TYPES: tuple[str, ...] = (
    "CONTROL",
    "TRADING",
    "MANUFACTURE",
    "POWER",
    "MEETING",
    "HIRE",
    "DORMITORY",
)

TARGETS: tuple[str, ...] = ("O_GOLD", "O_DIAMOND", "F_GOLD", "F_EXP", "F_DIAMOND")

RELATIONS: tuple[str, ...] = ("faction", "named", "combo")

# 房间类型 → 参考效率上限，用于把不同量纲的 parsed_score 压到 0~1。
# 这只是归一化常数，不参与评分逻辑；旧优化器的评分仍然由 ProductionSimulator 给出。
_ROOM_SCORE_SCALE: dict[str, float] = {
    "CONTROL": 30.0,
    "TRADING": 45.0,
    "MANUFACTURE": 40.0,
    "POWER": 25.0,
    "MEETING": 30.0,
    "HIRE": 30.0,
    "DORMITORY": 20.0,
}

# 组合机制关键词：出现即说明该技能与“同房间的特定干员”联动。
_COMBO_MARKERS: tuple[str, ...] = (
    "当与",
    "同一个贸易站",
    "同一贸易站",
    "同一个制造站",
    "同一制造站",
    "归零",
    "低语",
    "特别订单",
    "违约订单",
    "高品质贵金属订单",
    "仓库容量",
    "回收利用",
    "大就是好",
    "贸易站内",
    "制造站内",
)

# 阵营别名，和 data.FACTION_TAG_ALIASES 保持一致的方向，但这里只用于建边时去重。
_DERIVED_TAG_PREFIXES = ("nationId:", "groupId:", "teamId:")

FEATURE_NAMES: tuple[str, ...] = (
    "rarity",
    "elite",
    "level",
    "potential",
    "skill_level",
    "mastery_max",
    "room:CONTROL",
    "room:TRADING",
    "room:MANUFACTURE",
    "room:POWER",
    "room:MEETING",
    "room:HIRE",
    "room:DORMITORY",
    "target:O_GOLD",
    "target:O_DIAMOND",
    "target:F_GOLD",
    "target:F_EXP",
    "target:F_DIAMOND",
    "complex_ratio",
    "upgrade_flag",
    "faction_count",
    "named_out_degree",
    "combo_marker",
    "guide_anchor",
)


#: M3b 的房间节点目录：哪些「房型 × 产物」组合真的是一间可选房间。
#:
#: 顺序就是房间节点的稳定下标，也是 ``graph-audit`` 的打印顺序。
#: 这里只放**静态**组合（不含等级/工位/时长）——等级与时长是**查询**属性，
#: 由 ``graph_conv.RoomCondition`` 附在房间嵌入的尾巴上。这样房间节点目录与基建
#: 布局无关，构建一次即可复用，不会因为换了 layout 就让整张图的指纹变化。
ROOM_CATALOG: tuple[tuple[str, str | None], ...] = (
    ("CONTROL", None),
    ("TRADING", "O_GOLD"),
    ("TRADING", "O_DIAMOND"),
    ("MANUFACTURE", "F_GOLD"),
    ("MANUFACTURE", "F_EXP"),
    ("MANUFACTURE", "F_DIAMOND"),
    ("POWER", None),
    ("MEETING", None),
    ("HIRE", None),
    ("DORMITORY", None),
)

#: 房间节点的静态特征（与 :data:`ROOM_CATALOG` 一一对应，不含查询尾巴）。
#:
#: 和干员节点一样，这里的每一项都来自 ``building_data.json``，没有硬编码数值。
ROOM_FEATURE_NAMES: tuple[str, ...] = (
    "room:CONTROL",
    "room:TRADING",
    "room:MANUFACTURE",
    "room:POWER",
    "room:MEETING",
    "room:HIRE",
    "room:DORMITORY",
    "target:O_GOLD",
    "target:O_DIAMOND",
    "target:F_GOLD",
    "target:F_EXP",
    "target:F_DIAMOND",
    "capacity",
    "powerPerSlot",
    "isPowerRoom",
    "isResourceRoom",
)

#: 房间节点之间的关系类型：发电站供电、控制中枢加成。
ROOM_RELATIONS: tuple[str, ...] = ("power", "control")


@dataclass(frozen=True)
class RoomNode:
    """图上的一个房间节点（房型 + 产物）。``features`` 与 :data:`ROOM_FEATURE_NAMES` 对齐。"""

    room_type: str
    target: str | None
    features: tuple[float, ...]

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.room_type, self.target)

    @property
    def label(self) -> str:
        return self.room_type if self.target is None else f"{self.room_type}/{self.target}"


@dataclass(frozen=True)
class RoomEdge:
    """房间节点之间的关系边（无向语义，双向存储）。"""

    src: int
    dst: int
    relation: str
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "weight": round(self.weight, 6),
        }


@dataclass
class StaticRoomGraph:
    """房间节点图（M3b）。

    它**不依赖基建布局**：节点目录是 :data:`ROOM_CATALOG`，特征是 ``building_data``
    的房型常量（最大工位、每工位耗电）。等级 / 工位 / 时长在查询时拼到房间嵌入的
    尾巴上（``graph_conv.RoomCondition``），所以同一张表可以给任意「房间 × 干员」对打分。
    """

    nodes: list[RoomNode]
    edges: list[RoomEdge]
    index_by_key: dict[tuple[str, str | None], int]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.nodes)

    def index(self, room_type: str, target: str | None) -> int | None:
        return self.index_by_key.get((room_type, target))

    def node(self, room_type: str, target: str | None) -> RoomNode | None:
        idx = self.index_by_key.get((room_type, target))
        return None if idx is None else self.nodes[idx]

    def neighbours(self, room_type: str, target: str | None) -> list[tuple[RoomNode, str, float]]:
        """该房间节点的**入边**邻居（发电站 / 控制中枢），即消息传递时聚合的对象。"""
        idx = self.index_by_key.get((room_type, target))
        if idx is None:
            return []
        return [
            (self.nodes[edge.src], edge.relation, edge.weight)
            for edge in self.edges
            if edge.dst == idx
        ]

    def edge_count_by_relation(self) -> dict[str, int]:
        counts = {relation: 0 for relation in ROOM_RELATIONS}
        for edge in self.edges:
            counts[edge.relation] = counts.get(edge.relation, 0) + 1
        return counts

    def audit_report(self) -> dict[str, Any]:
        return {
            "nodeCount": len(self.nodes),
            "featureWidth": len(ROOM_FEATURE_NAMES),
            "rooms": [
                {
                    "room": node.label,
                    "capacity": round(node.features[ROOM_FEATURE_NAMES.index("capacity")] * 5.0, 3),
                    "powerPerSlot": round(
                        node.features[ROOM_FEATURE_NAMES.index("powerPerSlot")] * 60.0, 3
                    ),
                }
                for node in self.nodes
            ],
            "edgeCountByRelation": self.edge_count_by_relation(),
            "edges": [edge.as_dict() for edge in self.edges],
            "signature": self.meta.get("signature"),
            "dataVersion": self.meta.get("dataVersion"),
        }

    def to_dict(self, *, limit: int = 40) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "nodes": [
                {
                    "roomType": node.room_type,
                    "target": node.target,
                    "features": list(node.features),
                }
                for node in self.nodes[:limit]
            ],
            "edgeCount": len(self.edges),
        }


@dataclass(frozen=True)
class OperatorNode:
    """图上的一个干员节点。``features`` 与 :data:`FEATURE_NAMES` 一一对应。"""

    name: str
    char_id: str
    features: tuple[float, ...]
    skills: tuple[BaseSkill, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    src: int
    dst: int
    relation: str
    weight: float
    room_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "weight": round(self.weight, 6),
            "roomType": self.room_type,
        }


@dataclass
class StaticOperatorGraph:
    """构建完成的静态图。所有索引都是按名字排序后的稳定下标。"""

    nodes: list[OperatorNode]
    edges: list[GraphEdge]
    index_by_name: dict[str, int]
    faction_groups: dict[str, tuple[int, ...]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    _index: dict[tuple[int, str], tuple[tuple[int, float], ...]] | None = field(
        default=None, repr=False, compare=False
    )

    # ---- 便捷访问 -------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def names(self) -> list[str]:
        return [node.name for node in self.nodes]

    def index(self, name: str) -> int | None:
        return self.index_by_name.get(name)

    def node(self, name: str) -> OperatorNode | None:
        idx = self.index_by_name.get(name)
        return None if idx is None else self.nodes[idx]

    def adjacency(self) -> dict[str, list[list[tuple[int, float, str | None]]]]:
        """返回 ``relation -> 邻接表``，每项是 ``(邻居下标, 权重, room_type)``。"""
        adj: dict[str, list[list[tuple[int, float, str | None]]]] = {
            relation: [[] for _ in range(len(self.nodes))] for relation in RELATIONS
        }
        for edge in self.edges:
            adj[edge.relation][edge.src].append((edge.dst, edge.weight, edge.room_type))
        return adj

    def edge_count_by_relation(self) -> dict[str, int]:
        counts = {relation: 0 for relation in RELATIONS}
        for edge in self.edges:
            counts[edge.relation] = counts.get(edge.relation, 0) + 1
        return counts

    def _neighbour_index(self) -> dict[tuple[int, str], tuple[tuple[int, float], ...]]:
        """按 (源节点, 关系) 预聚合的邻接索引。

        ``neighbors()`` 会在一次 ``optimize()`` 里被调用上万次（候选池闭包），
        每次都线性扫全表会把 ``O(E)`` 乘进去，所以这里建一次索引。
        """
        if self._index is not None:
            return self._index
        buckets: dict[tuple[int, str], list[tuple[int, float]]] = {}
        for edge in self.edges:
            buckets.setdefault((edge.src, edge.relation), []).append((edge.dst, edge.weight))
        self._index = {
            key: tuple(sorted(values, key=lambda item: item[1], reverse=True))
            for key, values in buckets.items()
        }
        return self._index

    def neighbors(
        self, name: str, relation: str, *, min_weight: float = 0.0
    ) -> list[tuple[str, float]]:
        idx = self.index_by_name.get(name)
        if idx is None:
            return []
        entries = self._neighbour_index().get((idx, relation), ())
        return [
            (self.nodes[dst].name, weight)
            for dst, weight in entries
            if weight >= min_weight
        ]

    def audit_report(self, *, limit: int = 12) -> dict[str, Any]:
        """回答“这个仓库里的静态图信息够不够”的体检报告。"""
        relation_counts = self.edge_count_by_relation()
        degree_histogram: dict[str, int] = {}
        degree = [0] * len(self.nodes)
        for edge in self.edges:
            if edge.relation in {"faction", "named", "combo"}:
                degree[edge.src] += 1
        for value in degree:
            key = str(min(value, 20))
            degree_histogram[key] = degree_histogram.get(key, 0) + 1
        named_pairs = sorted(
            {
                tuple(sorted((self.nodes[edge.src].name, self.nodes[edge.dst].name)))
                for edge in self.edges
                if edge.relation == "named"
            }
        )
        faction_sizes = sorted(
            ({"tag": tag, "size": len(members)} for tag, members in self.faction_groups.items()),
            key=lambda item: item["size"],
            reverse=True,
        )
        isolated = [
            self.nodes[idx].name
            for idx in range(len(self.nodes))
            if degree[idx] == 0
        ]
        return {
            "nodeCount": len(self.nodes),
            "featureWidth": len(FEATURE_NAMES),
            "edgeCountByRelation": relation_counts,
            "edgeCountTotal": sum(relation_counts.values()),
            "factionGroupCount": len(self.faction_groups),
            "factionGroupsTop": faction_sizes[:limit],
            "namedDependencyPairCount": len(named_pairs),
            "namedDependencyPairs": [f"{a} <-> {b}" for a, b in named_pairs[:limit]],
            "degreeHistogram": dict(sorted(degree_histogram.items(), key=lambda kv: int(kv[0]))),
            "isolatedOperatorCount": len(isolated),
            "isolatedOperators": isolated[:limit],
            "specialMoraleNodeCount": len(self.meta.get("specialMoraleNodes") or []),
            "specialMoraleNodes": list(self.meta.get("specialMoraleNodes") or [])[:limit],
            "signature": self.meta.get("signature"),
            "dataVersion": self.meta.get("dataVersion"),
            "rosterSize": self.meta.get("rosterSize"),
        }

    def to_dict(self, *, include_edges: bool = False, limit: int = 200) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "meta": self.meta,
            "nodes": [
                {"name": node.name, "charId": node.char_id, "features": list(node.features)}
                for node in self.nodes[:limit]
            ],
            "edgeCount": len(self.edges),
        }
        if include_edges:
            payload["edges"] = [edge.as_dict() for edge in self.edges]
        return payload


def graph_signature(
    nodes: Sequence[OperatorNode],
    edges: Sequence[GraphEdge],
    meta: Mapping[str, Any] | None = None,
) -> str:
    """图内容的稳定指纹；用于给训练好的读出层做缓存键。"""
    digest = hashlib.sha256()
    for node in nodes:
        digest.update(node.name.encode("utf-8"))
        digest.update(b"|")
        digest.update(",".join(f"{value:.4f}" for value in node.features).encode("ascii"))
        digest.update(b";")
    for edge in edges:
        digest.update(
            f"{edge.relation}:{edge.src}->{edge.dst}:{edge.weight:.4f}:{edge.room_type};".encode("utf-8")
        )
    if meta:
        digest.update(json.dumps(meta, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return digest.hexdigest()[:32]


def room_graph_signature(
    nodes: Sequence[RoomNode],
    edges: Sequence[RoomEdge],
    meta: Mapping[str, Any] | None = None,
) -> str:
    """房间图的稳定指纹（给共享亲和度矩阵做缓存键）。"""
    digest = hashlib.sha256()
    for name in ROOM_FEATURE_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(b",")
    digest.update(b";")
    for node in nodes:
        digest.update(f"{node.room_type}|{node.target}|".encode("utf-8"))
        digest.update(",".join(f"{value:.4f}" for value in node.features).encode("ascii"))
        digest.update(b";")
    for edge in edges:
        digest.update(
            f"{edge.relation}:{edge.src}->{edge.dst}:{edge.weight:.4f};".encode("utf-8")
        )
    if meta:
        digest.update(json.dumps(meta, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return digest.hexdigest()[:32]


def build_room_graph(game_data: GameData) -> StaticRoomGraph:
    """从 ``building_data.json`` 构建房间节点图。

    特征全部来自数据：

    * ``capacity`` = 三级 ``maxStationedNum`` / 5；
    * ``powerPerSlot`` = 三级 ``electricity`` / 工位 / 60（发电站为正、耗电房为负）；
    * ``isPowerRoom`` / ``isResourceRoom`` 区分「供电」「产资源」「功能房」。

    边只表达两件事：**发电站给它供电**（``power``，权重 = 每工位耗电）与
    **控制中枢给它加成**（``control``，中枢的 ``basicCostBuff`` 对所有设施生效）。
    这两条正是旧优化器里散落在 ``power.py`` / ``morale.py`` 的 if 判断。
    """
    rooms = game_data.building.get("rooms") or {}
    control_buff = float((game_data.building.get("controlData") or {}).get("basicCostBuff") or 0)
    nodes: list[RoomNode] = []
    index_by_key: dict[tuple[str, str | None], int] = {}
    for room_type, target in ROOM_CATALOG:
        record = rooms.get(room_type) or {}
        phases = record.get("phases") or []
        top = phases[-1] if phases else {}
        capacity = float(top.get("maxStationedNum") or 1)
        electricity = float(top.get("electricity") or 0.0)
        power_per_slot = electricity / max(1.0, capacity) / 60.0
        features: list[float] = [0.0] * len(ROOM_FEATURE_NAMES)
        index = {name: position for position, name in enumerate(ROOM_FEATURE_NAMES)}
        room_feature = f"room:{room_type}"
        if room_feature in index:
            features[index[room_feature]] = 1.0
        if target is not None and f"target:{target}" in index:
            features[index[f"target:{target}"]] = 1.0
        features[index["capacity"]] = capacity / 5.0
        features[index["powerPerSlot"]] = power_per_slot
        features[index["isPowerRoom"]] = 1.0 if room_type == "POWER" else 0.0
        features[index["isResourceRoom"]] = (
            1.0 if room_type in {"TRADING", "MANUFACTURE"} else 0.0
        )
        index_by_key[(room_type, target)] = len(nodes)
        nodes.append(RoomNode(room_type, target, tuple(features)))

    edges: list[RoomEdge] = []

    def add_edge(src: int, dst: int, relation: str, weight: float) -> None:
        if src == dst or weight <= 0:
            return
        edges.append(RoomEdge(src, dst, relation, float(weight)))

    power_index = index_by_key.get(("POWER", None))
    control_index = index_by_key.get(("CONTROL", None))
    capacity_position = ROOM_FEATURE_NAMES.index("capacity")
    power_position = ROOM_FEATURE_NAMES.index("powerPerSlot")
    for node_index, node in enumerate(nodes):
        if node.room_type == "POWER":
            continue
        draw = abs(node.features[power_position]) * max(0.2, node.features[capacity_position])
        if power_index is not None and draw > 0:
            add_edge(power_index, node_index, "power", draw)
        if control_index is not None and control_buff < 0:
            # 中枢减耗对所有在岗干员生效（`morale` 侧的基准规则），所以是「中枢 → 每个房间」。
            add_edge(control_index, node_index, "control", 1.0)
    meta: dict[str, Any] = {
        "dataVersion": game_data.data_version,
        "catalog": [f"{room}" + ("" if target is None else f"/{target}") for room, target in ROOM_CATALOG],
        "featureNames": list(ROOM_FEATURE_NAMES),
        "controlDataBasicCostBuff": control_buff,
    }
    meta["signature"] = room_graph_signature(nodes, edges, {"dataVersion": game_data.data_version})
    return StaticRoomGraph(
        nodes=nodes, edges=edges, index_by_key=index_by_key, meta=meta
    )


def build_operator_graph(
    game_data: GameData,
    roster: Iterable[RosterOperator],
    *,
    allow_upgrades: bool = False,
    max_faction_group_size: int = 60,
) -> StaticOperatorGraph:
    """从游戏数据 + 练度表构建静态干员关系图。

    参数
    ----
    max_faction_group_size:
        超过这个规模的阵营（如 ``rhodes`` 94 人）不建边——它对排班没有区分度，
        只会让所有人互相连通、稀释小团体的信号。
    """
    operators_all = [operator for operator in roster if operator.recruited]
    operators, duplicate_names = _dedupe_roster(operators_all)
    operators.sort(key=lambda operator: operator.name)
    if not operators:
        return StaticOperatorGraph(nodes=[], edges=[], index_by_name={}, meta={"rosterSize": 0})

    skills_by_name: dict[str, tuple[BaseSkill, ...]] = {
        operator.name: tuple(game_data.skills_for_roster_operator(operator, allow_upgrades))
        for operator in operators
    }
    names = [operator.name for operator in operators]
    index_by_name = {name: position for position, name in enumerate(names)}

    # ---- 解析技能文本里的显式羁绊与组合机制 -----------------------------
    named_pairs: dict[tuple[str, str], float] = {}
    combo_member_by_room: dict[str, set[str]] = {room: set() for room in ROOM_TYPES}
    combo_marker_flag: dict[str, float] = {name: 0.0 for name in names}
    roster_name_set = set(names)
    for name, skills in skills_by_name.items():
        for skill in skills:
            text = f"{skill.buff_name} {skill.description}"
            if any(marker in text for marker in _COMBO_MARKERS):
                combo_marker_flag[name] = 1.0
                if skill.room_type in ROOM_TYPES:
                    combo_member_by_room[skill.room_type].add(name)
            for partner in _mentioned_names(text, roster_name_set, exclude=name):
                pair = tuple(sorted((name, partner)))
                named_pairs[pair] = max(named_pairs.get(pair, 0.0), 1.5)

    # ---- 阵营分组 -------------------------------------------------------
    tag_members: dict[str, list[str]] = {}
    tags_by_name: dict[str, list[str]] = {}
    for operator in operators:
        char_id = game_data.resolve_roster_char_id(operator.name) or ""
        tags = [
            tag
            for tag in game_data.faction_tags_for_char(char_id)
            if not any(tag.startswith(prefix) for prefix in _DERIVED_TAG_PREFIXES)
        ]
        tags_by_name[operator.name] = tags
        for tag in tags:
            tag_members.setdefault(tag, []).append(operator.name)

    faction_groups = {
        tag: tuple(sorted(members))
        for tag, members in tag_members.items()
        if 1 < len(members) <= max_faction_group_size
    }

    # ---- 边 -------------------------------------------------------------
    edges: list[GraphEdge] = []
    relation_weight_totals: dict[str, float] = {}

    def add_edge(src: str, dst: str, relation: str, weight: float, room_type: str | None = None) -> None:
        if src == dst or weight <= 0:
            return
        src_idx = index_by_name[src]
        dst_idx = index_by_name[dst]
        edges.append(GraphEdge(src_idx, dst_idx, relation, float(weight), room_type))
        relation_weight_totals[relation] = relation_weight_totals.get(relation, 0.0) + weight

    # faction：组内两两相连，权重按组规模衰减（小团体 = 强羁绊）
    for tag, members in faction_groups.items():
        decay = 1.0 / math.log2(2.0 + len(members))
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                add_edge(left, right, "faction", decay)
                add_edge(right, left, "faction", decay)

    # named：技能文本点名的显式羁绊（双向）
    for (left, right), weight in named_pairs.items():
        add_edge(left, right, "named", weight)
        add_edge(right, left, "named", weight)

    # combo：同房间类型下带组合机制的干员互相弱连接，让 GCN 能把“同房间联动”
    # 传导开；真正的强度由读出层从模拟器标签里学。
    combo_by_room = {
        room_type: sorted(members)
        for room_type, members in combo_member_by_room.items()
    }
    for room_type, members in combo_by_room.items():
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                add_edge(left, right, "combo", 0.4, room_type)
                add_edge(right, left, "combo", 0.4, room_type)

    # ---- 节点特征 -------------------------------------------------------
    named_out: dict[str, int] = {name: 0 for name in names}
    for (left, right) in named_pairs:
        named_out[left] += 1
        named_out[right] += 1
    guide_anchor_names = _guide_anchor_names()

    nodes: list[OperatorNode] = []
    for operator in operators:
        skills = skills_by_name[operator.name]
        features = _operator_features(
            operator,
            skills,
            faction_count=len(tags_by_name.get(operator.name, [])),
            named_out_degree=named_out.get(operator.name, 0),
            combo_marker=combo_marker_flag.get(operator.name, 0.0),
            guide_anchor=1.0 if operator.name in guide_anchor_names else 0.0,
        )
        nodes.append(
            OperatorNode(
                name=operator.name,
                char_id=game_data.resolve_roster_char_id(operator.name) or "",
                features=features,
                skills=skills,
            )
        )

    meta: dict[str, Any] = {
        "dataVersion": game_data.data_version,
        "rosterSize": len(nodes),
        "rosterRows": len(operators_all),
        "duplicateRosterNames": duplicate_names,
        "relationWeightTotals": {key: round(value, 3) for key, value in relation_weight_totals.items()},
        "comboMembersByRoom": {room: len(members) for room, members in combo_by_room.items()},
        "maxFactionGroupSize": max_faction_group_size,
        "featureNames": list(FEATURE_NAMES),
        # 「自律」这类「无法获得其他来源的心情恢复」的干员按**自给自足的独立节点**处理：
        # 宿舍等级、氛围、舍友技能、中枢加成都不接到它身上（`morale` 侧同一条规则）。
        # 这里只做标注，不加 GCN 特征——它对产能读出层没有梯度，只会变成噪声。
        "specialMoraleNodes": sorted(
            name
            for name, skills in skills_by_name.items()
            if any(
                self_contained_recovery_rate(skill.description) is not None
                for skill in skills
            )
        ),
    }
    signature = graph_signature(nodes, edges, {"dataVersion": game_data.data_version})
    meta["signature"] = signature
    return StaticOperatorGraph(
        nodes=nodes,
        edges=edges,
        index_by_name=index_by_name,
        faction_groups=faction_groups,
        meta=meta,
    )


def _operator_features(
    operator: RosterOperator,
    skills: Sequence[BaseSkill],
    *,
    faction_count: int,
    named_out_degree: int,
    combo_marker: float,
    guide_anchor: float,
) -> tuple[float, ...]:
    room_strength = {room: 0.0 for room in ROOM_TYPES}
    target_strength = {target: 0.0 for target in TARGETS}
    complex_count = 0
    for skill in skills:
        room = skill.room_type
        if room in room_strength:
            room_strength[room] = max(room_strength[room], float(skill.parsed_score or 0.0))
        for target in skill.targets:
            key = str(target)
            if key in target_strength:
                target_strength[key] = max(
                    target_strength[key], float(skill.parsed_score or 0.0)
                )
        if skill.complex_condition:
            complex_count += 1

    def squash(value: float, scale: float) -> float:
        if scale <= 0:
            return 0.0
        return max(0.0, min(1.0, value / scale))

    values: list[float] = [
        squash(operator.rarity, 6.0),
        squash(operator.elite, 2.0),
        squash(operator.level, 90.0),
        squash(operator.potential or 0, 5.0),
        squash(operator.skill_level, 7.0),
        squash(max(operator.masteries or (0, 0, 0)), 3.0),
    ]
    values.extend(squash(room_strength[room], _ROOM_SCORE_SCALE[room]) for room in ROOM_TYPES)
    values.extend(squash(target_strength[target], 45.0) for target in TARGETS)
    values.extend(
        [
            squash(complex_count, max(1, len(skills))),
            1.0 if any(skill.upgrade is not None for skill in skills) else 0.0,
            squash(faction_count, 4.0),
            squash(named_out_degree, 3.0),
            1.0 if combo_marker else 0.0,
            float(guide_anchor),
        ]
    )
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(
            f"feature width mismatch: {len(values)} != {len(FEATURE_NAMES)}"
        )
    return tuple(values)


def _dedupe_roster(
    operators: Sequence[RosterOperator],
) -> tuple[list[RosterOperator], dict[str, int]]:
    """同名干员去重（练度表里可能出现 ``Sharp`` 这类重复行）。

    保留练度最高的那一行，其余行只计入 ``meta.duplicateRosterNames``，
    避免图里出现两个同名节点导致索引歧义。
    """
    best: dict[str, RosterOperator] = {}
    counts: dict[str, int] = {}
    for operator in operators:
        counts[operator.name] = counts.get(operator.name, 0) + 1
        current = best.get(operator.name)
        if current is None or _progression_key(operator) > _progression_key(current):
            best[operator.name] = operator
    duplicates = {name: count for name, count in counts.items() if count > 1}
    return list(best.values()), dict(sorted(duplicates.items()))


def _progression_key(operator: RosterOperator) -> tuple[int, int, int]:
    return (operator.elite, operator.level, operator.rarity)


def _mentioned_names(text: str, roster_names: set[str], *, exclude: str) -> list[str]:
    from .dependency_parser import mentioned_operator_names

    return [
        name
        for name in mentioned_operator_names(text, roster_names)
        if name != exclude
    ]


def _has_combo_marker(skill: BaseSkill) -> bool:
    text = f"{skill.buff_name} {skill.description}"
    return any(marker in text for marker in _COMBO_MARKERS)

def _guide_anchor_names() -> set[str]:
    from .production import GUIDE_MANUFACTURE_LOOKUPS, GUIDE_SPECIAL_TRADE_ANCHORS

    names = set(GUIDE_SPECIAL_TRADE_ANCHORS)
    for lookup in GUIDE_MANUFACTURE_LOOKUPS:
        names.update(lookup.get("required", set()))
    return names
