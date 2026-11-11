"""把「羁绊加成」显式写成读入层的基函数。

图卷积只提供**结构**（谁和谁有关系），数值强度全靠 47 维嵌入去凑。
但游戏里真正生效的羁绊只有四种形态（见 ``docs/graph_optimizer.md`` §1.1），
其中两种可以写成**可加/可配对**的显式基：

* **阵营计数型**：``+20 × 本房间 glasgow 人数``。对同房间干员可加，
  所以「房间内该阵营人数」= Σ 干员指示函数 → 完全可以塞进一元项。
* **点名羁绊型**：``当与X在同一个贸易站时…``。同样是两人同房才有用，
  所以塞进二元项。

这样读出层就不再只是黑箱：它会学出「每多一名 glasgow 干员值多少分」
「一对有羁绊的干员同房值多少分」这种可以直接读出来、也可以写进报告的系数。
阶跃型的 guide 刊例行无法用这种线性基表达，仍然走「机制超边」注入
（见 :mod:`graph_optimizer`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .graph_features import StaticOperatorGraph

__all__ = [
    "IN_ROOM_COUNT_TAGS",
    "MECHANIC_FACTION_TAGS",
    "in_room_count_tags",
    "BondBasis",
    "build_bond_basis",
]

# 游戏内真正被技能/中枢机制引用的阵营标签（从 skill_rules.py 与 production.py 提取）。
MECHANIC_FACTION_TAGS: tuple[str, ...] = (
    "A1",
    "blacksteel",
    "glasgow",
    "karlan",
    "knight",
    "laterano",
    "lgd",
    "mh2",
    "minos",
    "op",
    "pinus",
    "rhine",
    "sargon",
    "siracusa",
    "ussg",
)

# 哪些「房间 × 产物」的机制是「**本房间**人数越多越强」——只有这类才适合做房间级显式基。
#
# 反例：黑钢（``+2 × min(全基建黑钢人数, 3)``）和莱茵（``+3 × min(全基建莱茵人数, 5)``）
# 用的是**全基建**计数，机械精通用的是**发电站**计数。把它们塞进房间级一元项，
# 就等于给一个与房间无关的量配上一个房间内的特征，只会加噪声
# （实测会让制造站 holdout R² 从 0.75 掉到 0.70）。
#
# 同时也必须按产物区分：``重聚时光``/``情同手足`` 只对 F_EXP 生效，
# 在 F_GOLD / F_DIAMOND 房间里这两列纯属噪声。
IN_ROOM_COUNT_TAGS: dict[tuple[str, str | None], tuple[str, ...]] = {
    ("TRADING", "O_GOLD"): ("glasgow", "laterano", "mh2"),
    ("TRADING", "O_DIAMOND"): (),
    ("MANUFACTURE", "F_EXP"): ("A1", "ussg"),
    ("MANUFACTURE", "F_GOLD"): (),
    ("MANUFACTURE", "F_DIAMOND"): (),
    ("POWER", None): ("op", "laterano"),
}


def in_room_count_tags(room_type: str, target: str | None) -> tuple[str, ...]:
    """该房间/产物组合下，语义上成立的「本房间阵营计数」标签。"""
    return IN_ROOM_COUNT_TAGS.get((room_type, target), ())

# 二元项里的三个显式指示：a 点名 b、b 点名 a、同房型组合机制。
PAIR_BOND_FEATURES: tuple[str, ...] = (
    "bond:A_mentions_B",
    "bond:B_mentions_A",
    "bond:combo_pair",
)


@dataclass(frozen=True)
class BondBasis:
    """一次房间规格下的显式羁绊基。

    ``tags`` 决定一元项的额外宽度，``named`` / ``combo`` 决定二元项的额外宽度。
    """

    tags: tuple[str, ...]
    members: Mapping[str, frozenset[str]]
    named: frozenset[tuple[str, str]]
    combo: frozenset[tuple[str, str]]
    room_type: str

    @property
    def unary_width(self) -> int:
        return len(self.tags)

    @property
    def pair_width(self) -> int:
        return len(PAIR_BOND_FEATURES)

    def feature_names(self) -> list[str]:
        return [f"faction_count:{tag}" for tag in self.tags] + list(PAIR_BOND_FEATURES)

    def unary_features(self, name: str) -> tuple[float, ...]:
        return tuple(1.0 if name in self.members[tag] else 0.0 for tag in self.tags)

    def pair_features(self, left: str, right: str) -> tuple[float, ...]:
        return (
            1.0 if (left, right) in self.named else 0.0,
            1.0 if (right, left) in self.named else 0.0,
            1.0 if _ordered(left, right) in self.combo else 0.0,
        )

    def describe(self, unary_weights: Sequence[float], pair_weights: Sequence[float]) -> dict:
        """把学到的系数翻成人话，用于报告。"""
        base_u = len(unary_weights) - self.unary_width
        base_p = len(pair_weights) - self.pair_width
        factions = [
            {"tag": tag, "members": len(self.members[tag]), "coefficientPerOperator": round(float(unary_weights[base_u + index]), 4)}
            for index, tag in enumerate(self.tags)
        ]
        factions.sort(key=lambda item: abs(item["coefficientPerOperator"]), reverse=True)
        pairs = [
            {
                "feature": name,
                "coefficient": round(float(pair_weights[base_p + index]), 4),
            }
            for index, name in enumerate(PAIR_BOND_FEATURES)
        ]
        return {
            "roomType": self.room_type,
            "factionCounts": factions,
            "pairFeatures": pairs,
            "namedPairsAvailable": len(self.named),
            "comboPairsAvailable": len(self.combo),
        }

    def to_dict(self) -> dict:
        return {
            "roomType": self.room_type,
            "tags": list(self.tags),
            "members": {tag: sorted(names) for tag, names in self.members.items()},
            "named": [list(pair) for pair in sorted(self.named)],
            "combo": [list(pair) for pair in sorted(self.combo)],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "BondBasis":
        return cls(
            tags=tuple(payload["tags"]),
            members={tag: frozenset(names) for tag, names in payload["members"].items()},
            named=frozenset(tuple(pair) for pair in payload["named"]),
            combo=frozenset(tuple(pair) for pair in payload["combo"]),
            room_type=str(payload["roomType"]),
        )


def _ordered(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def build_bond_basis(
    graph: StaticOperatorGraph,
    *,
    room_type: str,
    candidate_names: Iterable[str],
    target: str | None = None,
    max_tags: int = 24,
    include_tight_groups: bool | None = None,
    max_tight_group_size: int = 8,
    min_tight_group_size: int = 2,
) -> BondBasis:
    """为某个房间规格构造显式羁绊基。

    标签选择规则（都可解释，不含随机）：

    1. 先用**本房间计数**语义成立的阵营标签（:data:`IN_ROOM_COUNT_TAGS`）——
       只有这些做房间级一元项才是对的；
    2. 若 ``include_tight_groups`` 为真，再补上「小团体」（2~8 人的阵营，
       如 ``glasgow`` 5 人），它们往往是真正的羁绊组；
       默认只在贸易站开启，因为贸易站的主力机制就是阵营计数；
    3. 只保留在当前候选池里至少有 2 名成员的标签，避免学出全是 0 的系数；
    4. 别名标签（成员集合完全相同）只保留一个，避免完全共线。
    """
    candidates = set(candidate_names)
    members: dict[str, frozenset[str]] = {}
    for tag, names in graph.faction_groups.items():
        present = frozenset(name for name in names if name in candidates)
        if len(present) >= 2:
            members[tag] = present

    if include_tight_groups is None:
        include_tight_groups = room_type == "TRADING" and target == "O_GOLD"

    ordered_tags: list[str] = []
    for tag in in_room_count_tags(room_type, target):
        if tag in members and tag not in ordered_tags:
            ordered_tags.append(tag)
    if include_tight_groups:
        tight = sorted(
            (
                tag
                for tag, names in members.items()
                if min_tight_group_size <= len(names) <= max_tight_group_size
            ),
            key=lambda tag: (len(members[tag]), tag),
        )
        for tag in tight:
            if tag not in ordered_tags:
                ordered_tags.append(tag)

    # 别名标签（rhine/rh、blacksteel/bs、knight/kazimierz…）的成员集合完全相同，
    # 保留几列就会形成完全共线，岭回归下既不稳定也没法解释；这里按成员集合去重，
    # 优先保留游戏机制里真正引用的那个名字。
    deduped: list[str] = []
    seen_signatures: set[frozenset[str]] = set()
    for tag in ordered_tags:
        signature = members[tag]
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(tag)
    ordered_tags = deduped[:max_tags]

    named: set[tuple[str, str]] = set()
    combo: set[tuple[str, str]] = set()
    for edge in graph.edges:
        if edge.relation == "named":
            src = graph.nodes[edge.src].name
            dst = graph.nodes[edge.dst].name
            if src in candidates and dst in candidates:
                named.add((src, dst))
        elif edge.relation == "combo" and edge.room_type == room_type:
            src = graph.nodes[edge.src].name
            dst = graph.nodes[edge.dst].name
            if src in candidates and dst in candidates:
                combo.add(_ordered(src, dst))

    return BondBasis(
        tags=tuple(ordered_tags),
        members={tag: members[tag] for tag in ordered_tags},
        named=frozenset(named),
        combo=frozenset(combo),
        room_type=room_type,
    )
