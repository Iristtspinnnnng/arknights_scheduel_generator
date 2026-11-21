"""图价值模型与机制超边的磁盘缓存。

为什么要持久化：读出层要对每个「房间规格 × 时长」各拟合一次（288 次模拟器打分 +
一次闭式求解）。这些数值只取决于游戏数据、练度表和固定超参，跨进程完全一致，
所以第一次算完就可以落盘复用。``recommend`` 的进程池路径每个候选都会新建优化器，
没有这层缓存时每个候选都要重算一遍。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["GraphModelCache", "CACHE_SCHEMA_VERSION", "DEFAULT_CACHE_DIR"]

# 4：2026-11 价格口径修订（唯一价格表 + 碎片硬约束）改变了 room_output_score，
#    也就是读出层的**标签**；旧缓存里的模型与真实标签不再一致，靠版本号整体作废。
# 3：O_DIAMOND 校准 v2（速度参与、不裁剪订单上限）改变了这部分房间的产能标签。
CACHE_SCHEMA_VERSION = 4
DEFAULT_CACHE_DIR = Path("outputs") / "graph_model_cache"


@dataclass
class GraphModelCache:
    """极简 JSON 缓存：键是内容哈希，命中即跳过拟合。"""

    directory: Path | None = None
    enabled: bool = True
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def __post_init__(self) -> None:
        if not self.enabled:
            self.directory = None
            return
        if self.directory is None:
            self.directory = DEFAULT_CACHE_DIR
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            # 只读工作区或权限不足时静默退化成不使用缓存，不能影响排班生成。
            self.enabled = False
            self.directory = None

    @property
    def active(self) -> bool:
        return self.enabled and self.directory is not None

    def key(self, payload: dict[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        return digest.hexdigest()[:40]

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        if not self.active:
            return None
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.misses += 1
            return None
        if payload.get("schema") != CACHE_SCHEMA_VERSION:
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def save(self, key: str, payload: dict[str, Any]) -> None:
        if not self.active:
            return
        body = {"schema": CACHE_SCHEMA_VERSION, **payload}
        path = self._path(key)
        try:
            path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            self.writes += 1
        except OSError:
            self.enabled = False
            self.directory = None

    def profile(self) -> dict[str, Any]:
        return {
            "enabled": self.active,
            "directory": str(self.directory) if self.directory else None,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }
