"""检查 generate 导出 JSON 的心情门槛读数（M1 验收第 5 条）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for layout in ("243", "333", "342"):
    payload = json.loads(Path(f"outputs/_gen_{layout}.json").read_text(encoding="utf-8"))
    analysis = payload.get("analysis") or {}
    audit = analysis.get("moraleCycleAudit") or {}
    ledger = audit.get("moraleLedgerByOperator") or {}
    worst = min((item["balance"] for item in ledger.values()), default=None)
    levels = analysis.get("layout", {}).get("dormitoryLevels")
    shifts = payload.get("shifts") or []
    dorm_plans = sum(1 for shift in shifts if shift.get("dormitories"))
    print(
        f"{layout}: score={payload.get('score')} cycleHours={audit.get('cycleHours')} "
        f"passed={audit.get('hardGatePassed')} shifts={len(shifts)} "
        f"dormPlans={dorm_plans} worstBalance={worst} levels={levels}"
    )
sys.exit(0)
