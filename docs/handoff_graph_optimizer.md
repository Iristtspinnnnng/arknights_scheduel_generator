# 接手文档：图卷积优化器 + 心情/宿舍模型

> 写于上下文耗尽前。下一个窗口从这里开始，不必回读聊天记录。
> 最后更新：第 1 步（修 `morale.audit_morale_cycle` 判定）已完成，回归测试在 `tests/test_morale.py`。

---

## 0. 一分钟速览

- 分支 `main`，**领先 `origin/main` 2 个提交，未 push**：`8213fb3`（图卷积优化器首版）、
  `c2d39ba`（社区打分权重 + 显式羁绊 + numpy + 缓存 + 多起手）。
- **§3 第 1 步已完成**（修心情判定）：判据改成「每个干员一周期收支记账」，
  `morale.py` 审计 schema 升到 `version: 2`，新增回归测试 `tests/test_morale.py`（11 个）。
  细节见 §3 第 1 步与 P10。
- 另有多个文件未提交（见 §7）。
- 安全备份分支：`backup/pre-rebase-graph`（rebase 前的原始状态）。
- 测试：**239 passed / 1 skipped / 1 failed**（含新增的 `tests/test_morale.py` 11 个）。唯一失败是既有环境问题
  （`test_calibration_report_examples_pass` 需要仓库外的 `C:\tmp\yituliu_schedule_images_current.js`，
  报 `missing_local_asset`），与本项目改动无关，不要去"修"它。
- 环境：系统 `python`（WindowsApps，3.13）**没有 pytest**，不能直接跑测试；本轮实际用的是 DSH 自带的
  Python 3.12.14（pytest 9.2.0 / openpyxl 3.1.5）：
  `C:\Users\15455\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe -m pytest tests -q`
  整套约 3 分钟。`AGENT.md` 里写的 conda 环境在本机不存在。
  numpy 2.3.5 可用，但**不要**把 numpy 加进 `pyproject.toml`（用户明确要求）。

---

## 1. 现在有哪些东西

| 文件 | 职责 |
| --- | --- |
| `graph_features.py` | 静态干员图：阵营超节点 + 点名羁绊边 + 组合机制边；`graph-audit` CLI |
| `graph_bonds.py` | 显式羁绊基（阵营人数 / 点名羁绊指示函数），按「房间 × 产物」门控 |
| `graph_conv.py` | 固定超参图卷积（确定性稀疏投影）+ 闭式岭回归读出层；numpy 可选加速 |
| `graph_cache.py` | 读出层 / 机制超边的磁盘缓存（`outputs/graph_model_cache/`） |
| `graph_optimizer.py` | `GraphScheduleOptimizer(ScheduleOptimizer)`，同一契约 |
| `graph_benchmark.py` | `benchmark` CLI：新旧对照，异常容错 |
| `morale.py` | **上游**的心情周期审计（本次未改，见 §3） |
| `docs/graph_optimizer.md` | 设计 + 实测（含 §3 打分模型、§5b 排班语义现状） |

关键命令：

```powershell
python -m arknights_schedule_generator graph-audit --roster examples/fixtures/yituliu_full_roster_maxed.xlsx --data-dir data/cache
python -m arknights_schedule_generator benchmark --roster examples/fixtures/yituliu_full_roster_maxed.xlsx --data-dir data/cache --output-dir outputs/benchmark
python -m arknights_schedule_generator generate --roster ... --optimizer graph --layout 243 --mode balanced-orundum --output outputs/x.json
```

---

## 2. 心情/宿舍：正确模型（本轮核心结论）

### 2.1 wiki 公式与数据核对（已全部验证）

宿舍每小时恢复 = **白字 + 绿字 + 干员技能**

| 项 | 公式 | 数据来源 | 验证结果 |
| --- | --- | --- | --- |
| 白字 | `1.5 + 0.1 × 宿舍等级` | `dormData.phases[].manpowerRecover` 160~200（÷100） | 与 `morale.dormitory_base_recovery_rate` 一致 ✅ |
| 绿字 | `0.0004 × 实际氛围 + 技能加成` | `comfortLimit = 5000`；`decorationLimit` 按等级 1000/2000/3000/4000/5000 | **代码完全缺失**，全仓无「氛围」引用 ❌ |
| 技能 | 自身 `+0.7~+2.0`、所有干员 `+0.05` | `dorm_rec_oneself[*]` 等 | 已有解析 ✅（但见 §2.3 的死代码） |

L5 + 氛围 5000：白字 2.0 + 绿字 **2.0** = **4.0/h** ⇒ 0→24 恰好 **6 小时**。
所以「六小时回满」不是经验值，是 wiki 公式的直接推论。加上自身技能后常用值 4.7~6.0/h。

**氛围取值约定**：玩家通常维持满氛围，所以默认 `comfort = min(comfortLimit, decorationLimit(level))`。
L5→5000、L1→1000。这个假设要写进报告的 `assumptions`。

### 2.2 上游门槛的 4 处偏差（已实测，全部偏「更严」）

1. **减耗技能被丢弃**：47 条 buff 是「心情每小时消耗**−N**」（`热情澎湃 −0.52`、`拾荒者/囤积者 −0.25`），
   `_simulate_cycle`（旧名 `_advance_morale`）的 `1.0 + max(0.0, fatigue_delta_from_text(...))` 把负号抹平 → 高估消耗。
   正确写法：`max(0.0, 1.0 + delta)`（下限 0，但允许低于 1）。
2. **中枢全局心情效果完全没建模**：79 条相关文本，含 `control_clue_cost[000] 神经质`（中枢内所有干员 +1.5/h）、
   `至察`（+0.5）、`反抗者`（除自身外 +0.25）、7 条 `control_mp_cost[*]`（中枢内所有干员恢复 +0.05/h）。
   双向遗漏，需要按「中枢在岗干员集合」做一次聚合，类似 `ControlGlobalEffect` 的做法。
3. **宿舍只按基率**：见 §2.1，缺绿字 → 0→24 算成 12h。
4. **单体恢复分支是死代码**：`某一名/前一位…心情每小时恢复+` 在真实数据 **0 命中**，`single_bonus` 恒 0。

### 2.3 为什么我上一轮加了 4.0/h 下限却全线失败（重要，别重复踩）

- 加「6h 预算 = 4.0/h 下限」后，**原本可行的 243/333 全部变成 `cycle_does_not_restore_initial_morale`**。
- 只保留「减耗修正」（方向相反、更省心情）**同样全线失败**。
- 两个相反方向的改动都失败 ⇒ 不是恢复率不足，而是**门槛的数值判定不健壮**：

> （旧实现）`audit_morale_cycle` 最多迭代 16 次 `_advance_morale`，然后用「再走一步」的结果
> 与存下的 `cycle_start_morale` 比较。一旦干员停在 `MAX_MORALE = 24` 封顶处，
> 映射不是压缩映射，迭代不收敛，`end_deficit` 就会把可行排班判成不达标。
> 它只报「周期回不到初值」，不告诉你自己没收敛。

**结论：必须先修判定，再改恢复模型。** 这与用户提出的顺序一致。（第 1 步已按此完成，见 §3。）

### 2.4 252/342 不可行的真正原因（结构性）

`_ensure_sustainable_morale_cycle` 的错峰展开：

```python
source_index = ((plan_index + group_count - 1 - group_index) // group_count) % len(shifts)
```

以 342 / 3×8 / `group_count=2` 为例，某房间组取到的源班次序列是 `[0, 1, 1, 2, 2, 0]`——
**连续工作 6 个班次（48h）、零休息计划**。任何恢复率都救不了，必须保证每组每周期至少一个休息计划。

---

## 3. 建议的修复顺序（下个窗口照此执行）

### 第 1 步：修 `morale.audit_morale_cycle` 的判定（先加测试）——**已完成**

原计划（保留作背景）：

1. 迭代改为「跑到固定点或 `MAX_ITER`」，并把是否收敛写进返回值
   （`converged: bool`、`iterations: int`、`residual: float`）。
2. `end_deficit` 只在**收敛后**才作为失败依据；未收敛时报
   `iteration_did_not_converge`（新的失败原因），不要伪装成「回不到初值」。
3. 判据从「末值 < 初值」改为**收支记账**更稳：对每个干员累计
   `total_recovery - total_consumption`，周期可持续 ⟺ 每项 ≥ 0（且最低心情 ≥ 0）。
   这天然免疫封顶带来的不收敛问题。**这条是首选方案。**
4. 保留 `required_rest` / `active_dorm_conflict` 两项结构检查不变。

实际做法（测试先写，`tests/test_morale.py`，11 个用例）：

- **判定 = 闭式收支记账**（第 3 条）：一个周期内每个干员
  `balance = potential_recovery - consumption`，`balance >= -1e-4` 才可持续。
  收支里记的是**潜在恢复**（rate × hours，先不管 24 封顶），所以「满心情时休息」不被惩罚；
  落地的恢复另记 `appliedRecovery`，两者之差就是封顶浪费。
- **不引入以收敛为前提的判定**（第 1、2 条没照做）：既然收支是闭式的，把「是否收敛」写进
  判据反而会重新引入误判（耦合技能可以让重放永远绕圈，但排班是可持续的）。
  `converged` / `iterations` / `residual` 放在 `steadyState` 里**只作诊断、不参与判定**，
  也没有 `iteration_did_not_converge` 这个失败原因。
- `cycle_does_not_restore_initial_morale` 删除，换成 `morale_balance_deficit`；
  `unrestedOperators` 现在是「缺休息 ∪ 周期内心情 <0 ∪ 收支为负」的并集。
- **报告数字有界**：`minimumMorale` / `endingMoraleByOperator` 来自「全员满心情起步走一个周期」
  的一次确定性重放。旧实现的 16 次重放会把 −6 的缺口放大成 `minimumMorale = -102`；
  现在上限就是一个周期的消耗量。HTML / exporter 读的字段全部保留，只做加法。
- 审计 schema `version: 1 → 2`；`_ensure_sustainable_morale_cycle` 的 `ValueError` 文案额外打印
  每个问题干员的 `balance`。

回归测试（对应原计划的三条，都在 `tests/test_morale.py`）：
- 收支平衡的最小排班必须 `hardGatePassed=True`（含精确平衡 0 的边界）；
- 「某人零休息」必须失败，且原因明确（`required_rest_exceeds_dormitory_assignment`）；
- **封顶场景**（休息落在周期开头、恢复被 24 封顶丢弃、末值低于初值）必须通过；
- 收不抵支必须报 `morale_balance_deficit`，且 `minimumMorale` 不得跑到一周期之外；
- 重放不收敛（`steadyState.converged=False`）不得影响判定。

### 第 2 步：补绿字 + 减耗（数值修正）

```python
# morale.py
DORMITORY_COMFORT_LIMIT = 5000            # building_data.comfortLimit
COMFORT_RECOVERY_PER_POINT = 0.0004       # wiki: 0.0004 × 氛围
def dormitory_comfort(level):             # 默认玩家维持满氛围
    return min(DORMITORY_COMFORT_LIMIT, 1000 * clamp(level, 1, 5))
```
恢复速率 = 白字 + `0.0004 × comfort` + 技能加成；消耗 = `max(0.0, 1.0 + delta)`。
菲亚梅塔**按用户要求不管**（`自律` 的「无法获得其他来源」标记可以留一个 TODO 注释）。

预期：243/333 仍然通过且更宽松；252/342 在修完第 3 步后才可能可行。

### 第 3 步：修错峰展开（结构）

保证每个房间组在每个周期内**至少有一个休息计划**。可以直接改成「轮转表」：
按组数 `g` 生成 `g` 个相位，相位 `k` 的组 `j` 在 `(plan_index + j) % g == k` 时休息。
改完再跑 §4 的可行性扫描确认 252/342。

### 第 4 步：把宿舍做成图节点（用户提议，赞同）

**为什么对**：当前架构里宿舍是**事后贪心**（`_assign_dormitories`），所以门槛只能靠「事后否决 +
错峰发明」来兜；一旦把宿舍变成一等分配目标，心情就变成**可定价的软账**，不需要那条
「离岗必须立刻休息」的硬规则。

具体做法：

1. **节点**：为每个宿舍床位建节点（`dorm:1..N`），与干员节点连边。
   特征：宿舍等级、氛围、当前恢复率（白字+绿字）、是否已被占用。
2. **边**：`dorm_rec` 边（干员→宿舍），权重由该干员的宿舍恢复技能解析得到
   （`dorm_rec_oneself[*]` / 所有干员 / 单体）。这样 `graph_bonds` 的显式基可以直接复用。
3. **读出层**：一元项加「该干员在宿舍的恢复贡献」，二元项加「同宿舍恢复叠加」。
   现有 `RoomValueModel` 的 `unary + pair` 结构可以原样吃下，只需给 `RoomSpec` 增加
   `room_type="DORMITORY"` 的规格，并把 `_room_specs_for_targets` 里补上宿舍。
4. **解码**：每班次每个干员有且只有一个去向（工作房 / 宿舍 / 闲置）。这就是一个
   带容量的二分匹配，`_assign_work_rooms_jointly` 的 beam 结构可以直接扩展——
   把宿舍作为「容量 = 床位、价值 = 心情恢复」的房间加入同一个 beam。
5. **心情账本**：解码时按 §2.1 的 4.0/h 记账（6h 回满），把它作为
   **软目标**（进入 `room_output_score` / `score_breakdown` 的一个小权重项）
   加**硬约束**（每周期收支 ≥ 0）。这样门槛不再是事后否决，而是搜索时就知道。

风险提示：
- 不要让宿舍节点进入产能分项，只进「心情可持续性」分项，否则会把分数算错。
- 现有 `moraleCycleAudit` 的字段被 exporter / HTML 读，新增字段可以，删改要谨慎。
- 宿舍容量 = `layout.dormitory × 5`，而 243 在岗 29 人 > 20 床位 —— 这正是「每人每天只上 1 个班」
  与「离岗必须立刻休息」冲突的根源。做成图节点后，可以允许**部分人连班、部分人轮休**，
  这个冲突自然消失。

### 第 5 步（之后）：办公室选人

现状：`roomEfficiency` 已按「1% 办公室 ≡ 2% 贸易站」计入总分，但
`_assign_room` 对 HIRE 仍按 `score_skill` 的解析百分比排序，不走加权房间分、
也不吃中枢 `control_hire_spd`。改法：让 HIRE/MEETING 也走
`_score_room_combo` → `room_output_score`（`ROOM_EFFICIENCY_SCORED_ROOM_TYPES` 已包含它们），
即把它们加入 `_assign_room` 的 `_best_room_combo` 分支。

---

## 4. 复现用的诊断脚本（本轮用过，已删，可按需重建）

```python
# 可行性扫描：哪些布局/模式/班次能过心情门槛
for raw in ("243", "252", "342", "333"):
    for mode in ("normal", "balanced-orundum", "max-orundum"):
        for cnt, hrs in ((2, 12), (3, 8)):
            for drone in ("none", "auto"):
                try: ScheduleOptimizer(...).optimize(...); print("OK")
                except ValueError as exc: print("FAIL", str(exc)[:70])
```

本轮结果（rebase 后、修判定前）：**只有 243 与 333 通过**；252/342 全部失败。

修判定后（审计 `version: 2`）用同样口径复测 243 / 252 / 342（平衡搓玉、2×12、满练表）：

| 布局 | 结果 | 说明 |
| --- | --- | --- |
| 243 | 通过（混合 / 图卷积都通过） | `cycleHours` 48（混合）/ 144（图卷积），最差收支 +1.2 / +3.6 |
| 252 | 失败 | 只有 `morale_balance_deficit`，缺口每人每周期 −0.6 ~ −3.6 心情 |
| 342 | 失败 | 同上，缺口 −2.4 ~ −3.6 |

关键：252 / 342 现在是**数值差一点**，不是旧实现报的「回不到初值」或 −180 那种产物；
缺口量级（≤3.6/周期）远小于第 2 步「补宿舍绿字」（L5 满氛围 +2.0/h，一个 12h 休息计划多出 24）
能补回来的量。所以第 2 步很可能直接把 252/342 判成可行 —— 到时要重新跑基准，别再假设
「5 制造站结构性不可行」。`_ensure_sustainable_morale_cycle` 的错峰展开（§2.4）在这两条路径上
都给出了休息计划，没有触发 `required_rest_exceeds_dormitory_assignment`。

---

## 5. 基准现状（rebase 后，1 胜 3 负，均比 0.981）

| 场景 | 混合 | 图卷积 | 变化 | 模拟器调用比 |
| --- | --- | --- | --- | --- |
| 243 平衡搓玉 2×12 | 380.290 | 381.201 | +0.24% | 0.624× |
| 243 不搓玉 2×12 | 605.566 | 603.177 | −0.39% | 0.788× |
| 333 平衡搓玉 3×8 auto | 498.041 | 483.634 | −2.89% | 0.700× |
| 333 极限搓玉 2×12 auto | 504.514 | 482.235 | −4.42% | 0.760× |

333 落后的诊断（已排除）：加宽短名单更差（128 → −30.8，192/beam22 → −35.8）、
加第三个起手无变化。**推断**：短名单只优化房间产能分，而心情门槛给「宿舍恢复能力」定价，
混合算法穷举会碰巧选中休息友好的人。**这正好由 §3 第 4 步（宿舍图节点）解决。**

---

## 6. 已做过、别重复做的实验

- 拟合样本 288 → 1536：holdout R² 0.82→0.89、MAE 2.4→1.5，但**最终分数掉约 4%**
  （短名单变窄、下游 beam 失去多样性）⇒ 默认保持 288。
- 覆盖保障 `coverage_extra=24`：在 243 不搓玉上净损失 9 分 ⇒ 默认 0，羁绊闭包单独用
  `bond_closure_budget=12`。
- 羁绊闭包在 243 平衡搓玉赚 14 分、在 243 不搓玉亏 9 分（符号随场景变）⇒
  用 `initial_starts=(12, 0)` 多起手 + `target_selection_score` 择优。
- 插入搜索预算更快但对顺序高度敏感（±3%）⇒ 默认关闭。
- 显式羁绊特征必须按 (room_type, target) 门控：黑钢/莱茵是全基建计数、op 是发电站计数，
  强行做房间级一元项会让制造站 R² 从 0.75 掉到 0.70。
- 别名阵营标签（rhine/rh、blacksteel/bs、knight/kazimierz）成员集合相同，必须去重，
  否则完全共线。

---

## 7. 工作区状态（未提交）

| 文件 | 改动 |
| --- | --- |
| `morale.py` | **第 1 步**：判定改成「每人一周期收支记账」，schema `version: 2`，新增 `moraleLedgerByOperator` / `moraleBalanceDeficitOperators` / `steadyState` |
| `tests/test_morale.py`（新增） | 心情审计契约测试 11 个（`tests/test_morale.py`） |
| `optimizer.py` | `_ensure_sustainable_morale_cycle` 的 `ValueError` 文案加了每个问题干员的 `balance=` |
| `graph_benchmark.py` | 失败场景容错（`failed`/`error` 字段）、默认场景改为 243/333、羁绊系数表 |
| `graph_optimizer.py` | 纯赤金重复次数显式跳过 `base_hours × repeat > 168h`（否则触发心情门槛崩溃）；`initial_starts` 多起手；覆盖/闭包拆分 |
| `docs/graph_optimizer.md` | §3 打分模型、§4 numpy/缓存、§5b 排班语义（已按收支记账口径重写） |
| `docs/memory/next_steps.md` | P6/P7/P8/P9 四段 + P10（第 1 步完成记录） |
| `AGENT.md` | 「接手入口（图卷积 / 心情模型）」——注意这段**重复了两遍**，下次顺手合并 |
| `docs/handoff_graph_optimizer.md`（新增） | 本文档 |

---

## 8. 陷阱清单

- `optimize()` 现在会因心情门槛抛 `ValueError`；`recommend` 会把它记为 skipped candidate，
  CLI `generate` 会直接崩 —— 写新代码时记得 try/except。
- `_apply_pure_gold_drone_cycle_repeats` 里必须同时尊重 `MAX_CYCLE_HOURS`（168h），
  否则 `cycle_exceeds_seven_days`。
- 图优化器是 `ScheduleOptimizer` 子类，**新增的父类逻辑会自动继承**（心情门槛就是这样进来的）。
  改父类方法时要考虑子类的 `_closure_override` / 缓存失效。
- `docs/graph_optimizer.md` §5b 与 `AGENT.md` 的「关键口径」曾经写成「宿舍不进模型」，
  那是 rebase 前的旧基线；已更正为「宿舍参与心情硬门槛」。不要再照旧说法写文档。
- **不要再把「周期末心情 vs 周期初心情」写成判据**（`cycle_does_not_restore_initial_morale`
  已删除）。24 封顶会丢弃恢复，这个比较会把可行排班判成不可行。判定只认
  `moraleBalanceDeficitOperators`（每人一周期收支）；`steadyState.converged` 是诊断，不是门槛。
- 心情审计的 `minimumMorale*` / `endingMoraleByOperator` 来自「全员满心情起步走一个周期」的
  一次重放，所以不可行排班的数字也被一个周期的消耗量界定。别再拿封顶后的状态做报告。
