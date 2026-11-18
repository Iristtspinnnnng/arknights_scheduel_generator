# 接手文档：图卷积优化器 + 心情/宿舍模型

> 写于上下文耗尽前。下一个窗口从这里开始，不必回读聊天记录。
> 最后更新：第 1 步（修 `morale.audit_morale_cycle` 判定）+ 第 2 步（宿舍绿字、减耗、控制中枢）
> 都已完成；252/342 已由「全部失败」变成全部可行。回归测试在 `tests/test_morale.py`。

---

## 0. 一分钟速览

- 分支 `main`，**领先 `origin/main` 7 个提交，未 push**：`8213fb3`（图卷积优化器首版）、
  `c2d39ba`（社区打分权重 + 显式羁绊 + numpy + 缓存 + 多起手）、
  `d78daad`（心情审计改成收支记账 + 老的基准容错/循环上限）、
  `9e1b605`（宿舍绿字 + 作用域减耗 + 控制中枢 + 父类纯赤金 7 天上限），
  以及最后三个「单体宿舍恢复的锁定语义」「菲亚梅塔＝自给自足特殊节点」「252/342 基准记录」
  （`git log` 可见，内容见 §2.2 第 4、5 条与 §5）。
- **§3 第 1、2 步已完成**：
  - 判定改成「每个干员一周期收支记账」（审计 `version: 2` → 现在是 `version: 3`）；
  - 宿舍补上绿字（`0.0004 × 氛围`）、在岗减耗按作用域解析、控制中枢 −0.05/h × 在岗人数；
  - 单体宿舍恢复（`dorm_rec_single`，26 条）按「锁定最低心情舍友直到回满」的语义启用；
  - 菲亚梅塔「自律」按「恢复自给自足的特殊节点」建模（图侧只标注、不加特征）；
  - **252/342 现在全部可行**（16/16 组合通过，最差收支 +4.0）。
  细节见 §2、§3、§4。
- 安全备份分支：`backup/pre-rebase-graph`（rebase 前的原始状态）。
- 测试：**254 passed / 1 skipped / 1 failed**（含 `tests/test_morale.py` 24 个、纯赤金 7 天上限 1 个、图标注 1 个）。唯一失败是既有环境问题
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

### 2.1 公式与数据核对（已全部验证，且已实现）

宿舍每小时恢复 = **白字 + 绿字 + 干员技能**

| 项 | 公式 | 数据来源 | 状态 |
| --- | --- | --- | --- |
| 白字 | `1.5 + 0.1 × 宿舍等级` | `dormData.phases[].manpowerRecover` 160~200（÷100） | `dormitory_base_recovery_rate` ✅ |
| 绿字 | `0.0004 × 实际氛围` | `comfortLimit = 5000`、`comfortManpowerRecoverFactor = 25`（⇒ 1/(25×100) = 0.0004，满氛围 L5 = +2.0/h）；`decorationLimit` 按等级 1000~5000 | 已实现：`dormitory_comfort` / `dormitory_comfort_recovery_rate` ✅ |
| 技能 | 自身 `+0.7~+2.0`、所有干员 `+0.05`、单体 `+0.2~+0.75` | `dorm_rec_oneself[*]` / `dorm_rec_all[*]` / `dorm_rec_single[*]` 等 | `dormitory_recovery_bonuses` ✅（单体 26 条已启用，见 §2.2 第 4 条） |
| 中枢 | 每名在岗中枢干员让**所有在岗**干员消耗 −0.05/h；中枢技能再叠加 | `controlData.basicCostBuff = -5`（1/100 单位）；`control_mp_cost[*]` / `control_clue_cost[*]` | 已实现：`CONTROL_MORALE_RELIEF_PER_OPERATOR` + `parse_morale_effects` ✅ |
| 减耗 | `心情每小时消耗−N` | 47 条 buff（`热情澎湃 −0.52`、`拾荒者/囤积者 −0.25`…） | 已实现：按作用域解析后 `max(0, ·)` ✅ |

档位（可用 `python -c "from arknights_schedule_generator.morale import dormitory_passive_recovery_rate as r; print([r(i) for i in range(1, 6)])"` 复算，
应当是 `[2.0, 2.5, 3.0, 3.5, 4.0]`）：

| 宿舍等级 | 白字 | 绿字 | 合计 |
| --- | --- | --- | --- |
| L1 | 1.6 | 0.4 | **2.0/h** |
| L3 | 1.8 | 1.2 | **3.0/h** |
| L5 | 2.0 | 2.0 | **4.0/h** |

L5 + 氛围 5000 ⇒ 0→24 恰好 **6 小时**，这就是「六小时回满」的来源；不靠经验值。

**氛围取值约定**：玩家通常维持满氛围，默认 `comfort = min(comfortLimit, 1000 × level)`。
这条已经写进审计的 `assumptions`（"Full dormitory comfort is assumed..."）。

### 2.2 上游门槛的 4 处偏差（已全部处理）

1. **减耗技能被丢弃** —— 已修。现在按作用域解析（自身 / 本站全体 / 中枢内 / 其他设施 / 全基建），
   最后 `max(0, 1 + delta)`；并且会分给同房间的其他人（`manu_cost`、`trade_cost`）。
2. **中枢全局心情效果没建模** —— 已修。基准规则（−0.05/h × 在岗人数，含中枢自己）+ 中枢技能：
   `control_mp_cost[*]`（中枢内所有干员恢复 +0.05/h，按人数累加）、
   `control_clue_cost[000/010]`（中枢内所有干员 +1.5 / +0.5 消耗）、
   `control_clue_cost&faction[990]`（除自身外的中枢干员 +0.25）、
   `control_dorm_rec[*]`（所有宿舍内恢复 +0.05）、`control_mp_bd_cost_expand` /
   `control_mp_expand_double`（其他设施内在岗干员恢复 +0.05/+0.1）。**未建模**：按阵营人数（`control_mp_cost&faction[*]`）、
   指名干员、`部分设施`、`当与 X` 且 X 不在岗 —— 这些进 `unmodelledMoraleEffects`（真实数据 21 条）。
3. **宿舍只按基率** —— 已修，见 §2.1 绿字。
4. **单体恢复分支是死代码** —— 已修（第 2 步之后补的）。真实文本是「除自身以外心情未满的
   **某个**干员每小时恢复 +0.55」，旧正则只认 `某一名|前一位`；现在 `某个` 也认，
   真实数据 **26 条**命中（`dorm_rec_single*`、`dorm_rec_bd_n1_n3`、`dorm_rec_single&oneself*`）。

   语义按用户观察到的实际行为实现（这一条不是每帧取最低，而是**锁定**）：
   - 进入宿舍时，buff 落在**当前心情最低**的舍友身上（`除自身以外`，所以技能持有者永远不是目标）；
   - 一直锁定在该干员身上，直到他**回满心情**或**离开宿舍**；回满后按同样规则移交给下一个
     心情最低且未满的舍友；
   - 宿舍里没有正在恢复的人（全都满）时，等第一个心情未满的干员出现再给。
   实现方式是把一次宿舍停留拆成若干「恒定速率段」：每段解出当前目标的 `(24 - 心情) / 速率`
   饱和时间，取 `min(剩余时间, 饱和时间)` 推进，目标是精确的，不是近似迭代。
   见 `_recover_dormitory_room` / `_lowest_unfilled` / `_advance_dormitory_occupants`，
   回归测试 `test_single_target_recovery_*`（3 个）。
5. **菲亚梅塔「自律」** —— 已建模为**恢复自给自足的特殊节点**（用户提议）。
   `skill_rules.self_contained_recovery_rate()` 识别「…无法获得其他来源提供的心情恢复效果」，
   返回她自己的 `+2/h`；`morale._recover_dormitory_room` 把她当独立节点：只按 2.0/h 恢复，
   **不接**白字/绿字/舍友加成/中枢宿舍加成，也**不会**成为单体技能的目标。
   图侧在 `graph_features.meta["specialMoraleNodes"]` 里标注（`graph-audit` 的
   `specialMoraleNodes` 可见），但**不加 GCN 特征**——它只影响心情可行性、对产能读出层没有梯度，
   加了反而是噪声，还会白白让读出层缓存失效。
   「患难之交」（满心情换心情）按用户要求**先不管**，仍是未建模项。

### 2.3 为什么我上一轮加了 4.0/h 下限却全线失败（历史记录，别重复踩）

- 加「6h 预算 = 4.0/h 下限」后，**原本可行的 243/333 全部变成 `cycle_does_not_restore_initial_morale`**。
- 只保留「减耗修正」（方向相反、更省心情）**同样全线失败**。
- 两个相反方向的改动都失败 ⇒ 不是恢复率不足，而是**门槛的数值判定不健壮**：

> （旧实现）`audit_morale_cycle` 最多迭代 16 次 `_advance_morale`，然后用「再走一步」的结果
> 与存下的 `cycle_start_morale` 比较。一旦干员停在 `MAX_MORALE = 24` 封顶处，
> 映射不是压缩映射，迭代不收敛，`end_deficit` 就会把可行排班判成不达标。
> 它只报「周期回不到初值」，不告诉你自己没收敛。

**结论：必须先修判定，再改恢复模型。** 这与用户提出的顺序一致。
**两步都已按此完成**：第 1 步修判定（§3），第 2 步补绿字/减耗/中枢（§3 第 2 步）。
用新判定重跑同一个 4.0/h 实验会是 *放宽* 而不是失败 —— 这正是当初的坑在判定而不是数值上。

### 2.4 252/342 曾经不可行：错峰展开 + 数值缺口

`_ensure_sustainable_morale_cycle` 的错峰展开：

```python
source_index = ((plan_index + group_count - 1 - group_index) // group_count) % len(shifts)
```

在 342 / 3×8 / `group_count=2` 这种极端分组下，某房间组取到的源班次序列是 `[0, 1, 1, 2, 2, 0]`——
**连续工作 6 个班次（48h）、零休息计划**，任何恢复率都救不了。不过实测 2×12 与 3×8 的默认位置
并没有走到这个极端分组：修完判定后 252/342 只是**数值差一点**（缺口 −0.6 ~ −3.6 心情/周期），
补上绿字和中枢减耗之后 16 种组合全部可行（§4）。所以「必须保证每组每周期至少一个休息计划」
仍然值得做（是个潜在崩溃点），但已经不是 5 制造站不可行的原因。

### 2.5 电力 → 宿舍等级 → 恢复率：**已建模**（用户问过，结论是有）

链路是完整接通的：
`power.power_status()` 用 `building_data.rooms[*].phases[*].electricity` 算「供电 − 耗电」；
`resolve_power_layout()` 在 `--right-side full` 且余量为负时调 `auto_adjust_full_power_layout()`，
按 `(制造站总工位, 制造站等级和, 宿舍等级和, 余量)` 字典序**先砍制造站、再砍宿舍等级**；
调整结果写进 `layout.dormitory_levels` 与 `layout.power_adjustment`，而 `morale` 用的正是这个
`layout.dormitory_levels`。实测（满练表、`full` 预设）：

| 布局 | 初始余量 | 调整后余量 | 宿舍等级 | 恢复率 |
| --- | --- | --- | --- | --- |
| 243 / 333（3 发电站） | 0 | 0 | `(5,5,5,5)` | 4.0/h |
| 252 / 342（2 发电站） | **−330** | 0 | **`(1,1,1,2)`** | **2.0 / 2.5/h** |

也就是说「发电站 3→2 限制宿舍等级」这件事，模型里**真的会扣恢复率**：252/342 的心情账一直是按
2.0~2.5/h 算的（不是 4.0/h）。两点已知局限（用户决定先不管）：
1. 砍电力的**优先级**只看「保住制造站工位」，没有把「宿舍降级 → 恢复率下降 → 排班更难」计价；
2. `guide` / `basic` 预设直接把宿舍钉成 1 级，且 `resolve_power_layout` 只对 `full` 做自动调整，
   所以 252/342 + `guide` 会留下 `feasible=False` 的电力余量（导出里 `powerStatus.feasible` 可见）。

### 2.6 换班仍然是「批量固定时刻」

`_build_shifts` 给每个 `ShiftPlan` 一个固定 `start`（`shift_times[shift_index % len(shift_times)]`），
同一班次里所有干员**同一刻上、同一刻下**；`_ensure_sustainable_morale_cycle` 的错峰展开也只是
把计划数乘成 `base × group_count`，时刻仍沿用同一套网格。没有「某个干员提前 2 小时下班」这种
个体化换班。若以后要做（Mower 那种逐人换班），需要给 `ShiftPlan` 增加「每人区间」而不是一个 clock。

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

### 第 2 步：补绿字 + 减耗（数值修正）——**已完成**

```python
# morale.py
DORMITORY_COMFORT_LIMIT = 5000            # building_data.comfortLimit
DORMITORY_COMFORT_PER_LEVEL = 1000        # dormData.phases[].decorationLimit
COMFORT_RECOVERY_PER_POINT = 0.0004       # comfortManpowerRecoverFactor=25 ⇒ 1/(25*100)
CONTROL_MORALE_RELIEF_PER_OPERATOR = 0.05  # controlData.basicCostBuff = -5（1/100 单位）
```

已实现的模型（比原计划多了一层「作用域」）：

- 宿舍恢复 = 白字 + `0.0004 × 氛围` + 技能（+ 中枢给的「所有宿舍内」加成）；
- 在岗消耗 = `1 + 修正`，修正按作用域解析：**自身**（`心情每小时消耗−0.25`）、
  **本站全体**（`当前制造站内所有干员−0.1`、`贸易站内全体干员−0.1`）、
  **控制中枢内**（+1.5 线索代价 / −0.05 恢复，按人数累加）、
  **其他设施内**（`control_mp_bd_cost_expand`）、**全基建**；
- **控制中枢基准规则**：每名在岗中枢干员让所有在岗干员（含中枢自己）消耗 −0.05/h；
- 最后 `max(0, ·)`，并把它记进 `moraleLedgerByOperator`，所以可以逐人核对；
- 归属不明的条款（按阵营人数、指名干员、`部分设施`、`当与 X` 且 X 不在岗、精英限定）进
  `unmodelledMoraleEffects`，真实数据 21 条。

为什么必须做作用域：`fatigue_delta_from_text` 是把整段文本里的 `心情每小时消耗/恢复` 全部加起来的，
一旦开始计入「控制中枢内所有干员恢复 +0.05」这类条款，再按整段 delta 计自身就会**重复计算**。
所以 `parse_morale_effects` 先按分句判作用域，再分派到各通道。

菲亚梅塔仍然**按用户要求不管**（`自律` 的「无法获得其他来源」标记可以留一个 TODO 注释）。
`dorm_rec_single`（`某个干员`）仍未启用，见 §2.2 第 4 条。

实测（满练表，`ScheduleOptimizer`）：243/252/342/333 × 普通/平衡搓玉 × 2×12/3×8 **16 种组合全部可行**，
252/342 由「全部失败」变成最差收支 +4.0 ~ +7.2。详见 §4。

另外补上了 `dorm_rec_single`（26 条）：目标锁定在「除自身以外心情最低且未满」的舍友，
回满后移交给下一个，见 §2.2 第 4 条。补完之后 16 种组合的分数与最差收支都没有变化
（宿舍干员的选择本来就不进产能分，这项只是让心情账更准）。

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

修判定后（审计 `version: 2`，补绿字/中枢之前）：252/342 变成**数值差一点** ——
只有 `morale_balance_deficit`，缺口每人每周期 −0.6 ~ −3.6 心情，没有触发
`required_rest_exceeds_dormitory_assignment`。

补完绿字 + 减耗 + 中枢（审计 `version: 3`）后，同一脚本（满练表、普通/平衡搓玉、2×12/3×8）实测：

| 布局 | 2×12 | 3×8 | 最差收支 | `cycleHours` |
| --- | --- | --- | --- | --- |
| 243 | 通过 605.6 / 380.3 | 通过 563.5 / 354.9 | +14.0 | 48 / 72 |
| 252 | 通过 646.6 / 359.9 | 通过 599.7 / 333.1 | +4.0 | 48 |
| 342 | 通过 680.8 / 546.6 | 通过 632.4 / 499.0 | +4.0 | 48 |
| 333 | 通过 696.3 / 544.3 | 通过 641.5 / 498.0 | +20.0 | 48 |

（分数为「普通 / 平衡搓玉」，无人机 none。）243 与 333 的分数与 rebase 后基准一致
（605.566 / 380.290 / 498.041），说明这次修的只是心情判定，没有动收益口径。

结论：**「5 制造站结构性不可行」作废**。默认基准仍保留 243/333 只为让历史数字可比；
`graph_benchmark.py` 的注释已同步。

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

修完第 1、2 步之后 243/333 的分数没有变（605.566 / 380.290 / 498.041 全部复现），
因为这两步只改心情判定与心情模型。

**252/342 的新对照已经跑完**（心情模型放宽后第一次，2×12、无人机 none）：

```powershell
python -m arknights_schedule_generator benchmark --roster .\examples\fixtures\yituliu_full_roster_maxed.xlsx `
  --data-dir .\data\cache --layouts 252,342 --modes balanced-orundum,normal `
  --shift-patterns 2x12 --drone-policies none --output-dir .\outputs\benchmark_252_342
```

| 场景 | 混合 | 图卷积 | 变化 | 模拟器调用比 |
| --- | --- | --- | --- | --- |
| 252 平衡搓玉 | 359.870 | **361.821** | +0.54% | 0.625× |
| 252 不搓玉 | 646.577 | 644.897 | −0.26% | 0.583× |
| 342 平衡搓玉 | 546.560 | **563.977** | +3.19% | 0.846× |
| 342 不搓玉 | 680.768 | 659.705 | −3.09% | 0.865× |

2 胜 2 负，平均分数比 1.000938，平均墙钟比 0.772，平均模拟器调用比 0.730；短名单 5/5 命中、
top16 召回 1.0。**规律：搓玉目标稳赢、不搓玉目标稳输**（243/252/342 三组都是这个符号），
机制见 `docs/graph_optimizer.md` §5 与 §6 第 2 条（合成玉贸易站零梯度 → 短名单在该房间随机，
混合算法穷举碰巧更好）。下次要缩小不搓玉的差距，优先做 §3 第 4 步（宿舍/房间联合解码）。

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

## 7. 工作区状态（前三步都已提交）

第 1 步（`d78daad`）已提交：`morale.py` 收支记账判定 + `tests/test_morale.py` + `optimizer.py` 文案
+ `graph_benchmark.py` 容错 + `docs/*`。

第 2 步（`9e1b605`）已提交：绿字、作用域减耗、控制中枢、父类纯赤金 7 天上限。

第三步（单体宿舍恢复）也已提交，内容：

| 文件 | 改动 |
| --- | --- |
| `morale.py` | `dormitory_recovery_bonuses` 识别 `某个`（26 条真实命中）并返回 `(bonus, owner)`；新增 `_recover_dormitory_room` / `_lowest_unfilled` / `_advance_dormitory_occupants`，按「恒定速率段」精确模拟单体恢复的锁定语义；`assumptions` 增加一条 |
| `tests/test_morale.py` | 22 个（新增 3 个：最低心情舍友、不给自己、回满后移交） |
| `docs/*` | handoff §2.1/§2.2/§3/§4/§5、`graph_optimizer.md` §5b、`memory/next_steps.md` P12、`AGENT.md` 心情口径 |

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
- **`fatigue_delta_from_text` 不能直接当「自身消耗修正」用**：它把整段描述里的
  `心情每小时消耗/恢复` 全加在一起，会把「中枢内所有干员恢复+0.05」「该宿舍内所有干员+0.15」
  算到技能持有者头上（重复计算）。心情审计里必须走 `parse_morale_effects` 的作用域分派；
  生产侧的 `fatigueRisk` 仍在用旧的整段 delta，改那边要先有分数对照。
- 新增一类心情条款（新文本模式）时：先在 `data/cache` 里统计命中数和作用域，再改
  `_morale_clause_scope`，最后**加一条 `tests/test_morale.py` 用例**；归属不明的就留在
  `unmodelledMoraleEffects`，别瞎猜作用域。
- 单体宿舍恢复（`dorm_rec_single`）是**锁定**语义，不是「每帧重算最低心情」：目标进入宿舍后确定，
  直到回满或离开才移交。别把它改回「每帧取最低」，否则同一班次内会反复横跳。
  它还有一条隐含约束：技能持有者自己**永远不是**目标（「除自身以外」）。
- 控制中枢的 −0.05/h 是「每个在岗干员」的**基准规则**（`controlData.basicCostBuff`），
  不是技能；不要把它写成技能解析，否则换了中枢人员就丢。
