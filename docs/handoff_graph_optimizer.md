# 接手文档：图卷积优化器 + 心情/宿舍模型

> 写于上下文耗尽前。下一个窗口从这里开始，不必回读聊天记录。
> 最后更新：第 1、2 步 + 单体宿舍恢复 + 菲亚梅塔（自给自足节点）+ 252/342 基准 +
> **第 4 步 M1（宿舍做成一等分配目标）都已完成**。
> **下一个窗口从 §3 第 5 步（办公室选人）或第 3 步（错峰轮转表）开始** —— 两步的现状与落点
> 都写在 §3；M1 的实测结论见 §3 第 4 步开头与 §5。

---

## 0. 一分钟速览

- 分支 `main`，**领先 `origin/main` 8 个提交，未 push**：`8213fb3`（图卷积优化器首版）、
  `c2d39ba`（社区打分权重 + 显式羁绊 + numpy + 缓存 + 多起手）、
  `d78daad`（心情审计改成收支记账 + 老的基准容错/循环上限）、
  `9e1b605`（宿舍绿字 + 作用域减耗 + 控制中枢 + 父类纯赤金 7 天上限），
  三个「单体宿舍恢复的锁定语义」「菲亚梅塔＝自给自足特殊节点」「252/342 基准记录」，
  以及最新的「M1：宿舍一等分配」（`git log` 可见，内容见 §3 第 4 步与 §5）。
- **§3 第 1、2、4 步已完成**：
  - 判定改成「每个干员一周期收支记账」（审计 `version: 2` → 现在是 `version: 3`）；
  - 宿舍补上绿字（`0.0004 × 氛围`）、在岗减耗按作用域解析、控制中枢 −0.05/h × 在岗人数；
  - 单体宿舍恢复（`dorm_rec_single`，26 条）按「锁定最低心情舍友直到回满」的语义启用；
  - 菲亚梅塔「自律」按「恢复自给自足的特殊节点」建模（图侧只标注、不加特征）；
  - **252/342 现在全部可行**（16/16 组合通过，最差收支 +4.0）；
  - **宿舍是一等分配目标（M1）**：整条排班一次 beam（两套目标）+ 旧贪心三候选，
    用心情审计自己的账本 `(门槛, 最差收支, 总恢复)` 择优。产能分一格没动，
    总恢复在 7/16 组合上变好，最差收支一格没变差。
  细节见 §2、§3、§4。
- 安全备份分支：`backup/pre-rebase-graph`（rebase 前的原始状态）。
- 测试：**262 passed / 1 skipped / 1 failed**（新增 `DormitoryFirstClassAssignmentTest` 8 个）。
  唯一失败是既有环境问题
  （`test_calibration_report_examples_pass` 需要仓库外的 `C:\tmp\yituliu_schedule_images_current.js`，
  报 `missing_local_asset`），与本项目改动无关，不要去"修"它。
- 环境：系统 `python`（WindowsApps，3.13）**没有 pytest**，不能直接跑测试；本轮实际用的是 DSH 自带的
  Python 3.12.14（pytest 9.2.0 / openpyxl 3.1.5）：
  `C:\Users\15455\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe -m pytest tests -q`
  整套约 75 秒（并发跑基准时会到 95 秒）。`AGENT.md` 里写的 conda 环境在本机不存在。
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

### 第 4 步：把宿舍做成一等分配目标（M1）——**已完成**

**背景变了，先说清楚为什么还值得做。** 原计划的动机是「宿舍是事后贪心 → 只能靠事后否决兜底」。
第 1、2 步之后 16/16 组合都有 +4.0 以上的收支余量，**可行性已经不是瓶颈**，所以这一步的收益从
「让 5 制造站可行」变成下面三件事：
1. **质量**：现在 `_assign_dormitories` 只看「谁离岗了」+ 少量锚点偏好，不看「谁在宿舍里恢复得值」。
   同一批人换个宿舍、换个人轮休，产能相同但心情余量差很多 —— 余量是以后放松班次表的本钱。
2. **让部分人连班、部分人轮休**：宿舍容量 = `layout.dormitory × 5`（243 = 20 床），在岗 29 人，
   现在的错峰展开只能整组整组地轮休；一等分配后可以用「容量匹配」自然做到混合轮换。
3. **不搓玉场景的差距**（§5 实测：图卷积在不搓玉上稳定小负）需要更宽的搜索空间才能补，
   而心情目前是「事后否决」，把这种空间直接砍掉了。

#### 实测结论（先看这个再看实现）

- **产能分一格没动**：16 组扫描的 `score` 与 M1 前**逐位相同**（max |Δscore| = 0.0），
  4 场景基准的混合分/图卷积分也与 §5 表逐位相同。这是设计约束，不是巧合（宿舍不进产能分）。
- **心情账本不退化**：16 组的最差收支 delta 全部 = 0.0（最差仍是 252/342 3×8 的 +4.0），
  总恢复 delta 全部 ≥ 0，其中 7 组变好（+1.1 ~ +12.1 心情/周期）。
- **收益是「宿舍分配有据可依 + 不漏人」**：243 平衡搓玉 2×12 上有床位覆盖从 61 人升到 63 人
  （全部 63 名上场干员都能轮到床位），总恢复 +3.0。2×12 的紧凑组合（20 床刚好被必休者占满）
  本来就没什么自由度，账本择优会**保留旧贪心**——这正是"不消耗余量"的保证。
- **代价**：单场景墙钟 +0.4% ~ +1.5%（243 平衡搓玉 12.82s → 12.87s；333 平衡搓玉 3×8
  20.01s → 20.31s）。**关键是不能把 beam 放回解码热路径**——那样实测 17.7s → 107.6s（6×），
  而最终排班一模一样。

#### 原计划（保留作背景）

M1 的目标：宿舍成为解码的一等目标（不动读出层、不动产能分、不改 `score`）

1. 宿舍规格：`RoomSpec(f"dormitory_{i}", "DORMITORY", None, capacity=5, room_level=layout.dormitory_levels[i-1])`。
2. 宿舍价值函数（**手写，不学**，因为宿舍没有产能标签）。
3. 解码：在 `_assign_work_rooms_jointly` 里加第三段宿舍 beam。
4. 心情账本只用于宿舍选择，**不要**进 `room_output_score`。

M2（可选，先别做）：读出层加「宿舍恢复」一元项并重新标定 → 会改变短名单，需要重跑基准确认
`243 balanced` 不退步。**注意**：改 `FEATURE_NAMES` 会让 `graph_signature` 变化、读出层缓存失效、
下次跑基准要重新拟合（几分钟）。M1 不碰特征，所以缓存不受影响（实测基准直接复用缓存，没有重拟合）。

#### 实际做法（`optimizer.py`）

1. **宿舍规格**：`_dorm_specs()` 返回 `RoomSpec`（容量 5，`slots=5`），等级取
   `levels_or_default(layout.dormitory_levels, layout.dormitory, 5)` —— 必须和心情审计的
   `layout.dormitory_levels[i] if i < len(...) else 5` 同口径，否则 beam 按 4.0/h 定价、
   审计按 2.0/h 记账。
2. **共用速率函数（关键）**：`morale.py` 新增
   - `dormitory_operator_bonuses(skill)`（单个技能的 all/self/single 解析，`dormitory_recovery_bonuses` 改为基于它）；
   - `dormitory_recovery_rates(room, passive)`（**审计与 beam 共用的唯一速率入口**）；
   - `dormitory_single_target(room)`、`control_dormitory_boost(shift)`、`morale_consumption_by_operator(layout, shifts)`。
   `_recover_dormitory_room` 改成调用它们，所以不存在"两套 rate 函数漂移"的风险。
3. **价值函数**：`_dormitory_stay_value(operators, room_level, dwell_hours, dormitory_boost, need)`
   返回 `(accounted, deficit_credit)`：
   - `accounted` = 一次停留的账面恢复（Σ rate × hours，单体技能按"本宿舍最缺心情的舍友"近似、
     总量与实际一致）；
   - `deficit_credit` = `Σ min(rate × hours, need[X])`，**只把缺口补上就够**。用它当主目标可以
     避免"谁消耗高就给谁多塞床位"，让床位流向还没补上的人。
4. **beam**：`_beam_dormitory_rooms` 按房间顺序做 beam（宽度 8，候选池 40）：
   必休者先按边际价值落位（硬约束），空床位有两种填法（先补缺口 / 先最大化账面恢复），
   先处理等级高的宿舍（床位不够时先填满它），输出顺序仍按 `_dorm_specs` 下标。
5. **落点**：`_assign_dormitory_quarters(layout, shifts, required_rest=...)` 在
   **`_ensure_sustainable_morale_cycle`** 里对整条排班调用一次（不是在解码的每个班次里），
   见下面「为什么不在 `_assign_work_rooms_jointly`」。
6. **三候选 + 账本裁决**：一次调用产出 `deficit` beam、`recovery` beam、旧贪心三份宿舍分配，
   用 `_dormitory_plan_key` = 审计自己的 `(hardGatePassed, 最差收支, 总恢复)` 择优，
   并列时保留 beam。所以 **M1 的分配永远不会比旧贪心差**（旧贪心是候选之一），
   "提高余量但不消耗最差收支"不是靠手调权重，而是由门槛自己的账本裁决。
7. **锚点覆盖保留**：`_improve_dormitory_anchor_coverage` 的语义是"让用户指定的锚点出现在导出里"，
   与心情无关，仍在 `_build_shifts` / 门槛之后跑，且从不替换必休的人。
8. **锚点偏好也进了 beam 的并列裁决**（`_dormitory_anchor_key`），所以 beam 不会把锚点挤没。

#### 为什么不在 `_assign_work_rooms_jointly`（与计划的偏差，重要）

计划写在 `_assign_work_rooms_jointly` 的第三段，实际落在心情硬门槛里，原因两条：

1. **床位是跨班次共享的稀缺资源**：`_assign_work_rooms_jointly` 每个班次调一次，
   看不到"这个干员在整个周期里缺多少心情"，也就没法把 20 张床在 4 个班次之间分摊。
2. **性能**：`_rebuild_dormitories`（插入/替换搜索的热路径）一次 `optimize()` 会调几百次。
   在那儿做 beam 实测把 243 平衡搓玉从 17.7s 拖到 107.6s（6×），而**最终排班一模一样**
   —— 因为宿舍不进产能分，搜索层根本看不到它。所以：
   - `_build_shifts` / `_rebuild_dormitories` 保持**便宜的逐班次贪心**；
   - 真正的 beam 只跑在最终排班上（门槛里，一次）。

一个直接后果：**错峰展开仍然会触发**（29 个在岗 > 20 床是结构性的，与宿舍分配无关）。
原计划里"M1 之后门槛基本不触发"的预期不成立；那是第 3 步（轮转表）的事，不是宿舍分配的事。

#### 落点清单（实际改动）

| 位置 | 改动 |
| --- | --- |
| `_dorm_specs` | 返回 `RoomSpec`（含 level / slots），`levels_or_default(..., 5)` |
| `_assign_dormitories` / `_assign_required_rest_dormitories` | 保留为**贪心基线**（M1 的候选之一），签名改吃 `RoomSpec`，产出带 `room_level` / `slots` |
| `_assign_dormitory_quarters`（新增） | 整条排班一次分配：两套 beam + 贪心候选，审计账本择优 |
| `_beam_dormitory_rooms` / `_dormitory_*`（新增） | beam、价值函数、候选池、目标函数 |
| `_build_shifts` / `_rebuild_dormitories` | **保持贪心**（热路径），只补注释说明为什么 |
| `_ensure_sustainable_morale_cycle` | 门槛通过时用 beam 升级再复核；需要展开时对展开后的排班做 beam（`required_rest` = 离岗者） |
| `morale.py` | 新增共用速率/记账函数，`_recover_dormitory_room` 改为调用 |
| `room_output_score` | **不动**（宿舍不进产能分，测试锁死） |

#### 验收标准（全部通过）

1. `pytest tests -q`：**262 passed / 1 skipped / 1 expected-fail**（新增 8 个宿舍测试）。
2. 16 组可行性扫描全过，最差收支 **+4.0**（与 M1 前逐格相同，没有更差；7 格总恢复变好）。
3. 基准不退化：4 场景的混合分/图卷积分与 §5 表**逐位相同**（±0.0%），短名单 5/5 命中、
   top16 召回 1.0；252/342 的对照表也逐位相同。
4. 新增测试（`tests/test_optimizer.py::DormitoryFirstClassAssignmentTest`，8 个）：
   - 规格带等级（`_dorm_specs`）；
   - 每班次上床位数 ≤ capacity、同一人不能同时在岗和休息、跨宿舍不重复、离岗的人不漏；
   - 宿舍 beam 的账面恢复/账本恢复 ≥ 旧贪心（构造"工种分低但带全员加成"的干员，
     旧贪心会把他挤掉，beam 不会）；
   - 全员加成只按最大值计一次、自律节点只吃自己 +2/h；
   - `_dormitory_stay_value` 与 `audit_morale_cycle` 的 ledger **对得上**（同一套 rate 函数的锁）；
   - `result.score` 不受宿舍分配影响（把 `dormitories` 清空后重算分数不变 + 产能房完全不变）；
   - `_ensure_sustainable_morale_cycle` 在 beam 输出上仍然通过。
5. `generate` 端到端 243/333/342 各跑一次 exit 0（见 §4 末尾）。

#### 风险 / 陷阱（踩过的都在这）

- **宿舍价值与心情审计必须用同一套 rate 函数**。这条在实现上是
  `morale.dormitory_recovery_rates` 唯一入口；测试 `test_dormitory_stay_value_matches_the_morale_audit_ledger`
  就是防漂移的锁。别在优化器里再抄一份正则。
- **别把 beam 放回解码热路径**（`_build_shifts` / `_rebuild_dormitories`）。会 6× 墙钟，收益为零。
- **单目标（总和）会牺牲最差收支**：先用 `recovery`（总恢复）当目标时，
  252/342 的 2×12 组合最差收支从 7.2 掉到 6.0（总恢复反而略高）。所以最终是
  "三候选 + 审计账本按 `(门槛, 最差收支, 总恢复)` 字典序择优"，不是单目标最大化。
- **封顶效用（deficit）单独用也不够**：它会让 beam 不再奖励"全员加成"（因为室友已经补满），
  3×8 组合的总恢复反而降了 4~19。两个目标都要留着当候选。
- 图优化器是子类，父类解码改动会继承；`tests/test_graph_optimizer.py` 全绿，`_closure_override`
  与候选缓存不受影响（M1 没碰短名单/特征，读出层缓存继续复用）。
- 别把宿舍放进 `_room_specs_for_targets` 的目标枚举里 —— 会污染目标选择分。
- `_ensure_sustainable_morale_cycle` 仍然会**改写**计划（错峰展开），M1 之后也一样：
  结构性原因（在岗人数 > 床位数），与宿舍分配质量无关。

### 第 5 步（之后）：办公室选人

现状：`roomEfficiency` 已按「1% 办公室 ≡ 2% 贸易站」计入总分，但
`_assign_room` 对 HIRE 仍按 `score_skill` 的解析百分比排序，不走加权房间分、
也不吃中枢 `control_hire_spd`。改法：让 HIRE/MEETING 也走
`_score_room_combo` → `room_output_score`（`ROOM_EFFICIENCY_SCORED_ROOM_TYPES` 已包含它们），
即把它们加入 `_assign_room` 的 `_best_room_combo` 分支。

---

## 4. 复现用的诊断脚本

`tools/scan_m1_dormitory.py`（本轮新增，**没有删**）：跑 布局 × 模式 × 班次 的组合，
打印分数、`cycleHours`、最差收支、宿舍等级、总恢复与耗时，并把明细写成 JSON。
默认就是下面这 16 组；`--layouts/--modes/--shift-patterns` 可以只跑子集。

```powershell
python tools/scan_m1_dormitory.py --output outputs/scan_m1_dormitory.json
# 只跑紧凑的 2×12： --layouts 252,342 --shift-patterns 2x12
```

同目录还有：`tools/time_one_case.py`（单场景墙钟）、`tools/inspect_one_case.py`（审计摘要 +
宿舍分配）、`tools/inspect_worst_balance.py`（最差收支的干员与其休息/在岗班次）、
`tools/check_generate_outputs.py`（检查 `generate` 导出的 `cycleHours` / 门槛）。

---

### §4a 历史：修判定/补绿字的过程

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

### §4b M1 之后的 16 组扫描（同一脚本，逐格对比 M1 前）

| 组合 | 分数（前→后） | 最差收支（前→后） | 总恢复（前→后） |
| --- | --- | --- | --- |
| 243 普通 2×12 | 605.6 → 605.6 | 30.0 → 30.0 | 3891.4 → 3891.4 |
| 243 普通 3×8 | 563.5 → 563.5 | 14.0 → 14.0 | 5915.6 → **5920.4** |
| 243 平衡 2×12 | 380.3 → 380.3 | 30.0 → 30.0 | 3989.5 → 3989.5 |
| 243 平衡 3×8 | 354.9 → 354.9 | 20.0 → 20.0 | 3902.2 → **3908.1** |
| 252 普通 2×12 | 646.6 → 646.6 | 7.2 → 7.2 | 2243.9 → 2243.9 |
| 252 普通 3×8 | 599.7 → 599.7 | 4.0 → 4.0 | 2110.7 → **2122.8** |
| 252 平衡 2×12 | 359.9 → 359.9 | 7.2 → 7.2 | 2240.0 → 2240.0 |
| 252 平衡 3×8 | 333.1 → 333.1 | 4.0 → 4.0 | 2147.4 → 2147.4 |
| 342 普通 2×12 | 680.8 → 680.8 | 7.2 → 7.2 | 2244.6 → 2244.6 |
| 342 普通 3×8 | 632.4 → 632.4 | 4.0 → 4.0 | 2134.1 → **2135.2** |
| 342 平衡 2×12 | 546.6 → 546.6 | 7.2 → 7.2 | 2250.7 → 2250.7 |
| 342 平衡 3×8 | 499.0 → 499.0 | 4.0 → 4.0 | 2150.4 → **2158.2** |
| 333 普通 2×12 | 696.3 → 696.3 | 30.0 → 30.0 | 3889.7 → 3889.7 |
| 333 普通 3×8 | 641.5 → 641.5 | 20.0 → 20.0 | 3898.1 → 3898.1 |
| 333 平衡 2×12 | 544.3 → 544.3 | 30.0 → 30.0 | 3996.6 → **4008.6** |
| 333 平衡 3×8 | 498.0 → 498.0 | 20.0 → 20.0 | 3893.0 → **3899.0** |

- **16/16 通过，最差收支 +4.0**（与 M1 前逐格相同：`min Δbalance = 0.0`）。
- **分数逐位相同**（`max |Δscore| = 0.0`）——宿舍不进产能分这条锁住了。
- 总恢复 `min Δ = 0.0`，7 格变好（+1.1 ~ +12.1 心情/周期），一个都没变差。
- 单场景墙钟 +0.4% ~ +1.5%（隔离测：243 平衡 2×12 12.82s → 12.87s；333 平衡 3×8 20.01s → 20.31s）。
- `generate` 端到端（图优化器、平衡搓玉）：243 `cycleHours=144`、333/342 `cycleHours=48`，
  全部 exit 0 且 `hardGatePassed=True`，最差收支 93.6 / 31.2 / 7.2。

---

## 5. 基准现状（rebase 后，1 胜 3 负，均比 0.981）

| 场景 | 混合 | 图卷积 | 变化 | 模拟器调用比 |
| --- | --- | --- | --- | --- |
| 243 平衡搓玉 2×12 | 380.290 | 381.201 | +0.24% | 0.624× |
| 243 不搓玉 2×12 | 605.566 | 603.177 | −0.39% | 0.788× |
| 333 平衡搓玉 3×8 auto | 498.041 | 483.634 | −2.89% | 0.700× |
| 333 极限搓玉 2×12 auto | 504.514 | 482.235 | −4.42% | 0.760× |

**M1 之后重跑：四个场景的混合分与图卷积分逐位相同**（`outputs/benchmark_m1/`），
短名单仍然 5/5 命中、top16 召回 1.0，读出层缓存直接复用（没有重拟合）。
墙钟 13.00 / 9.24 / 25.88 / 20.48 秒（图卷积更快，平均墙钟比 0.884）。

333 落后的诊断（已排除）：加宽短名单更差（128 → −30.8，192/beam22 → −35.8）、
加第三个起手无变化。**推断**：短名单只优化房间产能分，而心情门槛给「宿舍恢复能力」定价，
混合算法穷举会碰巧选中休息友好的人。**§3 第 4 步（M1）做完后的结论：这条推断只对了一半** ——
宿舍分配确实变成一等目标了，但 16 组扫描里最差收支一格都没变、分数也没变，因为
**心情门槛在满练表上本来就有 +4.0 以上的余量**，宿舍再怎么分都不会改变"谁上场"。
要缩小 333 的差距，得动**搜索空间**（短名单/起手/插入预算）或**班次结构**（§3 第 3 步），
不是宿舍分配。

修完第 1、2、4 步之后 243/333 的分数没有变（605.566 / 380.290 / 498.041 全部复现），
因为这三步只改心情判定、心情模型与宿舍分配。

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

2 胜 2 负，平均分数比 1.000938，平均墙钟比 0.7777，平均模拟器调用比 0.731；短名单 5/5 命中、
top16 召回 1.0。**M1 之后重跑同样逐位相同**（`outputs/benchmark_252_342_m1/`），
墙钟 19.67 / 9.84 / 20.30 / 9.48 秒。
**规律：搓玉目标稳赢、不搓玉目标稳输**（243/252/342 三组都是这个符号），
机制见 `docs/graph_optimizer.md` §5 与 §6 第 2 条（合成玉贸易站零梯度 → 短名单在该房间随机，
混合算法穷举碰巧更好）。下一步要缩小不搓玉的差距，得动搜索空间或班次结构：
§3 第 4 步（宿舍一等分配）做完后已确认**不是**这个差距的原因（见 §5 的结论）。

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

## 7. 工作区状态（前四步都已提交）

第 1 步（`d78daad`）已提交：`morale.py` 收支记账判定 + `tests/test_morale.py` + `optimizer.py` 文案
+ `graph_benchmark.py` 容错 + `docs/*`。

第 2 步（`9e1b605`）已提交：绿字、作用域减耗、控制中枢、父类纯赤金 7 天上限。

第三步（单体宿舍恢复）也已提交，内容：

| 文件 | 改动 |
| --- | --- |
| `morale.py` | `dormitory_recovery_bonuses` 识别 `某个`（26 条真实命中）并返回 `(bonus, owner)`；新增 `_recover_dormitory_room` / `_lowest_unfilled` / `_advance_dormitory_occupants`，按「恒定速率段」精确模拟单体恢复的锁定语义；`assumptions` 增加一条 |
| `tests/test_morale.py` | 22 个（新增 3 个：最低心情舍友、不给自己、回满后移交） |
| `docs/*` | handoff §2.1/§2.2/§3/§4/§5、`graph_optimizer.md` §5b、`memory/next_steps.md` P12、`AGENT.md` 心情口径 |

第 4 步 M1（宿舍一等分配）的改动（提交信息：`Make dormitory assignment a first-class decode target (M1)`，
即 `git log` 里最新那个提交）：

| 文件 | 改动 |
| --- | --- |
| `morale.py` | 新增共用速率/记账函数 `dormitory_operator_bonuses` / `dormitory_recovery_rates` / `dormitory_single_target` / `control_dormitory_boost` / `morale_consumption_by_operator`；`dormitory_recovery_bonuses` 与 `_recover_dormitory_room` 改为调用它们（一套正则） |
| `optimizer.py` | `_dorm_specs` → `RoomSpec`（带 level/slots）；新增 `DormitoryRecoveryProfile` / `dormitory_plan_signature` 与 M1 方法组（`_assign_dormitory_quarters` / `_beam_dormitory_plan` / `_greedy_dormitory_plan` / `_select_dormitory_plan` / `_dormitory_plan_key` / `_beam_dormitory_rooms` / `_dormitory_fill_options` / `_greedy_dormitory_fill` / `_dormitory_insertion_key` / `_dormitory_stay_value` / `_dormitory_objective_value` / `_dormitory_candidate_objective` / `_dormitory_assignment` / `_dormitory_candidate_pool` / `_dormitory_anchor_key` / `_dormitory_profile`）；`_ensure_sustainable_morale_cycle` 落点；`_build_shifts` / `_rebuild_dormitories` 保持贪心 |
| `tests/test_optimizer.py` | 新增 `DormitoryFirstClassAssignmentTest`（8 个）+ 4 个宿舍 fixture helper |
| `tools/` | 新增 `scan_m1_dormitory.py` / `time_one_case.py` / `inspect_one_case.py` / `inspect_worst_balance.py` / `check_generate_outputs.py` |
| `docs/*` | 本文件 §0/§3/§4/§5/§7/§8、`graph_optimizer.md` §5b、`AGENT.md` 心情口径 |

---

## 8. 陷阱清单

- **下一步是 §3 第 5 步（办公室选人）或第 3 步（错峰轮转表）**。第 4 步（M1）已完成，
  别再当它是待办；也别再拿"M1 能补上不搓玉差距"当前提（§5 已证伪）。
- **宿舍分配不要放回解码热路径**（`_build_shifts` / `_rebuild_dormitories`）：那里一次
  `optimize()` 会调几百次，做 beam 实测 6× 墙钟、最终排班一模一样。beam 只在
  `_ensure_sustainable_morale_cycle` 里对最终排班跑一次。
- **宿舍速率函数只有一个入口**：`morale.dormitory_recovery_rates`。优化器侧不许再抄正则
  （`_dormitory_profile` 只是缓存 `dormitory_operator_bonuses` 的结果）。
  测试 `test_dormitory_stay_value_matches_the_morale_audit_ledger` 是防漂移的锁。
- **`_dorm_specs` 的等级必须走 `levels_or_default(layout.dormitory_levels, layout.dormitory, 5)`**，
  与审计的 `layout.dormitory_levels[i] if i < len(...) else 5` 同口径；否则 252/342
  （`(1,1,1,2)`）会按 4.0/h 定价、按 2.0/h 记账。
- **宿舍分配不能只最大化总恢复**：单目标会让 252/342 的 2×12 最差收支从 7.2 掉到 6.0
  （总恢复反而略高）。现在用三候选 + 审计账本 `(门槛, 最差收支, 总恢复)` 字典序择优；
  改目标函数时一定要重跑 §4 的扫描看 `worstBalance` 有没有掉。
- `optimize()` 现在会因心情门槛抛 `ValueError`；`recommend` 会把它记为 skipped candidate，
  CLI `generate` 会直接崩 —— 写新代码时记得 try/except。
- `_apply_pure_gold_drone_cycle_repeats` 里必须同时尊重 `MAX_CYCLE_HOURS`（168h），
  否则 `cycle_exceeds_seven_days`。
- 图优化器是 `ScheduleOptimizer` 子类，**新增的父类逻辑会自动继承**（心情门槛就是这样进来的）。
  改父类方法时要考虑子类的 `_closure_override` / 缓存失效。
- `docs/graph_optimizer.md` §5b 与 `AGENT.md` 的「关键口径」曾经写成「宿舍不进模型」，
  那是 rebase 前的旧基线；已更正为「宿舍参与心情硬门槛」，第 4 步之后又更正为
  「宿舍是一等分配目标（M1）」。不要再照旧说法写文档。
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
