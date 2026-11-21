# Agent 记忆入口

这个项目是一个明日方舟基建排班生成/评分工具。后续 agent 进入项目时，先读本文件，再按需要读 `docs/memory/` 下的模块记忆。

## Python 环境

- 本项目专用 conda 环境：`arknights-schedule-generator`。
- 解释器：`C:\Users\12182\.conda\envs\arknights-schedule-generator\python.exe`。
- 当前验证版本：Python `3.11.15`，`openpyxl 3.1.5`。
- 安装/刷新命令：

```powershell
conda run -n arknights-schedule-generator python -m pip install -e . pytest
```

- 常用命令统一使用：

```powershell
conda run -n arknights-schedule-generator python -m ...
```

- 环境验证命令：

```powershell
conda run -n arknights-schedule-generator python -c "import sys, openpyxl; print(sys.executable); print(sys.version); print(openpyxl.__version__)"
```

```powershell
conda run -n arknights-schedule-generator python -m pytest tests\test_production.py -q
```

```powershell
conda run -n arknights-schedule-generator python -m pytest -q
```

```powershell
conda run -n arknights-schedule-generator python -m arknights_schedule_generator calibrate --data-dir data\cache --profile all --output outputs\calibration_check_session.json
```

- 不要把 `fdm-examiner` 当作本项目环境；那是其他项目语境遗留。
- `traepython` 是当前 shell 默认环境，但缺少 `openpyxl`，不能作为本项目权威验证环境。
- `base` 只可作为历史诊断参考，不是本项目隔离环境。
- 若 `docs/memory/` 或 `outputs/` 中的历史记录还出现 `fdm-examiner` 或裸 `python -m ...`，先确认它是不是历史证据；当前开发与验证以本节环境口径为准。

## Git 操作约定

- 每次完成一轮文件修改、测试、生成输出或其他关键操作后，都要运行 `git status --short --branch`，确认当前分支和变更范围。
- 默认远端为 `origin`，地址为 `https://github.com/ChineseCricket/arknights_scheduel_generator.git`；默认主线分支为 `main`，后续提交和 push 都基于 `main` / `origin/main`，除非用户明确要求临时分支。
- 需要交付代码时，先完整检查并 stage 预期变更，再在本地 commit，并在回复中明确告知 commit 哈希和提交信息。
- push 是单独的发布动作：只有用户明确要求 push 时才推送。push 前必须先列出 `git log origin/main..HEAD --oneline` 和 `git status --short --branch`，让用户知道将被推送的本地提交范围；同一分支上的 push 会推送所有尚未推送的本地 commit，若存在不希望进入远端的中间提交，先 squash/rebase 或按用户要求整理历史后再推送。
- 不要回滚或覆盖用户/其他 agent 的未说明变更；若必须处理冲突，先说明具体文件和风险。

## 项目目的

- 从一图流网站导出的用户干员练度表 `.xlsx` 和一图流排班 JSON 出发，生成或评分明日方舟基建排班。
- 重点支持“搓玉”和“不搓玉”两类目标，输出每日预期 `龙门币 / 经验 / 合成玉`，同时报告赤金、源石碎片、材料和龙门币成本净变化。
- 最终希望生成没有同班冲突、能长期循环、接近社区成熟攻略收益的一图流兼容排班 JSON。

## 接手入口（图卷积 / 心情模型）

- 涉及图卷积优化器、心情门槛、宿舍建模的工作，先读 `docs/handoff_graph_optimizer.md`。
  里面有：现状、已做过的实验（别重复）、心情/宿舍的正确公式与数据来源、
  `morale.py` 的收支记账判据与已修的偏差、修正顺序、以及工作区未提交改动清单。
  **下一个窗口从 handoff §10 开始读**（模式目标函数 + 混合搜索作者的口径澄清）。
- 心情模型的当前口径（2026-11）：判据是「每个干员一个周期的收支记账」
  （`moraleLedgerByOperator`），不是「周期末心情 ≥ 周期初心情」；宿舍恢复 = 白字
  （1.6~2.0/h）+ 绿字（`0.0004 × 氛围`，满氛围 L5 再 +2.0/h）+ 技能；控制中枢每名在岗干员
  让所有在岗干员消耗 −0.05/h；无法归属的技能条款列在 `unmodelledMoraleEffects` 里而不是静默计 0。
  细节和实测见 handoff 的 §2 与 §4。
- **打分口径 v3（2026-11）**：唯一价格表在 `production.py` 顶部——
  龙门币 `6/1000`（收入=支出，唯一价）、玉 `1.2/点`（= 200 龙门币/玉）、经验 `4/1000`、
  碎片不单独计价（缺口由硬约束 `orundum ≤ 碎片 × 10` 兜住，盈余=库存）、赤金只在**逐房**目标里
  按转换价值 `3.0/片` 计价。逐房目标 `room_output_score` 与全局 `score_breakdown_for`
  必须用同一张表（有测试锁死）。**历史分数表都是 v2 口径**，跨口径不能直接比绝对值。
  细节见 handoff §9.6。
- **两个搓玉模式的目标不一样（混合搜索作者 2026-11 澄清）**：
  `balanced-orundum` 追求**龙门币收支平衡**（碎片成本必须由排班自己的龙门币收入覆盖，
  所以 243 平衡搓玉的正确量级是 ~200 玉/天，不是 492）；`max-orundum` **可以吃龙门币库存、
  且不假设库存有上限**（不设 cap、不用平衡罚项）。实现方案见 handoff §10.3。

## 当前可信状态

- 收益模型已经从“技能百分比总分”转为资源流模型，核心字段是 `dailyExpected` 和 `scoreBreakdown`。
- 现在有两套优化器，共用同一收益口径、同一约束和同一 `OptimizerResult` 契约：
  - `hybrid`（默认，`optimizer.py`）：房间组合全量枚举 + 插入/替换局部搜索。
  - `graph`（`graph_optimizer.py`）：固定超参图卷积短名单 + 同模拟器精确复核。
    数据审计、算法与实测对照见 `docs/graph_optimizer.md`。
    它有两种读出层形态（`GraphOptimizerConfig.readout`）：`per_room`（默认，每间房各拟合一套）
    与 `affinity`（M3b：房间节点 × 干员嵌入的共享亲和度矩阵；读数更准但排班分数 3 胜 2 负 3 平，
    所以**默认不开**）。细节见 handoff §3 第 6 步 M3b 与 `docs/graph_optimizer.md` §2.8。
- 实测（满练 447 行练度表，**价格口径 v3**）：图卷积 vs 混合 **1 胜 2 负 1 平、平均分数比 0.991**，
  模拟器调用 0.842 倍、墙钟 0.954 倍；252/342 是 1 胜 2 负 1 平、0.993。
  （v2 口径下是 0 胜 3 负 1 平、0.988——两列一起移动，所以只看符号分布。）
  逐房间短名单在 5/5 采样规格上仍取到全量枚举最优。
  （历史上那条「平均 1.030 倍」是校准 v2 之前的旧口径，不要再引用。）
- 图卷积的静态图信息来自 `data/cache`：55 个阵营组、36 对技能点名羁绊、2252 条组合机制边。
  数值强度不在数据字段里（`buff.efficiency` 多为 0），所以读出层用 `ProductionSimulator` 当老师标定。
- 一图流 2026-06「泡影苍霆」版本样例已作为主要校准源：
  - 视频源：`BV19jVZ69Evp`
  - 静态 `243 搓玉 一天两换`：标注 `30.6k LMD + 14.7k / 18.9k EXP / 582 玉`，公式复现通过。
  - 静态 `右满 342 搓玉 一天两换`：标注 `47.0k LMD + 13.7k / 0 EXP / 578 玉`，公式复现通过。
  - 动态 `右满 342 搓玉 动态换班 跑单`：标注 `77.0k LMD / 0 EXP / 540 玉 / 68.0k 赤金等值`，已绑定 Mower 计划 `1775555941084837`，以 `guide_calibrated_mower` 方式匹配。
- `outputs/production_calibration_report.json` 是机器报告；`outputs/production_calibration_report.md` 和 `.html` 是人工阅读版。

## 常用命令

```powershell
conda run -n arknights-schedule-generator python -m arknights_schedule_generator.cli update-data --data-dir data/cache
```

```powershell
conda run -n arknights-schedule-generator python -m arknights_schedule_generator.cli score `
  --schedule "C:\Users\12182\Downloads\324搓玉_260624.json" `
  --roster "C:\Users\12182\Downloads\干员练度表.xlsx" `
  --mode balanced-orundum `
  --metric-profile guide `
  --shard-formula rock `
  --drone-policy none `
  --output outputs/user_243_orundum_score_recalibrated.json
```

```powershell
conda run -n arknights-schedule-generator python -m arknights_schedule_generator.cli generate `
  --roster "C:\Users\12182\Downloads\干员练度表.xlsx" `
  --data-dir data/cache `
  --layout 243 `
  --mode balanced-orundum `
  --shift-count 2 `
  --shift-hours 12 `
  --shift-times 08:00,20:00 `
  --output outputs/schedule_243_balanced_orundum.json
```

```powershell
conda run -n arknights-schedule-generator python -m arknights_schedule_generator.cli calibrate `
  --data-dir data/cache `
  --profile all `
  --output outputs/production_calibration_report.json
```

```powershell
conda run -n arknights-schedule-generator python -m unittest discover -s tests -p 'test*.py'
```

```powershell
conda run -n arknights-schedule-generator python -m arknights_schedule_generator.cli graph-audit `
  --roster examples\fixtures\yituliu_full_roster_maxed.xlsx `
  --data-dir data\cache `
  --output outputs\graph_audit.json
```

```powershell
conda run -n arknights-schedule-generator python -m arknights_schedule_generator.cli benchmark `
  --roster examples\fixtures\yituliu_full_roster_maxed.xlsx `
  --data-dir data\cache `
  --output-dir outputs\benchmark
```

- `benchmark` 默认跑 4 个场景、每场景两套算法，耗时数分钟；用 `--cases` 只跑子集
  （`243-balanced-2x12` / `243-normal-2x12` / `333-balanced-3x8-auto` / `333-max-2x12-auto`），
  或用 `--layouts/--modes/--shift-patterns/--drone-policies` 自定义组合。
  （252/342 在心情模型补上宿舍绿字与控制中枢减耗之后已经可行，只是默认基准保持历史四个场景可比。）
- 新旧算法都能用 `--optimizer hybrid|graph` 切换（`generate` 和 `recommend` 都支持）。

## 关键口径

- 排班文件名不可信。用户文件名可能叫 `324搓玉`，但布局必须优先读 JSON 内 `scheduleType`；用户样例实际是 `2贸4制3电`，即 `243`。
- `lmdGross` 是贸易站毛龙门币收入；`lmdNet` 会扣除搓玉制造源石碎片的龙门币成本。社区攻略里常见的 `4w+ LMD` 多数对应 `lmdGross` 或含额外加速，不等于 `lmdNet`。
- 紫色加号是资源特定的额外加速参考值，不属于基础 `dailyExpected`，也不能把 EXP/LMD/赤金三个紫色加号同时相加。
- 绿色 `0.843/0.792` 是排班侧指标，不是 LMD、EXP、赤金或合成玉收益字段。
- `drone-policy none` 是基础产能口径；需要看无人机收益时单独跑 `lmd-trade`、`gold-factory`、`exp-factory` 或 `auto`。
- `scoreBreakdown` 里新增的 `roomEfficiency` 是**社区口径的房型增益分**，与资源流分项并列但口径不同：
  以 1% 干员增益为单位，贸易站=1、制造站=1、发电站=0.5（2%≡1% 贸易）、会客室=1/12（12%≡1% 贸易）、
  办公室=2（1%≡2% 贸易）。只有发电/会客/办公室进总分（贸易/制造已由资源项计价，只作对照）。
  换算锚点从既有常量推导（`TRADE_EQUIVALENT_PERCENT_SCORE ≈ 0.154` 分/%），不要改成魔法数字。
- 会客室/办公室的旧明细 `meeting`(`meetingSpeed/30`) 与 `office`(`officeSpeed/30`) **保留不动**，
  用于和既有校准口径对比；不要用它们代替 `roomEfficiency` 做跨房型比较。
- 心情模型口径（`morale.py`）：宿舍恢复 = 白字 `1.5+0.1×等级` + 绿字 `0.0004×氛围`（满氛围按
  `min(5000, 1000×等级)` 计）+ 技能（自身 / 所有干员 / 单体）；单体技能是**锁定**语义 ——
  目标=进入宿舍时心情最低的舍友（除自身以外），回满或离开才移交，不要改成每帧重算。
  菲亚梅塔「自律」是**恢复自给自足的特殊节点**：只按自身 +2/h，不接任何其他恢复源，
  图侧标注在 `graph_features.meta["specialMoraleNodes"]`（不加 GCN 特征）。
  菲亚梅塔「患难之交」按**手动心情重置**建模（审计 `version: 4` 的 `hardshipSwaps`）：
  持有者的**盈余**（recovery − consumption）转给最缺心情的干员，接收方 recovery += x、
  持有者 consumption += x（转移不生成，总收支不变）；**不要**回去建模「前一位/后一位」。
  在岗消耗 = 1/h + 自身/房间/中枢/全基建技能修正，`max(0, ·)`；
  控制中枢每名在岗干员让**所有在岗**干员消耗 −0.05/h（`controlData.basicCostBuff = -5`）；
  周期可持续 ⟺ 每人一周期 `潜在恢复 − 消耗 ≥ 0`。别再写回「末值 vs 初值」的判定。
  换班仍是**批量固定时刻**（一个 `ShiftPlan` 一个 `start`，全员同时上下班）。
  发电站数量→电力→宿舍等级→恢复率**已建模**（`power.resolve_power_layout`，252/342 会落到
  `(1,1,1,2)`，恢复率 2.0~2.5/h）。
- **搓玉（O_DIAMOND）口径 = 校准 v2（2026-11 已修，别再按旧说法写）**：源石订单的**速度只来自
  在岗干员技能**，并且**不裁剪订单上限**（真正的约束是源石碎片供给）；绝对量级由单一常数
  `production.ORUNDUM_ORDER_THROUGHPUT_SCALE`（当前 1.0，由参照点反解）锚住，
  参照 = L3 / 3 工位 / 技能 +132% / 无中枢加成 / 12h ⇒ **282.000** 合成玉。
  旧的 `guide_orundum_trade_lookup`（任何 L3 `O_DIAMOND` 都覆盖成 132%）已**删除**：它让 4750 个
  组合全产出 282.000、读出层 holdout R² = −110。现在同一批组合是 123.6 ~ 339.6，R² ≈ 0.75。
  改这条要先跑 `tools/measure_orundum_anchor.py` 与两套基准（基准数字会动）。
- 宿舍分配口径（M1，2026-11）：宿舍是**一等分配目标**。`_assign_dormitory_quarters` 对**整条排班**
  跑一次 beam（`deficit` 先补缺口 / `recovery` 先最大化总恢复）+ 旧贪心共三个候选，
  用心情审计自己的账本 `(hardGatePassed, 最差收支, 总恢复)` 择优，所以不会比旧贪心差。
  落点是 `_ensure_sustainable_morale_cycle`，**不要**放回 `_build_shifts` / `_rebuild_dormitories`
  （解码热路径，实测 6× 墙钟、结果不变）。宿舍速率只有 `morale.dormitory_recovery_rates` 一个入口，
  优化器不许另写正则。宿舍**不进产能分**（`room_output_score` 不动，测试锁死）。

## 最近用户样例结论

- 用户 `C:\Users\12182\Downloads\324搓玉_260624.json` 通过 `scheduleType` 识别为 `243 / 3班`，当前导入按 `8h/8h/8h` 评分，不是每班 12 小时。
- 最新保守评分：
  - 无无人机：`lmdGross 12526.13 / lmdNet -46225.87 / exp 10106.67 / orundum 387.2`
  - 无人机给 LMD：`lmdGross 17812.86 / lmdNet -40939.14 / exp 10106.67 / orundum 387.2`
- 这个结果低于一图流 243 搓玉标注，主要因为用户贸易站使用了 `但书`、`巫恋`、`可露希尔` 等特殊机制组合，而当前模型只对少数一图流速查表组合直接校准。

## 后续开发原则

- 先校准，再优化。不要把未校准的“近最优”结果当成可信收益。
- 未建模复杂技能必须进入 `unsupportedSkillEffects` 或显式假设，不能静默计 0。
- 新增校准口径时，优先把来源、图上标注、程序复现差异写进报告和测试。
- 修改文档时注意 UTF-8。PowerShell 控制台可能乱码，必要时用 Python 读回检查字符码。
