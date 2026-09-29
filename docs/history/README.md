# Verifiable-OPD：多步任务的在线蒸馏与SFT回放稳定化

> 2026-09-26 10:07 UTC：S2 三次连续更新全部通过（平均 602.93 秒/更新，无保存耗时），S3 连续验证已完成第 1 次并成功同步回推理。09:36 的停止是 Ray 最后一步 stdout 未转发导致的汇总误判，已从原始 worker 日志恢复，未重跑 S2；新运行逐步指标同步落盘。87 项 CPU 回归通过。完整接力 `opd_text_finish_v1` 已在 tmux 等待 S3 profile 与 S1 保存/恢复通过，再冻结两组各100更新、共同学生基线、正式对照、最终评测及报告。正式训练尚未开始，项目尚未完成；以 `artifacts/text_finish_controller_v1/status.json` 和 [运行手册](posttraining_plan/TEXT_RUNBOOK.md) 为准。以下旧状态按时间保留。

> 08:57更新：S3已更新并保存，但随后vLLM唤醒OOM，完整运行未通过。已修缓存释放顺序，正由 `opd_text_source_recover_v3` 复用保存点做恢复；`opd_text_profile_v2` / `opd_text_sft_smoke_v1` 等待接力。纯OPD首micro归一化顺序和单token轴处理也已修正，81项CPU回归通过，GPU连续更新仍待通过。用户接受更高预算，现规划正式S2/S3各100更新；正式训练与最终评测未启动。不要把下方历史状态/旧算术成绩当成新方案结果。

> 2026-09-26 08:34 UTC：新 Math/IF 双教师方案已进入真实更新。S2（纯 OPD）一次更新、完整保存及独立双卡恢复已通过；S3（OPD＋固定专家 CE 回放）更新验证正在运行。用户已取消“小 dev 教师优势”作为训练前置门禁，历史未通过结果仍保留；不重复教师优劣试验。后续控制器等待两组恢复通过，再各做 3 次不保存权重的连续更新 profile。正式训练和最终评测尚未完成。最新证据见 [STATUS_POSTTRAINING.md](STATUS_POSTTRAINING.md)，运行入口见 [TEXT_RUNBOOK.md](posttraining_plan/TEXT_RUNBOOK.md)。下方旧模运算结果不作为新方案结果。

> 2026-09-24 最新定位：以[后训练与能力整合执行方案](后训练与能力整合执行方案.md)为唯一升级主入口，优先研究公开 Math/IF 专项教师向同一学生的能力整合与 SFT 回放，不强行绑定 Agent。此前订单和 Coding Agent 方案仅保留为调研历史。**目前仅设计，待 HGPO 工作完成后再实施；未下载新权重、未完成新架构验证、未冻结训练预算。** 以下仍是旧模运算项目的已完成事实，不把新方案计入成果。

2026-09-21立项并完成第一轮冻结实验。项目已经跑通**精确环境与数据协议 → Qwen2.5-1.5B SFT冷启动 → Qwen2.5-7B LoRA教师门禁 → verl多轮在线蒸馏 → 均匀/失败感知专家CE回放 → FSDP保存与HF合并 → dev/ID/OOD评测**。三组首轮正式训练均为seed=17、50 steps；后续将纯OPD与均匀回放完整恢复并续训到200 steps，完成全量冻结评测。首轮结果见 [FORMAL_RESULTS.md](FORMAL_RESULTS.md)，延长实验见 [FORMAL_RESULTS_S200.md](FORMAL_RESULTS_S200.md)，完整复现说明见 [RUNBOOK.md](RUNBOOK.md)。这仍是单seed、小测试集的工程实证，不包装成已证明的新算法。

## 1. 为什么选这个，而不是照抄参考简历

参考图的第二个项目包含可执行数值任务、专家SFT、在线OPD、GRPO对照及冻结评测。借鉴的是实验链路，不复制其数据、指标或个人贡献。它展示顺序训练/对照，不能仅凭截图推定采用OPD与GRPO联合loss。

实际完成链路：**任务生成与精确验证 → 教师能力检查 → SFT冷启动 → 学生在线rollout → 固定教师在学生历史上打分 → OPD更新 → 加入少量SFT回放的对照 → ID/长程OOD评测**。

项目名中的“可验证”是可执行规则验证，不意味着已经超越基线。用户已确认SFT/OPD或GRPO/OPD均可，资源为本机3～4张A6000。选择前者作为主要研究问题：已有HGPO项目展示RL经验，这个项目补教师蒸馏、学生状态分布和稳定化。GRPO不从第一天加入，不加多教师。

术语约定：SFT→OPD是分阶段训练；OPD+SFT回放是在同一次优化更新中加入独立专家batch的CE目标；OPD+GRPO则增加可验证任务奖励驱动的策略目标。三者不能混称。TRL GKD的on/off-policy概率混采也不能直接称为本项目的硬标签SFT回放。

## 2. GitHub调研与选择

核查内容为作者README、官方文档及verl官方启动脚本；未声称所有项目都已复现。

| 项目 | 为什么看 | 本项目决策 |
|---|---|---|
| [verl官方OPD](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md) | 官方训练底座，已有OPD及与任务loss组合的配置；贴合已有verl经验 | **主底座**，独立环境与版本，不改day56实验仓库 |
| [Rethinking-OPD](https://github.com/Thinking-Space/Rethinking-OPD)（原thunlp/OPD） | 研究师生兼容性与OPD失败，提供训练/评测脚本 | 方法诊断与基线参考；不再维护完整fork |
| [OPDVR](https://github.com/LeapLabTHU/OPDVR) | 已有可验证奖励门控、OPD基线等方法 | 用来检查已有工作；**正确性门控不能包装成我们的原创** |
| [slime OPD](https://github.com/THUDM/slime/tree/main/examples/on_policy_distillation) | 提供另一套OPD与RL训练路径 | 工业框架对照，第一版不换整套后端 |
| [TRL GKD](https://huggingface.co/docs/trl/main/en/gkd_trainer) | 模块较小，适合学习student采样与teacher分布监督 | 教学/低预算备选；不是自动支持本项目多轮环境 |
| [OPSD](https://github.com/Xuyan923r/OPSD) | 教师与学生使用不同上下文的自蒸馏路线 | 选读边界；第一版不混同固定强教师OPD与自蒸馏 |

候选提交记录见 `upstream_versions.json`。实际底座为独立的 `upstream/verl-v0.8.0`，提交 `7aed6b230776f963fa09509c10d9c3a767d1102c`，来源及归档哈希见其 `OPD_SOURCE_LOCK.json`。该固定版本已通过真实四卡profile、三组正式训练、FSDP保存与合并；没有修改已有day56环境或仓库，也没有将候选main版本冒充已验证版本。

## 3. 任务与自己的小改动

采用可程序生成的模运算指令链：初值、加法/乘法指令、模数均公开。每轮学生提交一个JSON数值动作，环境执行并返回后续观测。环境有明确步数、失败与终止协议，精确oracle生成专家轨迹，不用LLM judge判定算术真伪。

两种训练环境：

- Sparse：合法数值动作直接进入下一轮，只有最后给整条轨迹是否全部正确的奖励。
- Constraint：同样的题目，但错误动作被拒绝，返回规则校验失败，可在调用预算内修正；不直接返回正确数值。

**主改动：在OPD更新中加入少量固定专家样本SFT回放。** 目标为 `L_OPD + alpha * L_SFT`，专家数据仅来自train，SFT CE与OPD各按明确有效token分母归一化，再加权。alpha=0是纯OPD基线。本轮固定alpha=0.1，不做自适应调权、不加入GRPO；检查动作合法性、长程成功率、教师成本和专家样本复用成本。该系数是训练前冻结的工程配置，不是用test挑出的最优值。

SFT回放是已有稳定化思路，不宣称原创；自己的工作是多步环境接入、严格mask/teacher-prefix一致性、SFT回放接入和同预算实证。官方OPD+task-reward开关**不是**SFT回放开关；后者需单独实现专家batch前向及CE分支，不能换个配置名字就宣称完成。

进一步的探索是**失败感知专家回放**：先采样学生轨迹，在固定专家token上限内优先选择本次失败任务的完整正确专家轨迹，其余预算用train池均匀回放补充。与均匀回放共用SFT起点、alpha和预算上限。注意：相同上限不保证实际使用token数相等，必须报告实际消耗；这也不是在学生错误状态上调用oracle的DAgger，不宣称新算法或必然有效。

原拟反馈退火课程改为后续选做，第一版固定同一种训练环境，避免同时改变loss与环境。Sparse/Constraint仅用于定义环境边界和排查反馈依赖，不要求首轮训练两个环境。

Constraint与Sparse的状态转移不同，课程是训练分布变化；不能把效果全部归因于OPD。最终所有方案统一在Sparse环境评测，控制最大调用/token预算，单独报告拒绝、修复和超时比例。

`task_env.py`仅兼容导入，环境唯一实现位于 `verifiable_opd/environment.py`。`reference_train.py`实现Transformers参考训练，用于先验证学生采样、同prefix教师评分、mask与loss；它的结果不冒充verl结果。正式verl路径已接入在线样本与专家样本分离构造、跨rank全局token分母、OPD+CE联合更新、失败感知选样、异常工具调用收口和FSDP恢复状态，并通过真实四卡profile及三组50-step训练。

## 4. 师生、预算与阻塞条件

- 学生为Qwen2.5-1.5B-Instruct，使用768步专家SFT checkpoint作为三组共同起点。原始Qwen2.5-7B-Instruct的自主工具执行门禁未通过，因此没有硬做蒸馏；随后只用train专家轨迹进行LoRA领域适配，并在held-out dev24的teacher-forcing门禁上确认教师优于学生（NLL 0.1129 vs 0.1765，token accuracy 0.9600 vs 0.9387）。该门禁证明评分优势，不代表教师自主执行完美。
- tokenizer、token ID映射、词表、chat template及特殊token逐项核验；“同系列”不是自动兼容证明。
- 教师门禁只使用dev专家动作，不看ID/OOD test；正式训练中教师冻结，只在学生真实prefix上给动作token打分。
- 教师冻结并只做前向；学生更新。普通只返回回答文本的聊天API不能冒充学生前缀上的teacher logprob接口。
- 已按单元测试 → 真实1-step三分支profile → 10-step异常路径profile → 50-step正式对照推进；没有把CPU参考后端或早期失败运行算作verl正式结果。
- 用户授权本机GPU 1、2、3、4。实测布局为GPU 1/2上的学生FSDP与两份rollout副本、GPU 3/4上的两份单卡冻结教师副本；主机无P2P/NVLink，显式关闭NCCL P2P/IB后稳定运行。
- 官方当前示例默认8B学生/32B教师、多GPU，不可原样当作我们的低预算配置。教师资源池与学生资源池需分别核算；LoRA接入也要实测，不能仅写在技术栈中。

## 5. 实验设计与证据

第一轮冻结预算实际完成前三个训练方案；Base/SFT是共同起点和评测参照，其余方案保留为后续扩展，不能写成已跑：

| 方案 | 作用 |
|---|---|
| Base | 确认任务难度和可用性，不训练 |
| SFT | 同一专家集、同一初始化的冷启动基线 |
| SFT→OPD（Sparse） | **已完成**；核心蒸馏基线 |
| SFT→OPD+均匀SFT回放 | **已完成**；同一SFT起点，alpha=0回归为纯OPD |
| SFT→OPD+失败感知SFT回放 | **已完成**；检验回放样本选择是否有效 |
| SFT→继续SFT | 排除额外监督量本身带来的增益；预算允许时纳入 |
| SFT→GRPO（Sparse） | 相同起点，区分教师反馈与终局奖励 |
| SFT→OPD+GRPO | 仅在确认需求和两条独立链正确后加入，不与顺序训练混称 |

同题组GRPO若n=1通常没有有意义的组内比较；组大小必须显式配置。纯OPD不需要把教师分数再按GRPO组内标准化。任务reward、蒸馏反馈、reference KL分别记录，不重叠计入。

数据按完整任务内容哈希去重，再冻结train/dev/test。训练长度初拟3/5/8，ID同分布独立题，OOD长度12/16；数值与指令采样规则需另记录，不能只靠不同seed保证无重复。正式规模先profile再确定。阶段一不宣称公开benchmark成绩，后续若加GSM8K则另列协议，合成环境结论不直接外推。

指标：episode成功率、分长度成功率、每步正确率、修复率、非法动作率、调用/token数、teacher评分token量、GPU小时、吞吐、峰值显存。测试时不提供教师/专家提示；统一解码协议，固定test不调参。有预算优先两个训练seed；只有一个则明确局限。等更新步数不等于等计算成本，比较同时报告实际学生/教师/环境成本。

## 6. 教学与实施顺序（P1～P8，每次约3～4小时，不含正式训练等待）

1. P1：飞书OPD两篇选段 + 官方OPD概念；解释谁采样、谁评分、谁求导、为什么teacher生成后SFT不是OPD。
2. P2：GitHub检索并固定版本，读官方启动脚本；确认资源、模型、tokenizer及隔离环境。
3. P3：运行本地环境测试，补数据去重与split清单；用oracle生成专家示范，建立Base/教师探针。
4. P4：SFT最小更新及冻结评测；验证输出协议与参数确有变化，不借用他人简历分数。
5. P5：先跑官方单轮小样本OPD验证teacher scoring，再接多轮环境，检查shift、mask、师生上下文、有限loss/梯度、保存恢复。
6. P6：接入均匀/失败感知专家CE回放；检查训练集隔离、label shift与mask、双loss分母及alpha=0一致性。区分预算上限与实际token消耗，不同时改反馈环境。
7. P7：完成SFT/OPD/OPD+SFT回放对照；按实测GPU小时与teacher-token预算分析。GRPO与反馈课程只在预算允许时扩展，不作为首轮硬要求。
8. P8：复盘失败案例、成本与泛化，形成图表、复现命令和诚实的简历表述。

无需GPU的回归入口：

```bash
python -m unittest discover -s 能力补全/OPD项目 -p 'test_*.py' -v
```

完成定义仍保持分层：环境测试 ≠ 随机小模型CPU更新 ≠ 真实模型GPU训练 ≠ 有效对照结果。第一轮后三层均有独立证据；正式结论只引用verl日志和冻结评测，不引用参考后端成绩。`analyze_formal.py`可从原始日志与评测trace重新生成结果表。

## 7. 简历替换规则

工程侧P5～P8证据已经落盘，可以写在线蒸馏链路、分布式联合loss、工具调用稳定性和失败诊断；性能数字必须连同单seed/小测试集边界一起写。SFT回放没有提高本轮长链成功率，不得写“显著提升”；可以如实写均匀回放降低非法动作、失败感知策略未奏效及其分析。项目属于基于开源框架的实现与实证，不将官方OPD loss或OPDVR门控算作自己的创新。学习者完成源码复现和答辩前，不急于替换正式简历。
