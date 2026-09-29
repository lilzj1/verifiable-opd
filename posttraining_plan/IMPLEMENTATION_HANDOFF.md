# 后训练能力整合：实施交接与验收细则

2026-09-24；**CPU_PREP / WAIT_FOR_HGPO，不是训练就绪声明**。

主方案：[后训练与能力整合执行方案](../后训练与能力整合执行方案.md)。本文件把它落到现有代码，不另选场景。这里的 `text_*` 文件名和命令接口是**待实现约定**，不是已经存在的训练功能。本轮不改运行中的训练器、不下载权重、不启动服务、不删除文件。

## 1. 实现边界与已有资产

目标：同一 MixSFT 起点，整合 Math/IF 两专项教师，比较继续 SFT、普通多教师 OPD、OPD+固定专家 CE。只检验 CE 回放这一项改动，不加 GRPO、动态权重或失败优先回放。

| 本地入口（相对 OPD项目） | 已核实行为 | 新版处理 |
|---|---|---|
| `run_verl_profile.sh` | 旧算术工具循环、单教师配置、旧长度与回放预算 | 新增独立文本启动器；不得替换参数后直接当新版运行 |
| `verifiable_opd/verl_replay_batch.py:ReplayPool` | 从 `tools_kwargs.submit_value` 解析算术题，要求固定条数完整轨迹 | 新增 `text_replay.py`；从独立专家数据集取完整回答，不用在线 prompt 池合成答案 |
| 同文件 `make_expert_input` | 强制 `arithmetic_expert_replay_agent` | 新建文本专家编码器，禁止学生重新生成“专家答案” |
| 同文件 `restore_rollout_identity` | 缺失 extra_info 时按输入顺序补回 | 新链路验证异步返回顺序；用不可变 sample_id 对齐，不能对错序结果盲目复制 |
| 同文件 `concat_for_joint_update` | 补零字段后合并两分支 | 可借用结构；重新审计在线、CE、padding 的分支标志与统计分母 |
| `verifiable_opd/verl_mixed_loss.py` | sampled k1/PG、token-mean、独立在线与 CE 分母 | 保留数学目标，补新文本与 alpha=0 契约测试，不替换成 full-vocab KL |
| `upstream/verl-v0.8.0/verl/trainer/ppo/ray_trainer.py:_update_actor` | **仅 replay 模式**强制整个 batch 一个 mini-batch、一个 epoch | S2/S3 都必须落实“一批在线样本一次 optimizer.step”，防止组间更新次数不一致 |
| 同文件 `fit` 的 replay 分支 | 从 `train_dataset` 取专家、调用算术专家 loop；n=1 | 加可配置回放适配器和独立 replay dataset；保持旧入口可回归 |
| `upstream/verl-v0.8.0/verl/experimental/agent_loop/single_turn_agent_loop.py` | 已有 `single_turn_agent`，整个生成响应 mask=1 | 优先复用，不再造在线工具循环；核实停止原因/EOS 能否完整保留 |
| `upstream/verl-v0.8.0/verl/workers/config/distillation.py` | 多教师解析、资源核算、旧 replay 配置校验 | 按本文件第 3 节适配；不绕过校验 |
| `upstream/verl-v0.8.0/verl/experimental/teacher_loop/teacher_manager.py` | 按 routing_key 查询教师 | 保留并测未知域报错、重排后路由一致 |
| `checkpoint_rotator.py` | 检查非空分片、三类 rank 集相同；保留至少两份 | **不足以证明分片完整可加载**，新实验启动前增强，见第 8 节 |

旧 `check_verl_replay.py` 依赖 Qwen tokenizer/算术样本。保留它做旧链路回归；不得把它通过写成 SmolLM3/Math/IF 已通过。

## 2. 交付顺序与模块接口

所有新功能先放 `verifiable_opd/posttraining/` 或独立新脚本，待测试后再接入最少的 trainer 分支；不重写整个 verl。

| 顺序 | 待实现模块/接口 | 输出与完成条件 |
|---|---|---|
| P0 | `audit_assets.py`：revision、许可、配置、tokenizer/chat template | `MODEL_MANIFEST.json`、来源许可清单；不把模型许可套给数据 |
| P1 | `build_text_data.py`：源格式→规范化记录→固定 split | `online_train.parquet`、`expert_train.parquet`、dev、只读 test manifest、`DATA_CARD.md` |
| P2 | `text_encode.py`：在线 prompt/专家 response 同模板编码 | token IDs、response mask、长度、EOS/截断记录；金样本逐 token 可复核 |
| P3 | `text_replay.py`：固定池、按域固定预算抽样 | 已编码专家 batch、样本 ID、真实 token 成本、选择器状态；不用工具、环境或教师生成 |
| P4 | `text_adapter.py` 与 trainer 最小接入 | 标识、教师路由、CE 分支合并、全局分母在重排/切分后正确 |
| P5 | `eval_text.py`、`summarize_text.py` | 原始响应→冻结判分器→逐题结果→配对报告；不使用训练损失作成功率 |
| P6 | `run_text_experiment.py`/启动器 | dry-run 配置→审计通过后生成明确命令；S0/S1/S2/S3 独立目录；默认不启动 |
| P7 | checkpoint 协议与 profile 汇总 | 保存恢复、预算估算、阶段状态与结束退出码；不自动进入无限长训 |

每阶段更新 `STATUS_POSTTRAINING.md`：完成项、进行项、阻塞、下一步、配置 hash、证据路径。不要覆盖旧算术项目的结果文档。

## 3. 配置落地：已经找到的陷阱

草案在 `protocol.draft.json`。**它不是 Hydra 配置，不可直接传给 verl**；`allow_training=false`，未定字段显式为 null。后续配置编译器只允许白名单字段，输出 resolved YAML，拒绝未决字段与冲突覆盖。

### 多教师键与资源池

- 本地 `_resolve_teacher_models` 要求最初有 `teacher_model` 默认模板；多教师时会把这个默认项 pop 掉。
- 正确做法：保留默认模板用于配置合成，另建 `math_teacher`、`if_teacher` 两个条目，分别设置 `key: math`、`key: if`。**不能把数学教师仍命名为默认 `teacher_model`**。
- 两条目显式 `num_replicas=1`、TP=1、DP=1、PP=1，教师池为 2；学生池为 2。检查解析后只剩两域且总资源数为 4。
- 数据 `data_source` 统一规范为 math/if，但必须先核上游实际值，再显式映射；不依据字符串猜域。
- 物理 GPU 1/2/3/4 只是候选授权集合，Ray 的逻辑 GPU 编号不等于物理号。启动前保存 UUID、PCI bus、PID/归属和实际各 actor 设备映射；不靠 `CUDA_VISIBLE_DEVICES` 顺序就宣布分配正确。

### 更新口径

- 初始 profile 候选每批 4 个 online prompt（两域各 2）、n=1、epoch=1、微批每卡 1。不是从 HGPO 的 8 轨迹设置照搬。
- S2：4 个在线样本一次 optimizer.step；S3：同样 4 个在线样本加专家样本仍一次 step。记录运行时 mini-batch、累积微批次数与 optimizer.step 计数，不只看配置名。
- 旧配置禁止 replay 开启时 alpha<=0。新版 alpha=0 首选直接走 S2、禁用专家采样和额外前向；另用固定 batch 的 loss 单测验证 CE 系数趋零。不要把“仍抽专家但 alpha=0”当作完全相同的随机执行路径。
- S1 用独立的纯 CE 更新入口，不伪造 online token 凑过混合 loss 的非空检查。
- 首轮训练温度候选 1.0，避免 CE 被 rollout 温度隐式缩放；若改变温度，必须区分采样分布与 CE 原始 logits 的定义并测试。
- clamp、scheduler、thinking/EOS 仍待核实。现有 clamp 值不能悄悄当新实验默认；实际生效参数写入 `CONFIG_DIFF.md`。

## 4. 数据契约、模板与抽样

### 中间格式（不是原始 parquet 的字段假设）

每条记录：`example_id`、`source_repo/revision/path/row_id`、`domain`、`kind`（online/expert/eval）、`split`、`dedup_cluster_id`、`messages`、许可标识、原始文本 hash。

- `messages` 只含 system/user 的 role/content，以 user 结束。模型输入通过白名单生成，不能 dump 整条记录。
- `expert_response` 只存在专家记录里，经编码才成为 CE 目标；`ground_truth`、IF instruction IDs/kwargs、checker 参数只给评测器或离线审计，不能进入 prompt。
- `sample_id = run_id/update/index/domain` 区分同题多次采样；另保留不变的 example_id。教师请求、rollout、平衡排序、微批切分、预测结果都携带 sample_id。
- 每域原始字段、源标签映射、缺失字段策略写成适配器，不假定两个公开子集 schema 相同。上游 manifest 只作审计输入，不替代本地检查。

### 划分与泄漏

先建立规范化问题/变体簇，后划 train/dev/test；精确文本 hash 只是第一层，人工复核高相似候选。IF 去重不要抹掉改变约束含义的标点、大小写、数字。冻结 seed、抽样排序、排除原因及每域数量。

train online 与 train expert 可有同题，但必须记录重叠数量并固定；任何训练池与 dev/test 跨 split 同簇必须失败。公开模型此前见过哪些题无法完全确认，报告暴露限制。合成 CPU 测试不能证明真实数据已去污染。

### token 化验收

1. 对三模型比较完整 vocab→ID、special tokens、added tokens、chat template 和 thinking 模式；同系列不等于必然一致。
2. prompt 与专家完整对话用同一模板；验证 prompt token 前缀关系后取 response 范围，不能任意 `encode(answer)` 拼接造成边界错位。
3. CE：标签与原 input_ids 对齐，prompt/pad label=-100，由模型/计算函数**只移位一次**；第一答案 token 由最后 prompt 位置预测。对 thinking、EOS 的监督口径显式测试。
4. online：使用真实生成 token ID，不 decode→重新 encode 再送教师；教师对每个学生 token 前缀打分，不替换成教师自己的回答。
5. 长 prompt 先过滤并报告，而非静默左截断任务条件。专家回答超长先排除/单列，禁止截断后伪装完整示范；最终在线响应到长度上限算截断，不补造 EOS。
6. 原始响应和供 checker 使用的 final text 分开保存，final 提取规则三组一致，测试前冻结。

### 回放预算策略

profile 前先离线测每域专家 token 长度分布，再定每步专家 token 上限和完整回答数量。候选按域各分一半预算、固定 seed 均匀打乱池，跳过放不下的完整回答并记录；这会偏好短答案，报告入选长度/域占比。第一轮不做失败优先。

若一个域在预算内无任何完整示范，停下调整预算/长度方案，不偷偷只回放另一域。冻结 S3 预算和抽样策略后才训练。不足 DP 对齐时使用显式执行占位行（不伪装真实示范）；其所有 loss mask=0，不计任何有效 token 分母。

## 5. Loss 与分布式正确性验收

目标：`L = sum_online(loss_token)/N_online + alpha * sum_expert(CE_token)/N_expert`。这是全局在线 token-mean 加全局专家 token-mean，**不是两域等权，也不是两分支拼起来共用一个分母**。

若底层 DP 对梯度取平均，rank 的每个 micro-batch 贡献为 `DP_size * local_sum / global_branch_tokens`；同一次更新所有 micro 累加一次，不再额外除累积步数。若引擎本身缩放，必须消除双重缩放，通过梯度实测而非套公式确认。

CPU/后续 GPU 用例清单：

- 在线长度 `[1,3,2,5]`、专家 `[2,4]`；任意换 rank/micro 切分，与整批参考的 loss/梯度一致。
- 一个 rank 全专家、另一个无专家；单微批仅 padding；仍执行同样 collective/反向序列。全局所需分支为零则所有 rank 一致拒绝，不让一个 rank 单独抛错卡死其他卡。
- 占位行旧 logprob/teacher 字段为有限 dummy 值；注入 masked NaN 的负例确保不出现 `0 * NaN` 污染。
- alpha=0 的 S2 快路径一致；S1 纯 CE 与标准 teacher-forcing CE 一致。
- 数据重排前后每个 sample_id 的域、教师 revision、响应 token、mask 一致；未知域/缺失教师拒绝。
- 教师冻结：无梯度、无参数变化；学生有有限梯度且权重发生变化。仅 loss 下降不算验证更新。
- 16 条固定短样本核对 Transformers 与服务端同前缀 logprob，包含不同 padding、两域、EOS；先依据精度基线登记容忍值，不能失败后随意放宽。
- SmolLM3 的 remove-padding/梯度检查点/权重同步分别 smoke；注册表存在不算通过。失败优先关闭可选优化诊断，不升级 HGPO 共用环境。

本目录的 `test_contracts.py` **只实现前期 schema/配置负例检查**；上述 loss、服务端、真实分布式验收均尚未完成。

## 6. 教师门禁、profile 与冻结实验

先完成来源许可与本地 train/dev 构建，再读取本地 dev，不用公开最终测试挑模型或调长度。

1. 固定 dev 每域前 16 个 ID 作成本/长度探针，顺序预先冻结；不因题目不会而替换。只比较事先登记的 response=4096/8192，二者都保持相同问题和 prompt。
2. 选择能负担的统一长度后，三模型 × 两域 × 每域 64 个 dev，默认每题一次，共 384 个响应；少量探针与完整矩阵重复必须记为同题，不增加独立样本量。
3. 建议教师效果门槛：本域相对学生配对点差>0，报告置信区间；两个专家之间存在对应专项的互补证据。若样本太少导致不确定，最多按预登记规则补至每域128题、且只补一次。此为**候选门槛，必须在见矩阵结果前冻结**；点差为正本身不等于显著优越。
4. 教师检查包括解析有效比例、截断率和 prefix logprob，阈值应先由长度探针和成本约束登记；未冻结则只能诊断，不自动开始正式训练。
5. 小规模：S2/S3 各一次真实更新与保存恢复；接着各 3～5 步 profile，明确为调试数据。正式对照从干净 S0 重启，不复用 smoke 权重。
6. profile 报告初始化/rollout/teacher score/学生前反向/CE/保存/评测分别耗时，记录重叠阶段；不要把可并行的子阶段简单相加当墙钟。

预算冻结建议：S2/S3 各50更新、10更新阶段检查，scheduler 从开始按50定义。仅数值/数据/恢复/资源故障是硬停止条件，不能因为单个 dev 点不好看而只让一组少训后再声称等预算。

S1 的有效目标 token 上限在正式前由 S2 调试 profile 的每步平均目标 token×候选更新数估算并取整，写入协议；不看最终谁好再挑预算。S1 的更新数、循环专家池次数、最终批次超出预算 token 均报告。S3 的额外 CE 计算明确单列，此设计不能完全排除“额外监督和计算”因素。

默认 checkpoint 选择用最终计划步，而非四处挑最好一份；如要 dev-best，先固定每域指标和选择规则，并为其留保存空间。正式 test 只评选定的每组一个 checkpoint。

## 7. 评测与成本，先算完整再开训

候选最终测试每个 checkpoint：`(30+30)*4 + 541 + 300 = 1081` 次响应。S0/S1/S2/S3 共 **4324 次响应**，不含教师 dev 矩阵、探针、调试与定期 dev。数据实际数量变更时重新计数，不硬套这个数字。

只在 dev 做三教师/学生对照矩阵，第一轮不额外要求两教师完整公开 test。正式前估算总训练+评测成本，评测也可能占相当时间。

候选指标定义：

- AIME24/25 分别报告题内4次平均正确率，再两集合等权宏平均；不是“4次中有1次对就成功”。
- IF 的两个集合分别报告 prompt-level strict；instruction-level strict 单列。IF域分数候选为两个集合 prompt-level strict 等权平均；若对应 checker 不支持该口径，冻结前修订定义，不硬拼分数。
- 两域总分候选 `(Math域分数+IF域分数)/2`；报告每域相对 S0 点差、最差域变化，不用总分隐藏退化。
- 同一最终评测集按题目/约束任务配对重采样计算差值区间；AIME多次采样绑定同一题。推断只覆盖当前训练 seed，不能冒充多 seed 稳定性。
- 数值/解析失败、超时、截断、token数、峰值显存、GPU小时和有效更新数均保留；不只交一个准确率。

逐响应输出字段至少：run/checkpoint/revision、example/sample ID、domain、prompt hash、原始文本与token、finish_reason、checker版本与细项、score、生成耗时和token数。不要把标签放回推理服务日志输入。

汇总同时检查任务集合 hash、每题预期次数、重复/缺失、checker失败。模型错误保留为失败；基础设施故障是否重跑、重跑次数在协议里统一，保留所有尝试，不能只重跑错题。

## 8. 存储与恢复：复用思路，不直接信旧清理器

现有 `complete_checkpoint` 只比较 model/optim/extra 三类已找到的 rank 集。如果三类都缺同一 rank 仍可能返回真；未校验 world_size一致、文件可加载、checksum、保护列表或根路径符号链接。**正式前不能原样自动清理新实验。**

新版最低要求：

1. 每个 run 独立根目录，显式 run_id/路径白名单，拒绝越界与符号链接；不跨项目删除，不触碰公开起点、教师、旧结果。
2. staging 保存全部预期 `0..world_size-1` 的模型/优化器/额外状态及 data cursor；校验尺寸、hash、元数据和可读性；实际恢复 smoke 还需验证 scheduler/RNG/step、专家选择器状态。
3. 完成后原子提交 checkpoint manifest，再更新 latest 指针；控制器和清理器有单写锁；保存不完整/磁盘不足立即停止清理并报告。
4. 保留最新和上一份完整可恢复点；dev-best/final/正在评测引用的目录加入保护集合。新的完整点出现之后才删除明确过期者；先 dry-run 输出目标清单。
5. 峰值磁盘至少考虑三份共同资产、各组保留状态、正在写入的一份新状态、必要HF导出、数据/日志/临时空间。第一份真实 checkpoint 测量后估算，不用“模型6.2GB”推断优化器状态占用。
6. 新版轮换测试：缺同一rank、world_size冲突、0字节/损坏、写入中、stale latest、保护引用、重复控制器、路径越界。只能在临时目录做删测。

每组最终保留恢复点还是仅推理导出须在空间报告里明确；如空间不够，暂停请求取舍，不能默默删掉唯一可续训状态。位级连续等价需要额外复现实验，保存了优化器不自动等价。

## 9. 现在可运行的 CPU 准备检查

从 OPD项目目录运行，不加载模型、不请求网络、不创建GPU进程：

```bash
python posttraining_plan/check_contracts.py
python -m unittest discover -s posttraining_plan -p 'test_*.py' -v
```

第一个只检查**草案内部一致性**，输出未冻结字段、评测响应预算、关键现有源码 SHA256；不是启动门禁。第二个是合成数据单测，不代表真实数据/分布式链路通过。计划 schema 后续若变更须同步测试。

交接时的执行顺序：读主方案与本文件 → 检查 HGPO 状态与许可/预算 → P0/P1/P2/P3 CPU实施 → 明确GPU归属且资产准备齐全后教师探针 → P4/P5真实 smoke/profile → 报告预算并冻结 → 正式对照和最终评测。**不能因为本目录测试全绿就启动50步。**
