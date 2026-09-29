# Math/IF 后训练升级：实际实施状态

## 2026-09-26 10:30 UTC：修正 S1 CPU placement 资源不足

- S2、S3 连续三次真实更新 profile 均已完成，两个 rank 的实际 optimizer 更新有记录；不重复执行。
- 随后的 S1 smoke 在模型更新之前阻塞：入口 `ray.init(num_cpus=16)`，上游 SFT 默认每 GPU 申请10个CPU，两GPU合计20。日志反复出现 `GPU: 2, CPU: 20` 无法调度，并非 GPU OOM。
- 只终止此次已核实的 S1 子进程，由它关闭自己的 Ray；原失败运行目录、日志与退出码移动到 `artifacts/text_s1_smoke_controller_v1/cpu_resource_failure_20260926/`，没有删除权重或其他项目。
- smoke/正式 SFT 共用 `posttraining_plan/sft_resources.py`：按上游资源池推导CPU需求，本机登记24个CPU；启动时检查集群资源，不改变样本、损失、学习率或响应长度。93项CPU回归通过。
- 已在 tmux 恢复 S1 一步更新/保存/独立恢复，正式主链随后继续等待该验证；当前不能称 S1 验证或正式训练已经通过。实时状态以控制器 JSON 为准。
- 修复后 S1 已完成实际第1次更新：两 rank Adam 0→1、抽样参数 delta约1.01e-6，loss=0.382586、grad_norm=4.434357；目前进入 checkpoint 保存阶段，独立恢复仍待完成。调度阻塞已经通过真实运行验证解除。

## 2026-09-26 10:07 UTC：S2 三步通过，S3 连续运行，正式自动接力已部署

- S2 `text_s2_source16k_profile3_v1` 两个 rank 均确认 Adam 0→1→2→3，有实际参数变化；三次权重同步完成，无 OOM。平均完整更新 602.93 秒，不含初始化和保存。
- 09:36 profile 控制器因控制台缺少最后一步指标退出，并非训练失败。原始 Ray TaskRunner 日志包含全部三步，已复制为 `artifacts/text_profile_controller_v2/S2_original_worker.log` 并记录 SHA；保持原始失败状态归档，不重复 S2。
- S3 `text_s3_source16k_profile3_v1` 正在运行，10:03 完成第1次真实更新及权重同步（366.90秒）；不能据此宣称三步/长训已经通过。
- 新日志同步写入 `optimizer_audit/step_metrics/<step>.json`，规避最后一步 Ray stdout 转发丢失；只影响观测，不改训练目标。
- S1 一步 Continue-SFT + 独立恢复由 `opd_text_sft_smoke_recover` 等待接力。尚未完成真实 GPU 验证。
- 新主接力 `opd_text_finish_v1` / `artifacts/text_finish_controller_v1/status.json` 等待上述检查，然后执行冻结协议→S0 dev→S2 100步→S3 100步→S1预先冻结目标token预算→同协议dev/test→配对统计。前两组第10步以不变100步日程保存并真实恢复，再继续。遇到失败停下，不循环调参。
- 原模型不重训教师、不复做教师优势门禁。单训练seed17、共同MixSFT起点、S2/S3相同4prompt、16384/2048响应上限；S3额外CE成本公开。
- CPU配置构建试验推得S1约493次短CE更新（2,741,059目标token），来自S2三步目标token估算，不是额外493次在线蒸馏；最终以正式冻结协议为准。
- 128条dev、901道test（1081次回答）的token预算与检查器接口在CPU验证通过，未生成test答案/未用test选模型。正式评测采用30K响应预算、Math温度0.6/IF温度1，AIME每题4次平均而非pass@4。
- 87项CPU回归通过。正式控制器、HF合并、完整评测仍须真实执行验证，不能称项目已完成。
- 已验证并清理S2/S3各一步的临时checkpoint，保留日志/恢复证据。当前约163GiB空闲；正式运行中保留最近2份，完成并导出后仅清理被替代旧点，最终完整恢复点和HF保留。不动其他项目。

以下是历史记录，后续状态以主接力状态JSON和实际产物为准。

## 2026-09-26 08:27 UTC 最新状态：修复保存后配置校验，复用S2

- S2已完成一次真实更新：两rank Adam step0→1、参数抽样delta约1.01e-6、梯度范数48.31、loss约0.3222；生成512.38秒、更新11.65秒、保存/提交490.76秒。任务性能未评测，不能据此说明能力提升。
- 完整checkpoint约36GiB已经保存。外层校验08:21:49失败，唯一配置差异为优化器`total_training_steps`外层-1、trainer运行时1；原始退出码1和失败报告保留。
- 已修复编译器提前赋有效总步数；恢复契约仅把未设置的-1解析为显式trainer步数，其他显式步数差异仍拒绝，不改变LR/数据/权重/其他契约字段。
- `tmux opd_text_source_recover_v2` 已启动，目录 `artifacts/text_source_smoke_controller_v2/`，正在全分片哈希复核。通过后创建额外`smoke_revalidated.json`证据，不覆盖旧失败报告。
- 顺序为已有S2复核→新进程S2恢复→S3更新→S3恢复，不重跑S2更新。70项CPU测试通过。恢复实际结果尚未完成。

## 2026-09-26 08:03 UTC 最新状态：已进入真实更新 smoke

- 用户明确取消“小样本教师胜率超过学生”作为开训硬门槛。采用公开专项教师，旧失败/打平报告不改，见 `posttraining_plan/ADMISSION_REVISION_20260926.md`。
- `tmux opd_text_source_smoke_v1` 正在运行，后台目录 `artifacts/text_source_smoke_controller_v1/`；已创建本任务独立Ray集群，加载学生和教师服务。目前为S2一次真实更新smoke，不是正式训练。
- 分域生成预算已接入：Math16384、IF2048；数据batch4（Math2+IF2）、学生FSDP2卡，两教师各1卡；仅授权GPU1～4。原采样token k1目标及S3专家CE方案保留，不能称为Open-MOPD top-k目标严格复现。
- 顺序：S2更新→新进程restore-only→S3更新→新进程restore-only。每个子作业最多2小时，失败停止；未自动接力正式长训或清理checkpoint。
- 67项CPU测试通过，覆盖域长度控制、原门禁默认不变、显式新策略保留旧失败结果而允许工程smoke。
- 实际GPU更新/恢复结果尚未得出；训练前后评测与正式预算待工程检查后推进，不再扩大教师胜率测试。

## 2026-09-26 07:54 UTC 最新结果：长预算诊断完成

- `source_length_diagnostic_v1` 四路exit=0，checker错误0，整体约24分钟完成，未启动训练。
- 同一8题：学生8K正确3/8、截断4/8；Math教师8K正确2/8、截断6/8；学生30K正确4/8、截断2/8；Math教师30K正确4/8、截断0/8。
- 30K教师与学生各有一道对方没做对的题，成对正确率差0；不能宣称教师更准确。教师生成93735 token，学生109046 token，单次生成耗时分别1172.04秒与1425.40秒（含引擎初始化）。这里只是小样本运行成本，不是可靠吞吐benchmark。
- 长预算改善本批完成率，数学教师优势仍未建立；原8K完整矩阵失败不因此改为通过。不能用30K评测去批准8K训练，也不能仅凭更少截断放行长训。
- 详见 `posttraining_plan/SOURCE_LENGTH_DIAGNOSTIC_RESULTS.md`；机器汇总可运行 `analyze_source_length_diagnostic.py artifacts/source_length_diagnostic_v1` 重建。
- 当前无自动接力训练；此前07:29“正在运行”为历史快照。

## 2026-09-26 07:29 UTC 最新状态

- 已启动 `tmux opd_source_length_v1`，仅GPU1～4；四个推理引擎加载成功并开始生成。不是训练。
- 固定dev最前8道数学题，不按做对与否选题；学生/数学教师 × 8192/30000回答上限，统一32768上下文、temperature=.6、top_p=.95、top_k=-1、FP16、seed17。共32次生成，不重跑384响应矩阵。
- 正式来源协议与本轮残余差异已记录：本轮仍是DAPO dev单次采样/FP16，不冒充AIME mean@64论文复现。源模型卡32K指总上下文，不是明确回答上限。
- 每个作业45分钟硬上限；失败保留日志，不自动加时、不换题、不自动批准长训。控制器结束后保存分项结果，等待核查。
- 实时证据：`artifacts/source_length_diagnostic_v1/{protocol.json,status.json,*/*.json,*/*.jsonl,*.log}`。
- 本轮60项CPU测试再次通过；测试打印的模拟更新日志不是实际GPU训练。
- 上一版8K门禁失败仍有效。优先验证作者解码与长度预算，再确定是否需要更新训练长度和数值/显存profile。

## 2026-09-26 07:25 UTC 最新状态（优先于下方历史快照）

- 完整384响应已完成，三路exit=0、checker错误0；控制器07:16:28停在 `STOPPED_AT_TEACHER_GATE`，未启动任何新训练。
- Math：学生17/64、Math教师12/64、IF教师15/64；IF：学生20/64、Math教师22/64、IF教师33/64。
- Math教师50/64响应在8192上限截断；其本域优势与跨教师专项优势均未通过。IF教师本域优势通过。原始数据/失败门禁保持不变。
- GPU1～4已释放，无OPD后台任务运行。用户要求优先修通双教师，随后要求先核查源项目；当前进行来源与配置诊断，不擅自接力长训。
- 新审计：[UPSTREAM_TEACHER_AUDIT.md](posttraining_plan/UPSTREAM_TEACHER_AUDIT.md)。确认作者Math教师训练回答上限30000、正式评测总上下文32768；本地8K回答、DAPO dev、单次采样、FP16与官方成绩协议不同。长度是重要嫌疑，尚未证明是唯一原因。

## 06:04实施快照（下方“正在运行”均为历史状态）

更新：2026-09-26 06:04 UTC。**数据及文本适配已实施，真实长前缀数值检查通过；完整教师矩阵正在运行，未开始新训练。** 旧算术项目结果不计为本方案验证。

## 当前进程

- 模型下载v4于04:08 UTC成功退出（exit=0），三份共20,026,715,304字节权重，官方SHA256/索引及tokenizer探针对齐通过。
- `artifacts/pretrained_gpu_v1/report.json`：三模型GPU 1上HF BF16真实前向、cache生成均通过；完整tokenizer backend相同，未改发布分词规则。约6.24GiB峰值仅为短推理，不是训练显存。
- 当前只有推理诊断，没有 Math/IF optimizer 更新。完整矩阵 tmux `opd_matrix_student_v1`、`opd_matrix_math_v1`、`opd_matrix_if_v1` 分别使用GPU 1/2/3；每模型 Math64+IF64，共384次响应。05:53快照为学生20/128、Math12/128、IF16/128，不是完成结果。GPU4空闲，其他项目/他人进程未动。
- 原始数据下载已完成；候选 `data/posttraining_text_v3/`：online1024、dev128、完整专家256、公开test901题。dev/test与v1字节一致，训练数据补人工去重；许可/噪声限制见 `posttraining_plan/DATA_CARD.md`。未对公开test生成响应。
- `.text_runtime` 隔离了评测ANTLR4.13依赖；Hydra训练仍用原conda/ANTLR4.9，不改共享环境。隔离环境NumPy固定2.2.6，初次NumPy2.5导致的vLLM启动失败已解决、日志保留。
- 编码、固定专家选样、独立CE分母、文本trainer分支、S2/S3单次更新规则、身份校验、checkpoint提交/恢复前校验、S1专家dataset、评测/配对汇总已实现。**60项CPU测试及7项旧链路回归通过；新四卡训练接线和完整模型实际恢复仍未通过GPU验证。** 60项包括控制器门禁失败不启动/子作业失败不接力等负例。
- 真实2进程CPU Gloo梯度测试通过：`artifacts/text_ddp_loss_cpu_v2.json`。修复了单token响应的教师概率误squeeze问题；不同rank/micro的在线和CE分母与整批参考一致。这不是FSDP GPU训练通过。
- `run_text_smoke.py` 默认只检查，执行模式只允许一次更新，校验两rank参数变化/Adam step、完整checkpoint；新增独立进程restore-only诊断，逐字节核对实际加载的模型/optimizer/scheduler/RNG以及driver数据状态，尚未真实执行。不宣称后续vLLM采样与连续训练逐位相同。
- GPU4随机小Llama的真实FSDP checkpoint保存→另一进程恢复已通过：`artifacts/tiny_fsdp_restore_v1/load_result.json`，模型/Adam/scheduler/RNG字节摘要一致。单rank会退化为NO_SHARD，因此**不能冒充3B模型两rank分片恢复通过**。诊断已退出并释放GPU4。
- 随后CPU/Gloo两rank随机小Llama的**分片**保存→另一组进程恢复也通过：`artifacts/tiny_fsdp_cpu_sharded_v1/result.json`，覆盖真实ShardedTensor和两rank状态摘要；仍不代表3B模型/NCCL训练通过。
- 接力控制器05:59:44 UTC已启动，tmux `opd_text_gate_v1`；`artifacts/text_matrix_controller_v1/status.json` 是实时阶段。新说明见 `posttraining_plan/TEXT_RUNBOOK.md`，来源/参数差异见 `posttraining_plan/CONFIG_DIFF.md`。

## 数值与质量证据

- 旧 `vllm_preflight_v1` 与部分BF16教师探针失败仍保留，不放宽max=0.5、mean=0.05阈值。
- 学生BF16/FA2长前缀检查：`artifacts/long_student_prefix_student_bf16_v1/report.json` PASS。
- 两教师FP16/SDPA长前缀检查：`artifacts/long_student_prefix_{math_teacher,instruction_following_teacher}_fp16_v1/report_sdpa.json` PASS。
- 这些检查均使用16条**真实学生生成**响应（数学最多4096 token、IF含EOS），检查逐条与padding batch；不是教师自己生成回答后假冒学生前缀。
- 新配置编译器采用教师FP16作为数值候选。此前自主回答质量探针是BF16，不能把它冒充FP16质量验证；精度差异须补验并冻结。

4096长度探针，每模型数学16题+IF16题，checker错误0：

| 模型 | Math正确 | Math截断 | IF正确 | IF截断 | 作业推理墙钟秒（含初始化） |
|---|---:|---:|---:|---:|---:|
| MixSFT学生 | 0/16 | 15/16 | 9/16 | 1/16 | 805.11 |
| Math教师 | 0/16 | 16/16 | 9/16 | 1/16 | 813.63 |
| IF教师 | 2/16 | 14/16 | 11/16 | 1/16 | 816.08 |

证据：`artifacts/teacher_dev_probe_v2_*/{protocol,predictions,summary}.*`。生成完整结束后解释器退出挂住，确认产物齐全后只终止本任务父进程/engine，退出码143保留，不称作零退出作业。

8192 BF16长度对照已正常零退出：学生Math0/16、IF10/16；Math教师Math2/16、IF7/16；IF教师Math3/16、IF12/16。数学截断分别14/16、14/16、11/16；见 `artifacts/teacher_dev_8192_v1_*/`。不能据此宣布数学教师优于另一教师。

官方本地脚本采样参数（temperature=.6、top_k=20）的小诊断也已完成，未消除数学截断；它是本地脚本默认，不是论文完整评测协议。公开模型卡的正式评测采用32768上下文、top_k=-1，IF温度1、数学温度.6，我们的8192响应、.6/20开发矩阵并非严格论文复现。保留协议差异，不中途修改正在运行的矩阵。

当前完整矩阵统一FP16、temperature=.6、top_p=.95、top_k=20、response8192。启动时间：学生05:28:10、IF05:32:05、Math05:35:29 UTC；每任务2小时超时上限，详见 `posttraining_plan/TEACHER_MATRIX_PROTOCOL.md`。前16题与诊断重合，不能相加当独立样本。显式engine shutdown已在前述8192小诊断验证正常退出。

当前阻塞：完整教师优势/互补矩阵未完成；分布式真实更新/保存/恢复未验证；显存和checkpoint峰值空间未知（当前可用约164GiB）。不自动开50步。

### 下载历史（已完成，下面不是当前运行任务）

- 下载启动时GPU 1/2/3/4 已核实空闲，恢复四卡方案；当前使用情况以上方最新进程为准。
- tmux：`opd_posttraining_assets_fast_v2`（保留原会话名），`下载续传`窗口运行v4，学生、Math教师、IF教师并行，每个权重文件最多8个分段连接。
- 日志：`artifacts/posttraining_assets_20260926_resume_v4/launcher.log`；同目录各角色 `*.download.log` 记录下载速度。
- 03:04 UTC，v2因TLS断连/EOF导致三个下载任务均退出，最终退出码2。约6GiB分段缓存保留。v3增加传输错误的进程级重试（每文件最多6轮），降低并发连接、延长超时，继续复用`.aria2`断点；校验/文件系统错误不按网络错误自动重试。此前1.5～2小时估计已受网络失败影响，需重估。
- v3在Math元数据读取遇到`tqdm`共享锁并发错误，主动停止并改为v4：串行获取小元数据，保留三个大权重任务并行，原缓存继续复用。
- 原 `opd_posttraining_assets_v1` 下载已主动停止，改用 aria2 加速，不是训练失败。保留原 HF 缓存；新权重位于 `artifacts/accelerated_assets_v1`，使用官方 SHA256 校验。原 HTTP 不完整文件不直接复用为分段文件，已完成元数据及分段测试缓存保留复用。
- 机器状态：同目录 `status.json`，退出后写 `exit_code.txt`。本轮续传有四小时上限，超时/失败不自动开训。
- 下载完成后自动检查权重文件hash/index、构建独立兼容配置overlay、检查词表/特殊token/模板探针一致性；这些通过仍不等于真实模型前向或教师门禁通过。

## 就绪边界（纠正“准备完成”的歧义）

| 项目 | 当前状态 |
|---|---|
| 主方案、逐文件交接、配置草案 | 已完成 |
| 配置兼容小模型CPU回归、契约测试 | 已完成 |
| 三份完整公开模型权重 | 已完成，SHA256/索引通过 |
| 三份真实HF GPU前向/cache和完整tokenizer backend | 已通过，非质量门禁 |
| 真实数据获取、上游许可核查、train/dev划分 | 已完成候选v3；不宣称完整转授权或无语义泄漏 |
| 单轮文本专家回放接入 | 编码/选样/trainer接线与CPU回归已实施，真实更新待验证 |
| 真实教师能力矩阵、服务端logprob | 长前缀数值通过；完整384响应开发矩阵正在运行 |
| 新架构真实更新、保存恢复、四卡profile | 未完成 |
| 正式训练预算冻结与训练 | 未开始 |

前期只完成CPU规划与兼容性工作，未完成模型和数据资产准备。双卡尝试停止时下载也停止，约66MiB缓存保留；这不应被称为全部准备就绪。模型下载不需要GPU，本应更早完成。

下一步：完整矩阵→按预登记门禁判断→通过才做S2/S3各一次更新与独立恢复→profile及预算冻结。后台控制器 `watch_matrix_and_smoke.py` 已启动，实时状态见 `artifacts/text_matrix_controller_v1/status.json`。控制器只接力有界smoke/恢复，**不会自行接力正式训练**。失败停下，不自动换题/改门禁；没有新实验自动删除动作。轮换工具已有单测，仍要求真实恢复报告，未应用到实际权重。
# 2026-09-26 08:40 UTC 最新执行状态

- S2 纯 OPD 的一次更新、完整 checkpoint 保存与独立双卡恢复全部通过：`artifacts/text_s2_source16k_smoke_v1_restore/restore_result.json` = `DISTRIBUTED_RESTORE_VERIFIED`。实际加载的 model/optimizer/scheduler/RNG 及 driver 进度/数据状态吻合；不承诺后续在线 rollout 逐位等价。
- 当前 S3（OPD＋固定专家 CE 回放）在 `opd_text_source_recover_v2` 执行一步更新/保存/恢复，未重复 S2 更新。
- 已部署 `opd_text_profile_v1`：等待现有控制器成功，再各执行 3 次连续更新、不存 checkpoint，汇总更新/生成耗时与 token；不是正式结果。失败即停，不调整教师/奖励/长度，不触碰其他进程。
- 用户授权清理临时 checkpoint。S2 `checkpoints/global_step_1` 已删除 37,969,602,016 字节（约35.4 GiB）；配置、日志、rollout、完整 manifest 副本、更新/恢复证据保留，权重不可恢复。S3 同样等独立恢复子进程成功退出后才清理。磁盘剩余约163 GiB（会动态变化）。
- 已接入正式 checkpoint 提交后的可选自动轮换：保留最近两点，保护显式结果；默认没有正式标记不删除。真实正式训练尚未开始，不能称正式轮换已实跑。CPU测试覆盖先提交后删、保留最近两点、拒绝用不同预算的 smoke 报告启用正式轮换。
- 已知配置审计待办：当前异步 vLLM 实际日志 `enable_prefix_caching=True/chunked_prefill_enabled=True`，与 YAML false 不一致（CLI 默认值问题待核实）；不要把 YAML 写值冒充实际引擎设置。两组当前路径相同，保留实际日志。
# 2026-09-26 08:57 UTC 最新状态（覆盖下方历史）

- 用户允许正式预算超过50更新；当前规划 **S2/S3各最多100更新，10更新检查，scheduler从开始按100定义**。这不是已经启动正式训练。训练/最终评测仍未完成。
- S3 两rank均完成 Adam 0→1、被观测参数变化约1.01e-6、梯度范数1.815，完整 checkpoint 已提交。随后唤醒 vLLM 权重时 OOM，旧运行如实保留 `SMOKE_FAILED`；不把它标记为完整训练成功。
- 已修复权重同步前的缓存释放顺序（先 `aggressive_empty_cache` 再 `rollout.resume(weights)`），不改变长度、batch、目标或 offload 配置。是否解决连续训练切换 OOM，仍待 profile 实证。
- 保存的 S3 分片哈希重核通过，`smoke_revalidated.json` 只证明更新/保存、明确 `post_update_weight_sync_passed=false`。新 tmux `opd_text_source_recover_v3` 正在从该点独立恢复，不重跑长 rollout。
- 新 tmux `opd_text_profile_v2` 等待 v3 恢复成功后，清理S3临时点，再从共同初始权重跑 S2/S3 各3更新（不保存权重）。旧 v2恢复控制器及 v1profile 已失败停止，日志保留。
- 同时修正纯 OPD 的归一化初始化顺序：`distillation_ppo_loss` 在读取当前全局有效token信息前就计算loss，可能沿用默认/旧分母；现在先刷新再算loss。另修复纯OPD单token响应轴被误squeeze。新增真实函数的S2/alpha=0回放损失与梯度等价测试通过。旧S2 smoke只作工程历史，不作任何收益对照。
- `run_sft_smoke.py` 新增 Continue-SFT 的真实更新/独立恢复入口，复用verl SFTTrainer和S3同一专家编码；CPU配置/小状态测试通过，真实GPU尚待执行。tmux `opd_text_sft_smoke_v1` 在两组profile成功后接力S1一步更新、完整保存、独立恢复，再清理临时权重。
- **81项CPU回归通过**（测试输出里的模拟控制器状态不是实际任务进度）。GPU实际状态以 `text_source_smoke_controller_v3/status.json`、`text_profile_controller_v2/status.json`、`text_s1_smoke_controller_v1/status.json` 为准。
- 最新接力只覆盖上述工程验证，不会冒充已实现正式训练/评测自动全流程；100更新正式入口、S1目标token预算、统一S0/训练后评测与报告仍待衔接。
