# Math/IF 新链路运行手册

SFT资源检查（2026-09-26修正）：S1 smoke及正式入口均使用
`sft_resources.init_sft_ray`。上游默认2GPU需要20个CPU placement资源，
本机私有Ray登记24个CPU；不要再改为16，否则GPU空闲且任务无限等待。
该设置不改变实际训练batch、token预算或学习率。旧阻塞日志归档于
`artifacts/text_s1_smoke_controller_v1/cpu_resource_failure_20260926/`。

## 2026-09-26 10:07 UTC 自动接力入口

当前已有 tmux，**不要重复启动**：

```bash
tmux attach -t opd_text_finish_v1
```

主状态：`artifacts/text_finish_controller_v1/status.json`；主日志：`artifacts/text_finish_controller_v1_launcher.log`。该会话启动命令把日志重定向到文件，查看实时进度可在自己的终端运行：

```bash
tail -f artifacts/text_finish_controller_v1_launcher.log
```

控制器等待 S3 三步profile与S1一步保存/恢复成功，随后按顺序：

1. CPU评测接口审计，冻结 `artifacts/text_formal_protocol_v1/protocol.json`、S1/S2/S3 YAML和dev/test协议。
2. 共同起点S0的128条dev评测；这不是教师门禁。
3. S2/S3各100更新，均从共同原始学生开始。第10步保存→独立恢复审计→从10续至100；不把profile当正式训练的前3步。
4. 每10步保存，完整提交后轮换保留最近2份；运行结束且最终HF导出校验通过后，清理其非最终旧点，保留最终完整模型、optimizer、scheduler/RNG和driver状态。
5. Continue-SFT控制组S1使用预先由S2三步profile估算的目标token预算；完整专家样本，不截断答案。不宣称和OPD等FLOPs。因当前空间限制S1放在两组OPD之后，仅保存最终完整checkpoint并独立恢复；S1中途故障需要从共同起点重做，不承诺中间断点续训。
6. 各组同协议dev评测；全部模型身份确定后，对S0/S1/S2/S3完成901道test、每模型1081次回答。Math avg@4、IF严格prompt/constraint指标、题目配对bootstrap，生成 `TEXT_FORMAL_RESULTS.md`。公共测试不用来调参或挑checkpoint。

目前接力停在检查阶段，**有脚本不等于长训/评测已经完成**。主状态只有 `TRAINING_AND_FROZEN_EVALUATIONS_COMPLETE` 才表示端到端执行结束。任何子进程失败会记录 `STOPPED_NEEDS_DIAGNOSIS`；不自动换模型、缩长度、降低标准或重试。GPU只使用1/2/3/4，发现占用不杀别人的进程。

新增实现：`freeze_text_formal.py`、`run_text_formal.py`、`run_sft_formal.py`、`export_text_final.py`、`cleanup_finished_text.py`、`finish_text_project.py`、`report_text_formal.py`。正式执行前冻结代码哈希；运行后勿编辑相关Python文件，否则下一阶段会拒绝不一致的实现。

09:36的S2汇总停止不是GPU失败：三次更新已通过，恢复脚本使用了匹配run路径的原始Ray worker日志；原日志副本和SHA在profile控制器目录。后续每步指标独立写盘。下面旧章节保留作历史，不用其中过期的“只做smoke”描述覆盖此处用户新授权。

2026-09-26。本文是新方案入口，项目根目录 `RUNBOOK.md` 的模运算命令只适用于旧实验。
所有命令先进入 `能力补全/OPD项目`；不要同时启动两份控制器。

## 当前后台（2026-09-26 更新）

**08:57更新：** 当前应看 `opd_text_source_recover_v3` 和 `artifacts/text_source_smoke_controller_v3/status.json`。v2的S3更新/保存完成后，在切回vLLM时OOM，已停止。缓存释放顺序已修，现利用保存点恢复；不重跑刚才的长生成。

后续接力为 `opd_text_profile_v2`（状态 `artifacts/text_profile_controller_v2/status.json`）→ `opd_text_sft_smoke_v1`（S1一步CE更新/恢复，状态 `artifacts/text_s1_smoke_controller_v1/status.json`）。旧v1profile已停止。下面旧会话名仅保留历史。当前用户接受超过50更新，规划正式S2/S3各100、10更新检查；正式作业尚未启动，不能把smoke接力当成正式训练自动化已完成。

```bash
tmux attach -t opd_text_source_recover_v3
tail -f artifacts/text_profile_controller_v2.log
```

- 当前恢复/更新接力：tmux `opd_text_source_recover_v2`，状态 `artifacts/text_source_smoke_controller_v2/status.json`。
- S2 的后保存配置检查错误已修复；旧失败文件保留，独立 `smoke_revalidated.json` 补充核验结果。S2 实际恢复已通过。
- S3 一步更新/保存/恢复通过后，tmux `opd_text_profile_v1` 自动接力：清理已验证的临时 smoke checkpoint，然后 S2/S3 各 3 次连续更新。
- profile 状态：`artifacts/text_profile_controller_v1/status.json`；总日志：`artifacts/text_profile_controller_v1.log`；分组日志：该目录的 `S2.log`/`S3.log`；成本汇总：`cost_report.json`。
- 新协议使用 `--teacher-policy published_source --response 16384`，训练实际域上限为 Math 16384、IF 2048；不再因小 dev 教师未领先而阻止工程验证。
- profile **不保存权重**，从共同 MixSFT 起点开始，不是正式训练的前 3 步；不评判方法收益。
- 两个有界控制器不自动启动 50 步；S1 验证及正式训练/评测预算仍需落实。

```bash
tmux attach -t opd_text_source_recover_v2
# 或看接力进度（不要重复启动）：
tail -f artifacts/text_profile_controller_v1.log
```

## 历史后台（已完成或停止，不再重启）

- 教师矩阵三会话：`opd_matrix_student_v1` / `opd_matrix_math_v1` / `opd_matrix_if_v1`。
- 接力会话：`opd_text_gate_v1`。
- 状态：`artifacts/text_matrix_controller_v1/status.json`。
- 控制器日志：`artifacts/text_matrix_controller_v1.log`。
- 完整矩阵落盘后：`artifacts/text_matrix_controller_v1/matrix_result.json`。

```bash
tail -f artifacts/text_matrix_controller_v1.log
tmux attach -t opd_text_gate_v1
```

这只是服务器上的有界脚本，**不是后台AI**：不会自动修代码、调整门禁或选择新算法。断开客户端不影响tmux；服务器重启、进程被杀或磁盘故障仍会中断。

接力范围：等已有矩阵→重算教师门禁→S2一次更新→S2独立恢复→S3一次更新→S3独立恢复。每个子作业最多2小时，等待矩阵最多3小时。任何失败立即停止；完成后仍需profile、S1真实验证和正式预算冻结，**不自动启动50步**。

控制器不会删除任何checkpoint，也不会清理其他项目/占用其他GPU。仅使用物理1/2/3/4；启动前发现其中任一卡有进程则短暂等待后停止，不杀占用者。该检查不是集群调度器的排他资源租约。

## CPU与配置验证

```bash
PYTHONPATH=upstream/verl-v0.8.0:. /home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python -m unittest discover -s posttraining_plan -p 'test_*.py'
PYTHONPATH=upstream/verl-v0.8.0:. /home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python check_verl_replay.py
/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python posttraining_plan/run_text_smoke.py --group S2 --run-id text_readiness_only
```

最后一条没有 `--execute`，不会申请GPU或训练。已有控制器时不要手工执行更新命令。
`.text_runtime`用于判分/推理探针；训练使用上面原conda解释器，避免ANTLR版本冲突。不升级共享conda。

## 一次更新通过的定义

每组新目录：`artifacts/text_s2_smoke_matrix_v1` / `text_s3_smoke_matrix_v1`：

1. `preflight.json`：矩阵判定、前缀数值证据、GPU UUID、剩余磁盘；`resolved.yaml`是真正传入配置。
2. `optimizer_audit/rank_0.jsonl`、`rank_1.jsonl`：每rank恰好一次Adam step，有限非零梯度及被观测参数实际变化。
3. `checkpoints/global_step_1/TEXT_COMMIT.json`：两rank模型/optimizer/extra_state与data.pt完整，CRC及SHA核验。
4. `smoke_result.json`：只有 `ONE_UPDATE_AND_SAVE_VERIFIED_RESTORE_PENDING` 才允许恢复诊断；它本身不证明恢复。

8192回答长度有开发质量矩阵，但既有数值前缀检查最多4096；8192实际更新仍是待验证的诊断，不把短前缀通过外推为长训练已通过。

## 恢复通过的定义

新进程、新目录 `artifacts/text_s2_smoke_matrix_v1_restore`（S3同理），从原step1恢复：

- 原训练预算仍为1，不为了读取checkpoint把它改为2/50。
- `OPD_TEXT_RESTORE_ONLY=1` 在恢复后、rollout权重同步与训练循环前退出。
- 各rank对**实际加载后的**model/optimizer/scheduler/RNG与保存值计算完整本地字节摘要，不只看文件存在。
- driver恢复global_step和dataloader状态；诊断中若调用optimizer.step则拒绝通过。
- `restore_result.json` 必须为 `DISTRIBUTED_RESTORE_VERIFIED`。

它证明上述训练状态恢复，不证明外部vLLM生成器随机状态完整延续，也不承诺后续在线采样/最终权重与不中断训练逐位相同。当前smoke在epoch中途；epoch边界恢复另需验证，不能直接外推。

## 下一阶段仍需完成

- 新S1纯CE真实更新/恢复；S2/S3 3～5步profile及各阶段耗时、实际token/GPU小时、显存/磁盘峰值。
- 按profile冻结训练与评测预算，确定S1目标token预算。S2/S3正式训练从同一干净MixSFT起点重启，不续用调试权重。
- 完成冻结dev/最终测试、逐题配对报告、成本与失败分析；不能用旧算术成绩替代。
- checkpoint轮换工具虽有测试，实际删除还要求真实恢复证据、精确run marker和显式计划校验；当前控制器没有删除步骤。

## 磁盘与轮换（覆盖上方历史“不删除”描述）

用户于本轮授权清理已完成验证的临时 checkpoint，并要求正式轮换。

`cleanup_text_smoke.py` 只接受两个固定 source16k smoke 目录。必须先看到独立恢复子进程 exit=0、完整恢复报告与原 manifest 一致；保留 manifest 副本、审计、日志及配置再删除 `checkpoints/global_step_1`，移除过时 latest 指针，记录 `checkpoint_cleanup.json`。删除后不能再从该临时权重恢复，历史恢复证据仍可核查。S2 已清理，S3 由接力在完成恢复后清理。

正式训练使用 `enable_automatic_rotation` + `rotate_after_commit`：先以不变的正式总步数保存并真实恢复早期 checkpoint，再启用该 run 的 `TEXT_RUN.json`。新完整 checkpoint SHA/CRC、数据状态及 latest 指针提交后才轮换，保留最近两个＋明确保护的步骤。`max_actor_ckpt_to_keep` 继续为 null；不能启用上游可能先删除旧点的清理逻辑。

当前约 35.4 GiB/完整状态，轮换写新点时的峰值至少 3 份约 106 GiB，另计其他组最终权重；保存频率不会降低每份大小。空间不足必须在写盘前停止，不能为赶进度删唯一恢复点。共享师生、旧模运算结果和其他项目不在这次清理范围。
