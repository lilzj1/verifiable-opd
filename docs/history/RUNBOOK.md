# Verifiable-OPD 实施与复现说明

> 2026-09-22 存储状态补充：纯 OPD 与均匀回放已续训至 200，结果见
> [FORMAL_RESULTS_S200.md](FORMAL_RESULTS_S200.md)。本轮已清理四份早期 SFT
> step1/8/96/384 的模型与 training_state，保留 metadata；保留共同起点 step768。
> step128 教师只保留 LoRA adapter，当前训练教师 step256 merged_model 完整保留。
> 两组 step175 已删除，step200 的 FSDP（含 optimizer/extra_state/data.pt）与 HF 模型保留。
> 失败感知组唯一正式 step50 及三组 step50 的 HF 评测模型保留。
> 下文“三个 global_step_50”等是首轮完成时的历史记录；其中纯 OPD/均匀回放
> 的 step50 FSDP 已在此前续训成功后轮换删除，恢复应使用各自的 step200。
> 删除收据见 `../维护/cleanup_20260922_receipt.json`，不要按旧探针路径尝试续训。

更新：2026-09-21。本文只描述已落盘、可复核的实现和实验；结果见 [FORMAL_RESULTS.md](FORMAL_RESULTS.md)。

## 1. 问题、方案与结论边界

研究问题是：SFT冷启动后的学生进行On-Policy Distillation（OPD）时，少量专家CE回放能否保持合法工具调用和多步执行能力；相同专家token上限下，失败感知选择是否优于均匀选择？

固定三组对照均从同一个Qwen2.5-1.5B SFT checkpoint开始：

1. 纯OPD；
2. OPD + 均匀专家轨迹回放；
3. OPD + 失败感知专家轨迹回放。

任务是自建的可执行模运算链，不是公开Agent benchmark。第一轮只有一个训练seed、每个评测split 24题、50个训练step，能证明工程链路和给出初步现象，不能声称统计显著、通用SOTA或原创算法成立。

## 2. 当前完成状态

| 环节 | 证据状态 |
|---|---|
| 精确环境、oracle、去重split、JSONL/Parquet | 已完成；train96、dev24、ID24、OOD24 |
| CPU参考后端 | 已完成；只作mask/loss/保存恢复正确性证据，不作verl性能结果 |
| Qwen2.5-1.5B SFT冷启动 | 使用`step_000768/model`作为三组共同起点 |
| Qwen2.5-7B教师 | 原始模型自主工具调用不合格；用train专家轨迹LoRA适配到累计256步 |
| 教师门禁 | held-out dev24 teacher-forcing通过：教师NLL 0.1129、学生0.1765；教师token accuracy 0.9600、学生0.9387 |
| verl专家CE回放接入 | 已完成跨rank token归一化、均匀/失败感知选样、联合反传和日志 |
| 真实GPU profile | 三分支1-step通过；10-step异常工具输出路径通过 |
| 正式训练 | 三组seed=17、50-step全部完成并保存模型/optimizer/extra state |
| FSDP转HF | 三组均合并成功；每组338张量、3.087 GB safetensors |
| 冻结评测 | SFT起点与三组模型均跑完整dev/ID/OOD，greedy、seed=123 |

教师门禁衡量的是在专家prefix上的评分能力，不等于教师能自主完成24题；训练时教师冻结，评分对象是学生真实生成的prefix与动作，不用教师生成轨迹替换学生状态分布。

## 3. 数据与代码流

```text
data.py / environment.py / trajectory.py
  生成互不重复任务 → oracle专家轨迹 → 严格JSON工具环境

teacher_sft.py / teacher_gate.py
  只用train适配教师 → 只用dev专家动作比较师生NLL/token accuracy

agent_loop.py
  学生多轮在线rollout；成功、失败、非法tool JSON统一输出固定schema

verl_replay_batch.py / expert_agent_loop.py / replay.py
  从train专家池按token预算选择完整轨迹 → 构造独立专家batch

verl_mixed_loss.py + verl actor/trainer接线
  在线动作做OPD → 专家动作做CE → 各自按跨rank有效token数归一化 → 联合更新

FSDP checkpoint → verl.model_merger → HF模型 → evaluate.py冻结greedy评测
```

`data/protocol_v2`冻结为96/24/24/24。train/dev/ID长度为3/5/8，OOD长度为12/16；按完整任务内容哈希去重。test从未用于教师门禁、选参或提前停止。

## 4. 冻结配置

- verl fork：v0.8.0，commit `7aed6b230776f963fa09509c10d9c3a767d1102c`。
- 学生起点：`artifacts/qwen_sft_step1_probe_v1/step_000768/model`。
- 冻结教师：`artifacts/qwen_7b_teacher_lora_s256_v1/merged_model`。
- GPU：仅物理1、2、3、4；学生FSDP/rollout使用1、2，两个单卡教师副本使用3、4。
- seed=17，train batch=2，rollout n=1，temperature=1，LR=1e-6，constant scheduler，50 steps。
- replay alpha=0.1，最多2条完整专家轨迹、512个专家动作token；实际消耗另行报告。
- 评测：greedy、seed=123、`max_new_tokens=96`、`max_context=4096`，每个split全量24题。
- 主机GPU拓扑全部为SYS，无P2P/NVLink；使用`NCCL_P2P_DISABLE=1`和`NCCL_IB_DISABLE=1`避免FSDP初始化卡住。

Transformers对本地Qwen tokenizer打印Mistral-regex兼容警告。已对protocol_v2全部168条记录比较两种加载模式，token ID不一致数为0。为保持训练/评测语义一致，本轮没有在评测阶段单独切换tokenizer行为；该审计不能外推到任意文本。

## 5. 复核与正式训练

从项目目录执行：

```bash
cd /home/ubuntu/hello_agents_main/能力补全/OPD项目
PY=/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python

OMP_NUM_THREADS=1 "$PY" -m unittest discover -s . -p 'test_*.py' -v
PYTHONPATH="$PWD/upstream/verl-v0.8.0:$PWD" OMP_NUM_THREADS=1 "$PY" check_verl_adapter.py
PYTHONPATH="$PWD/upstream/verl-v0.8.0:$PWD" OMP_NUM_THREADS=1 "$PY" check_verl_replay.py
```

正式运行模板如下。三组必须分别从同一个SFT起点开始，不能串接：

```bash
for mode in opd uniform failure; do
  CUDA_VISIBLE_DEVICES=1,2,3,4 \
  MODE="$mode" TOTAL_STEPS=50 TOTAL_EPOCHS=100 SAVE_FREQ=50 \
  TEACHER_TP=1 \
  TEACHER_MODEL=artifacts/qwen_7b_teacher_lora_s256_v1/merged_model \
  EXPERIMENT_NAME="formal_${mode}_s50_seed17_repro" \
  bash run_verl_profile.sh 2>&1 | tee "artifacts/formal_${mode}_s50_seed17_repro_launcher.log"
done
```

本轮实际实验目录因保留失败审计而使用版本后缀：

- 纯OPD：`artifacts/formal_opd_s50_seed17_v2`；
- 均匀回放：`artifacts/formal_uniform_s50_seed17_v1`；
- 失败感知回放：`artifacts/formal_failure_s50_seed17_v1`。

三个`global_step_50/actor`都含两份model shard、两份optimizer shard、两份extra state、FSDP配置和HF配置，能够按原模式继续训练。由于本轮scheduler为constant且warmup为0，延长总步数不会改变LR曲线；仍需保持模型、数据、模式、batch和loss配置一致。HF合并目录只用于推理，续训应指向FSDP checkpoint。

## 6. 合并、评测与汇总

合并示例：

```bash
PYTHONPATH="$PWD/upstream/verl-v0.8.0:$PWD" "$PY" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir artifacts/formal_opd_s50_seed17_v2/checkpoints/global_step_50/actor \
  --target_dir artifacts/formal_opd_s50_seed17_v2/merged_hf

"$PY" -m verifiable_opd.model_integrity artifacts/formal_opd_s50_seed17_v2/merged_hf
```

冻结评测示例；test不得传`--limit`：

```bash
CUDA_VISIBLE_DEVICES=1 "$PY" -m verifiable_opd.evaluate \
  --model artifacts/formal_opd_s50_seed17_v2/merged_hf \
  --data data/protocol_v2/test_id.jsonl \
  --output artifacts/repro_opd_test_id --device cuda:0 --seed 123
```

重新从日志和trace生成JSON及Markdown结果：

```bash
"$PY" analyze_formal.py
```

默认输出`artifacts/formal_results_v1.json`和`FORMAL_RESULTS.md`。汇总脚本还从环境trace重算局部动作准确率；它不会覆盖原始评测summary。

## 7. 关键故障与修复

- **非法tool JSON导致DataProto concat失败**：早退轨迹缺少`environment_done`等字段；统一在agent loop结束前补齐`environment_success/environment_done/episode_success`，10-step真实profile覆盖该路径。
- **专家batch温度丢失**：专家batch显式继承训练rollout temperature，防止新旧logprob协议不一致；有回归测试。
- **无MFU字段时KeyError**：自定义混合loss并不总返回MFU；读取改为可选默认0，并做静态回归检查。
- **分布式loss缩放**：OPD和专家CE分别使用全局有效token分母，不按每卡或每micro-batch简单平均；2-rank等价测试覆盖不等token数。
- **填充副本污染目标**：执行对齐所需的中性副本只参与前向/集合通信，loss mask为0，且在advantage/回放选择之后加入。
- **GPU初始化卡住**：本机无直连P2P，关闭NCCL P2P/IB后四卡profile与三次正式训练稳定完成。

## 8. 证据与结论纪律

- 原始训练日志：`artifacts/formal_*_launcher.log`。
- 每步在线轨迹：各正式目录的`rollouts/*.jsonl`。
- 可恢复checkpoint：各正式目录的`checkpoints/global_step_50`。
- 合并模型：各正式目录的`merged_hf`。
- 评测trace与summary：`artifacts/formal_eval_v1/<model>/<split>/`。
- 教师门禁：`artifacts/qwen_7b_teacher_lora_s256_gate_dev24_v1/gate.json`。

早期一步profile的19GB checkpoint在确认仅属可丢弃探针后已删除，以释放正式实验空间；其launcher日志仍保留。三套正式checkpoint和optimizer均保留。任何简历表述必须以`FORMAL_RESULTS.md`的实际数字为准：可以写完整训练链路、分布式mask/归一化与失败诊断，不能把一次单seed现象写成显著提升。
