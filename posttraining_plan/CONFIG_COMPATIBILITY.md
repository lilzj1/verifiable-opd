# SmolLM3 配置兼容修复与证据

日期：2026-09-24。状态：**配置语义适配与 CPU 回归通过；真实权重、GPU 引擎与分布式训练待验**。

## 1. 纠正初次判断

10:51 的检查发现 `rope_parameters.rope_theta=5000000` 与旧版实际生效的 `rope_theta=2000000` 不同，这是一个真实歧义。但当时直接把500万当作唯一正确值、把200万判为错误，**证据不充分**。

进一步核对：

- [作者公开安装脚本](https://github.com/BytedTsinghua-SIA/Open-MOPD/blob/4809a96cf85a869106ff0ff3f37d0a51e12010ae/training/install_requirements.sh) 固定 Transformers 4.57.6。只读核查，未执行其安装脚本。
- 两教师原始配置同时保存了旧字段200万和新字段500万；学生配置只显式保存新字段500万。`transformers_version` 记录保存版本，不能独自证明训练/评测实际使用哪个参数。
- 独立安装4.57.6后，三份原始配置实际均采用200万；本机4.57.3与此一致。旧版默认 RoPE 实现读取 `config.rope_theta`。
- 独立5.3.0读取新字段，三份原始配置实际采用500万。核对 [5.3.0 SmolLM3配置实现](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/smollm3/configuration_smollm3.py)。

因此，本次不是“为省显存修改模型参数”，而是**选定公开运行环境作为明确参照，消除版本切换时的字段优先级歧义**。不声称确认了作者所有历史作业或证明200万比500万质量更好；教师在本地任务上是否有效仍须真实验证。

初次 `two_gpu_preflight_v1/report.json` 保留，不篡改原始检查。其“嵌套字段优先就是正确值”的判定已被本报告取代，审计脚本现在要求参照证据，不再仅凭字段不同宣布旧环境错误。

## 2. 采用的独立适配策略

策略 ID：`open_mopd_published_runtime_tf4576_v1`。

`smollm3_compat.py` 只接受已审计的三份原始配置 SHA256和4.57.6实测参照：

- 旧 `rope_theta` 与新 `rope_parameters` 都显式设为参照采用的200万；默认、不缩放 RoPE。
- 显式保留参照采用的 `tie_word_embeddings=true`。
- 不改变NoPE层序列、hidden/layer/head配置、上下文长度、EOS/PAD、dtype、dropout等其他字段。
- 原始配置不覆盖；单独生成配置视图和变更 manifest，任何未审计来源都拒绝自动适配。
- 需要真实模型目录时，使用 `--snapshot` 创建独立 overlay，权重和tokenizer使用只读源的符号链接，配置单独保存。不复制/修改大权重，不覆盖已有目录，拒绝缺失分片/非法分片路径。

当前仅生成了 `artifacts/config_compat_v1/normalized/*.config.json`，**这些不是完整可加载模型目录**。真实overlay须等原始权重和tokenizer完整后生成；该功能的文件系统测试使用合成占位文件，不算权重校验。

## 3. 实测矩阵

以4.57.6原始配置构造的固定CPU小模型为参照：4层，包含3个RoPE层和1个NoPE层，head dimension保留128；同一随机权重、输入、FP32、eager attention。检查旋转频率、完整前向logits、CE、embedding梯度、KV-cache续接logits。

| 被测环境/配置 | 三角色结果 | 相对参照 |
|---|---|---|
| 本机4.57.3，原始配置 | 3/3通过 | 五项最大绝对差均0 |
| 本机4.57.3，适配配置 | 3/3通过 | 五项最大绝对差均0 |
| 独立4.57.6，适配配置 | 3/3通过 | 五项最大绝对差均0 |
| 独立5.3.0，原始配置（负对照） | 3/3未通过 | logits最大差约0.001578；频率差约0.022476 |
| 独立5.3.0，适配配置 | 3/3通过 | 五项最大绝对差均0 |
| 本机vLLM 0.11.0配置加载器，适配配置 | 3/3通过 | theta、NoPE、结束token等契约通过；**未启动引擎** |

容差预设 atol/rtol=2e-5。观察到差值0只适用于这些CPU固定样本，不能外推为完整BF16/GPU训练逐位等价。小模型不具备公开权重的任务能力，也不替代logprob质量门禁。

报告位于 `artifacts/config_compat_v1/`：`reference4576`、`local4573`、`raw530`、`normalized4573`、`normalized4576`、`normalized530` 下的 `report.json`，另有 `vllm_config_report.json`。测试权重为本机生成的 `reference_fixture.pt`，不是下载的模型。

规划/schema及适配负例单测合计22项通过。检查缺少参照、来源hash改变、配置漂移、源文件不变、拒绝覆盖、缺资产等。

## 4. 可复核命令

从 OPD项目目录执行。下面输出目录均须尚不存在；重跑使用新后缀，避免覆盖旧证据。

```bash
# 纯CPU规则测试
python -m unittest discover -s posttraining_plan -p 'test_*.py' -v

# 生成配置视图（不含权重，不能当模型路径启动）
python posttraining_plan/smollm3_compat.py \
  --configs artifacts/two_gpu_preflight_v1 \
  --reference artifacts/config_compat_v1/reference4576/report.json \
  --output artifacts/config_compat_v2/normalized

# 用共享环境只读执行CPU回归，不改它的安装包
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python \
  posttraining_plan/probe_smollm3_runtime.py \
  --configs artifacts/config_compat_v2/normalized \
  --fixture artifacts/config_compat_v1/reference_fixture.pt \
  --output artifacts/config_compat_v2/local4573
```

两份参考环境是 `.compat_tf4576` 与 `.compat_tf530`，使用 system-site-packages 只读复用已有PyTorch，再在各自venv装固定Transformers。**只是核验环境，不是新的训练环境**；pip提示继承的旧verl包版本约束不满足，因此不能直接用它们启动verl训练。共享环境保持4.57.3，后续verl仍须明确使用项目固定源码入口。

## 5. 下一道门禁

配置歧义已消除，不必等待四卡解决。但真实运行前仍需：

1. 下载固定revision完整模型，核对tokenizer/chat template/结束规则和分片；overlay只解决配置，不解决数据问题。
2. 真实权重下学生/教师加载、前缀logprob与生成检查；如要比较原始路径与overlay，在同一4.57.3环境同样本比较，不拿跨版本差异当算法收益。
3. vLLM实际推理及verl/FSDP/remove-padding/保存恢复验证；本次配置加载器通过不代表这些通过。
4. 最后才是 GPU 5/6 显存与单步profile，不能声称两卡显存已验证够用。

模型任务质量、公开作者成绩复现都不在本次CPU修复结论里。若后续教师门禁不过，不能反复在最终测试集上挑theta；需要独立、预登记的dev诊断或暂停。
