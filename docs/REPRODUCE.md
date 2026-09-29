# 环境与复现说明

## 0. 复现范围

这是完整项目源码与已完成实验的公开快照，不是包含模型/数据/运行目录的服务器镜像。首页结果是真实已完成实验；本次发布只重新运行 CPU 测试，不重新训练。

历史控制器有机器相关解释器、物理卡号、资产路径与严格证据校验。新机器必须迁移这些配置、重新 smoke/profile/冻结；不能仅复制原协议并删除校验来直接进入正式训练。`docs/history/` 和 `posttraining_plan/TEXT_RUNBOOK.md` 是历史记录，其中“正在运行/尚未完成”的文字不代表目前状态。

## 1. 不需要 GPU 的结果查看

```bash
python scripts/check_results.py
```

仅用标准库检查八份 dev/test 汇总及配对报告一致性。由于原始题目/回答未再分发，它不是从原始答案重新评分，更不是新一轮评测。

## 2. 固定上游、准备测试

先阅读 `THIRD_PARTY.md`，然后：

```bash
python scripts/fetch_upstreams.py
python scripts/prepare_cpu_fixtures.py
```

源码获取不会安装依赖或启动 GPU。如果网络失败留下空仓库，检查目标 origin 后对该目录重新 fetch 固定提交、checkout；脚本不会覆盖一个已有的不同版本。无需复制其他项目或原机器的虚拟环境。

CPU 测试需要 PyTorch、NumPy、Hydra/OmegaConf，以及部分模块的 Ray/verl 依赖。完整训练环境中运行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=upstream/verl-v0.8.0:. \
  python -m unittest discover -s posttraining_plan -p 'test_*.py' -v
```

`prepare_cpu_fixtures.py` 只复制已发布的小随机模型配置兼容性参考 JSON，不产生真实模型前向、更新或恢复通过的伪证据。

## 3. 环境隔离与实际版本

原实验 Linux / Python 3.12 / RTX A6000 48GB，核心训练环境实测版本：

| 组件 | 版本 |
|---|---|
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.57.3 |
| vLLM | 0.11.0 |
| Ray | 2.48.0 |
| Hydra / OmegaConf | 1.3.6 / 2.3.1 |
| flash-attn | 2.8.3.post1 |
| tensordict / torchdata | 0.10.0 / 0.11.0 |
| NumPy / datasets | 2.2.6 / 5.0.1 |
| huggingface-hub | 0.36.2 |

这是已运行环境的版本记录，不保证任意平台从 PyPI 一条命令就能求解该环境。先根据 GPU 驱动与 CUDA 安装匹配的 torch/vLLM/flash-attn，再安装固定上游及其依赖，并用 `pip check`、CPU 回归与 GPU smoke 实测。不要升级其他正在运行项目的共享环境。

评测使用独立环境，额外依赖见 `posttraining_plan/requirements-eval.txt`。math-verify 使用 ANTLR 4.13，而训练 OmegaConf 依赖 ANTLR 4.9，不要装到同一环境强行覆盖。原配置兼容参考解释器为 Transformers 4.57.6，并非实际训练解释器；具体归一化策略见 `posttraining_plan/CONFIG_COMPATIBILITY.md`。

## 4. 模型与数据

`posttraining_candidate_assets.json` 提供学生、两位教师和数据的精确 repo_id/revision。主线不是旧 Qwen 模运算实验，而是三份 SmolLM3-3B 模型。

```bash
# 使用训练环境；会下载三份模型，先检查磁盘和外网
CUDA_VISIBLE_DEVICES='' python posttraining_plan/stage_public_models.py \
  --run-dir artifacts/posttraining_assets_20260926_resume_v4
# 下载指定公开数据；需要 aria2c，并自行核实各数据源许可
python posttraining_plan/stage_text_data.py
python posttraining_plan/build_text_data.py
```

前一步模型下载还要求已准备兼容性参考报告（步骤2）。脚本会核对文件、tensor 与 tokenizer，并生成只读权重链接和兼容 config overlay；文件通过不等于 GPU 前向验证通过。历史日期目录名是当前脚本契约，不代表新运行产生于历史日期。

训练两域各512个任务槽位，专家池两域各128条，dev 两域各64题。最终 test 为 AIME24/25 各30题、IFEval541题、IFBench300题。构建脚本做去重和近重复排除，专家答案仅来自 train。固定 test 不用于调参。

不要上传原始数据/模型到本仓库。公开数据可能存在预训练污染，不能宣称测试题从未被基础模型见过。

## 5. 配置与训练路径

1. 阅读 `compile_text_config.py`、`compile_sft_config.py`，以及 `results/frozen_protocol/S1.yaml`、`S2.yaml`、`S3.yaml`。
2. 在自己的实验工作副本调整解释器常量和物理 GPU 检查；替换 YAML 中模型/数据/输出的绝对路径。原快照、历史协议保持不变，另生成新协议。
3. 先进行模型/前缀数值验证，再使用 `run_text_smoke.py --teacher-policy published_source --response 16384` 分别验证 S2/S3 更新和独立恢复；具体参数先看 `--help`。该策略不把旧教师诊断失败改写成通过。
4. 完成三步 profile、S1 更新/恢复；由 `freeze_text_formal.py` 根据本机真实证据冻结预算。需要的证据由脚本检查，不能伪造或直接使用仓库 results 代替。
5. 正式入口为 `run_text_formal.py` / `run_text_formal_v2.py` 与 `run_sft_formal.py`；全流程编排在 `finish_text_project*.py`。迁移完成前不要直接启动历史后台控制器。

原训练使用4张卡：两卡学生 FSDP/rollout、两卡分域教师。训练 batch=4，Math 回答上限16384、IF2048，prompt2048，学习率1e-6 constant，S3回放alpha=0.1。模型上下文预算不是每条实际生成长度。

S2/S3 同一初始模型、100更新总日程、第10步暂停后独立恢复，续至100；不宣称生成器 RNG 与完全不中断训练逐位一致。S1按预先估算的目标 token 预算训练493更新，不宣称同 FLOPs。

## 6. Checkpoint 和磁盘

完整恢复点约35.4GiB，另需原模型、HF导出、数据和新点写入空间。先测算峰值，不能仅按“最终留一份”预留空间。

`checkpoint_protocol.py` 完整提交/哈希校验后才更新指针；`rotation_v2.py` 保留最新一个以及显式保护点，不在新点完成前删除唯一恢复点。受信本地文件元数据缓存减少重复扫描，恢复仍做完整校验，不等于能检测所有静默位翻转。详见 `posttraining_plan/CHECKPOINT_ROTATION_V2.md`。

清理脚本默认针对历史具体目录，**不是通用删除工具**。新机器重新确认精确目标，不能对 workspace 或共享模型目录递归清理。

## 7. 评测

最终 checkpoint 身份固定后才运行 test。评测 `max_response_tokens=30000`、`max_model_len=32768`；Math temperature=0.6、IF=1.0、top_p=0.95、seed17。不同域使用严格任务检查器，不用模型自评来替代正确率。

`eval_text.py` 支持单副本；`eval_text_parallel.py` 使用物理1234，`*_1345.py` 是本次后续评测换卡的资源适配。它们维持相同数据分片和评分协议。新机器需显式选择拥有授权的卡，不终止他人进程。

`summarize_text.py` 从原始回答重新汇总并执行配对 bootstrap；`report_text_formal.py` 输出最终报告。检查器异常与协议不一致会停止，不把检查器失败算为模型答错。

仓库 `results/` 是历史聚合证据，不提供本轮最终权重；不能仅 clone 后直接对 S1/S2/S3 推理。需要从本地保留的 checkpoint 导出，或按新协议自行训练。
