# Verifiable-OPD · 双教师在线蒸馏与 SFT 回放

面向小模型的**数学推理与复杂指令遵循能力整合**，基于 verl 构建学生在线采样、按任务域路由教师、同前缀 token 评分、OPD＋专家 CE 联合优化、分布式恢复与冻结评测的后训练实验系统。

这是已经完成真实 GPU 训练与评测的工程实验，不是只提供 CPU demo，也不将已有 OPD/SFT 回放方法宣称为原创算法。

## 最终结果

共同起点为 Open-MOPD 的 SmolLM3-3B-MixSFT，教师为同系列 Math / IF 专项模型。S2/S3 各100更新，S1为493更新的 Continue-SFT 对照，训练 seed=17。

| 最终测试 | S0 共同起点 | S1 继续 SFT | S2 纯 OPD | S3 OPD＋SFT 回放 |
|---|---:|---:|---:|---:|
| AIME24 | 15.83% | 15.83% | 22.50% | **25.00%** |
| AIME25 | 17.50% | 15.83% | 22.50% | **23.33%** |
| IFEval 严格整题通过率 | 68.39% | 66.17% | 70.79% | **72.83%** |
| IFBench 严格整题通过率 | 16.33% | 18.33% | 21.67% | **25.67%** |

每模型901题、1081条回答；数学为每题4次采样平均正确率，**不是 pass@4，也不是原论文 avg@64**。S3 相对 S0 数学宏平均 +7.50、指令宏平均 +6.88 个百分点。

必须保留的边界：dev 上 S2 优于 S3；test 上 S3 得分最高。S3−S2 数学差异的95%题目配对区间跨零，指令差异为 +3.02 个百分点〔+0.78,+5.18〕。单训练 seed 的题目 bootstrap 不能证明跨训练种子的稳定收益。S3 有额外专家前向/反向开销，同更新数不等于同 FLOPs。

完整结论见 [实验报告](TEXT_FORMAL_RESULTS.md)，全部汇总见 [results](results/)。

## 自己实现的部分

- **能力整合链路**：按 Math/IF 域选择固定教师，教师在学生实际生成前缀上评分；公开 MixSFT 是初始化，不把教师生成离线答案再训练称为 OPD。
- **联合训练**：将独立专家样本 CE 回放接入真实 verl 分布式更新，分别处理在线样本与专家 token mask、有效 token 分母和跨 rank 归一化，固定 alpha=0.1。
- **配置兼容与显存治理**：审计 SmolLM3 的 RoPE/NoPE、tokenizer 和模型配置解释差异；隔离训练/评测依赖；诊断 rollout 引擎唤醒的显存生命周期。
- **可信执行**：逐 rank 参数更新审计、完整 checkpoint 提交/校验、优化器/scheduler/RNG 与 driver 恢复验证；新增保存完成后才轮换，后续策略只保留一个最新恢复点。
- **评测闭环**：去重划分、固定测试 manifest、四分片并行评测、严格指令检查器、截断/Token/耗时记录、题目配对 bootstrap；失败不自动降标准或选择性重跑。

目标函数采用 sampled-token k1 PG OPD 加专家 CE，**不是 Open-MOPD 原始 top-k 全分布蒸馏的严格复现**。旧模运算环境与参考后端保留为历史/教学，不能冒充当前 Math/IF 成绩。

## 代码导航

| 目录 / 文件 | 职责 |
|---|---|
| `verifiable_opd/posttraining/` | 编码、分域采样、回放、评测、保存恢复和轮换 |
| `verifiable_opd/verl_mixed_loss.py` | verl 蒸馏与专家 CE 联合优化接入 |
| `posttraining_plan/compile_text_config.py` | 真实生效训练配置编译 |
| `posttraining_plan/run_text_smoke.py` | 小规模更新、恢复与三步 profile |
| `posttraining_plan/run_text_formal*.py` | 正式训练与保存策略版本 |
| `posttraining_plan/eval_text*.py` | 单副本/四副本评测及换卡适配 |
| `posttraining_plan/summarize_text.py` | 协议核对、指标和配对统计 |
| `posttraining_plan/test_*.py` | CPU 回归测试 |
| `results/frozen_protocol/` | 原实验冻结 YAML、评测协议与哈希 |
| `docs/REPRODUCE.md` | 环境、数据与复现流程及迁移限制 |

## 开始使用

```bash
git clone https://github.com/lilzj1/verifiable-opd.git
cd verifiable-opd
# 不加载模型、不训练，核对公开汇总并打印四组结果
python scripts/check_results.py
# 下载固定提交的外部源码，不包含模型/数据下载；先查看第三方许可
python scripts/fetch_upstreams.py
python scripts/prepare_cpu_fixtures.py
# 在具有训练依赖的隔离环境中执行 CPU 回归
CUDA_VISIBLE_DEVICES='' PYTHONPATH=upstream/verl-v0.8.0:. \
  python -m unittest discover -s posttraining_plan -p 'test_*.py' -v
```

训练前请完整阅读 [复现说明](docs/REPRODUCE.md)。**历史控制器包含原机器解释器、物理 GPU 与证据路径，不能在新机器直接运行并声称一键复现。** 本仓库保留原始科研执行逻辑；迁移需要配置路径、重新产生 smoke/restore 证据并冻结新协议。不要把本仓库的历史结果复制成新运行的准入凭证。

## 发布范围与来源

包含项目源码、单元测试、配置、历史设计文档和聚合结果；不包含模型权重、checkpoint、原始数据、专家回答、逐题长回答、密钥、缓存或虚拟环境。

上游源码通过固定提交获取，不在此重新授权。模型/数据的 repo 与 revision 见 `posttraining_candidate_assets.json`（其中旧状态为调研时快照，完成状态以当前结果为准）。数据集许可必须逐源核查，模型许可证不等于数据集许可证。

见 [THIRD_PARTY.md](THIRD_PARTY.md)、[发布验证记录](docs/PUBLICATION_VALIDATION.md)。原机器路径仅作为执行溯源记录，不是访问凭据。源码快照哈希见 `SOURCE_MANIFEST.json`。
