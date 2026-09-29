# 当前配置与公开配方的区别

核查日期2026-09-26。当前是资源约束下的Math/IF能力整合工程实验，不是严格论文复现，也未完成新训练。

## 来源层次

- 固定作者仓库：`upstream/open-mopd-reference`，commit `4809a96cf85a869106ff0ff3f37d0a51e12010ae`。
- 作者[MixSFT模型卡](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT)：共同初始化与报告的正式评测协议。
- 作者本地示例 `scripts/local/eval.sh`：可运行示例默认，不等于报告分数时的完整协议。
- 本地真正执行的配置：每个artifact里的 `protocol.json` / `resolved.yaml`，不靠本文替代。

## 核心区别

| 项 | 公开配方/报告 | 本项目当前选择 | 边界 |
|---|---|---|---|
| 学生起点 | 混合SFT后进行专项RL及M-OPD | 公开MixSFT，不是公开Final学生 | 原始SFT不是我们的贡献 |
| 能力域 | Math/Code/IF三域 | Math/IF两域 | 不宣称代码能力整合 |
| 教师 | 同一SFT起点的专项RL模型 | 公开Math、IF两位冻结教师 | 不保证在本地预算下自动更优 |
| 正式评测上下文 | 模型卡max_model_len=32768 | 开发矩阵max_response=8192、max_model_len=10241 | 可能显著增加推理截断，分数不可直比 |
| 报告采样 | 数学temperature=.6、avg@64；IF temperature=1、n=1；top_k=-1 | 开发矩阵所有域.6、top_k=20、n=1 | 取本地示例采样作诊断，不是原论文复现 |
| 正式候选评测量 | 公开报告更多数学采样 | 候选AIME每题4次，IF每题1次 | 尚需冻结，数学按mean而非pass@4 |
| dtype | 发布权重BF16 | 开发矩阵统一FP16；教师评分FP16；学生训练候选BF16 | 因服务端/参考数值检查结果选择，需记录差异 |
| 在线目标 | 多教师OPD研究 | 域路由sampled k1/PG，无任务奖励；n=1 | 不是GRPO/full-vocab KL，不把零task reward当性能 |
| 自己的对照 | 上游包含其路由改进 | S0起点、S1继续SFT、S2普通路由OPD、S3同S2加固定专家CE | 只检验回放，不复现上游路由算法 |
| S3权重 | 不是照抄作者默认 | CE alpha=.1，独立全局token分母 | 额外监督/计算单列，不宣称免费增益 |
| 训练资源 | 不沿用作者集群预算 | 4×A6000，学生FSDP使用2卡、两教师各1卡 | 显存和性能仍以真实profile为准 |

## 目前编译出的真实更新候选

- 每批4个在线问题（Math2+IF2），rollout n=1、temperature=1、top_p=.95、top_k=-1；一次optimizer更新/批。
- prompt上限2048、response8192，micro每卡1、epoch1、LR1e-6、seed17；无任务奖励、无额外reference KL。
- S3独立专家池256条，每次每域一条完整答案，各不超过4096 token，合计上限8192；不裁切专家示范。
- 学生remove-padding关闭，gradient checkpointing开启，torch compile关闭；vLLM eager、无prefix cache、无chunked prefill。
- 初次smoke总更新数1、保存完整状态；诊断输出不能当正式50步结果。
- 动态GPU使用先查进程，磁盘至少60GiB余量才开新smoke；不做自动清理。

## 保留的限制

完整矩阵若失败，不事后缩小样本、更换容易题或改判分器凑通过。短上下文对推理模型的限制与教师专项优势是不同问题，报告时应区分。即使矩阵点差为正，仍可能不显著；完整长训不是由单个开发分数自动批准。
