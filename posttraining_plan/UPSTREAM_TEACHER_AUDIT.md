# 源项目教师与本地失败矩阵核查

2026-09-26；只做来源/配置审计，未启动新训练或延长推理。旧矩阵完整保留。

## 1. 源项目实际链路

SmolLM3-3B-Base → 四轮混合SFT → 从同一MixSFT分别做Math/Code/IF专项GRPO → 学生重新从MixSFT开始多教师OPD。
教师与学生同规模，但训练阶段不同；MixSFT不是最终蒸馏模型。原项目按已知domain标签路由教师，不是每题挑表现最好的教师。

原始报告（不是本地复现结果）：

| 域 | MixSFT | 对应RL教师 | 差值 |
|---|---:|---:|---:|
| Math：AIME24/25均值 | 17.95 | 24.24 | +6.29个百分点 |
| IF：IFEval/IFBench均值 | 41.46 | 51.08 | +9.62个百分点 |

来源：[作者论文§2.3、表2及附录表7/8](https://arxiv.org/html/2608.19098v1)、[公开模型清单](https://github.com/BytedTsinghua-SIA/Open-MOPD)。

Math教师模型卡标注公开checkpoint为step100，而论文列1000为教师训练配置步数；IF公开checkpoint为step2440、计划3000。不能把计划步数说成下载权重实际训练步数，也不能仅据二者不同断言文件错误。
来源：[固定Math模型卡](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math/blob/5e901bb626b69d711074d2832c41ce1aa4232da8/README.md)、[IF模型卡](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-IF)。下载的revision/分片SHA此前已验证，不能将兼容overlay或模型名称当成权重损坏证据。

## 2. 最重要的协议差异

| 项目 | 作者报告/实际阶段配置 | 本地矩阵v1 |
|---|---|---|
| Math教师GRPO时回答上限 | 30,000 token | 复用公开教师，没有重训 |
| M-OPD训练回答上限 | Math/Code16K，IF2K | 尚未训练；候选统一8192 |
| 评测总上下文 | 32768 | 10241 |
| 本地评测回答上限 | 不能把32K总上下文直接等同32K回答 | 明确8192 |
| 采样top_k | 模型卡正式协议-1 | 20，来自本地演示脚本默认 |
| Math题目/采样次数 | AIME24/25，mean@64，temperature=.6 | DAPO来源dev64，每题一次，temperature=.6 |
| IF题目/温度 | IFEval/IFBench，temperature=1 | Nemotron来源dev64，temperature=.6 |
| 精度 | 公开BF16权重，HF示例BF16 | 为数值诊断统一FP16 |

作者本地脚本默认2048回答/4096上下文/top_k20，是快速示例，不等于论文表格的生成预算。
来源：[固定eval脚本](https://raw.githubusercontent.com/BytedTsinghua-SIA/Open-MOPD/4809a96cf85a869106ff0ff3f37d0a51e12010ae/scripts/local/eval.sh)、[MixSFT模型卡正式评测协议](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT#evaluation-protocol)。

另外：原论文的学生top-k16概率加权蒸馏，与本地sampled-token k1/PG不是同一实现。采样top_k20与蒸馏top-k16也不是同一个参数。此差异尚未产生本轮教师失败，因为本地还没有新训练，但后续绝不能声称严格复现Open-MOPD目标。

## 3. 本地已确认与尚未确认

证据：`artifacts/text_matrix_controller_v1/matrix_result.json`；三模型各128输出、退出码0、checker错误0。

- 学生Math17/64，Math教师12/64；IF学生20/64、IF教师33/64。
- Math截断：学生35/64、Math教师50/64、IF教师37/64。
- 教师在本地8K生成协议下的数学优势未建立；没有启动更新/恢复/正式训练。
- 抽查原始响应确有长时间停留在think内的推演/重复，但不能把每一次失败都归因于截断。
- 分布、预算、采样与精度均不同，因此本地与论文数字不可直接比较。
- “本域教师必须优于另一个教师”的交叉检查是本地预登记条件，不是论文直接给出的前置门禁。即使去掉它，Math教师对学生的本域优势仍没通过；不事后删门禁来宣布成功。

## 4. 下一步边界

优先保留双教师；先以固定dev问题、小规模、有明确超时的诊断检查作者解码和长推理预算，分别识别长度与采样影响，不把两个变量一起变好后全部归因于长度。问题选择不依赖谁做对，不读取最终测试成绩。

如果只有长预算才显示教师优势，必须再验证学生rollout/教师评分/反向的对应可承受配置，不能拿32K评测通过去批准8K训练。若长预算仍无优势，再排查实现或提交更换教师/范围的方案；当前不自动换模型、不无限扩样、不宣称双教师已跑通。

## 5. 已启动的有界诊断（07:28:58 UTC）

`run_source_length_diagnostic.py --execute` 已在 `tmux opd_source_length_v1` 启动四路；只用GPU1～4。按固定dev文件顺序选前8道数学题，不依照答案或模型得分选题。

| GPU | 模型 | 回答上限 |
|---|---|---:|
| 1 | MixSFT学生 | 8192 |
| 2 | Math教师 | 8192 |
| 3 | MixSFT学生 | 30000 |
| 4 | Math教师 | 30000 |

所有组保持总上下文32768、FP16、温度.6、top_p=.95、top_k=-1、seed17、batch4不变。只做开发诊断；保留DAPO与AIME、一次与mean@64、FP16与BF16等差异，不能称为论文成绩复现。

每组45分钟超时退出；不自动扩大样本、删除失败结果或启动训练。采样改变的影响可与旧8K矩阵中同一8题比较，但旧矩阵上下文不同，不能宣称完全单变量；新四路的8K/30K对照除长度外配置相同。

实时目录：`artifacts/source_length_diagnostic_v1/`。完成后在项目根目录运行：

```bash
python posttraining_plan/analyze_source_length_diagnostic.py artifacts/source_length_diagnostic_v1
```

汇总器要求作业零退出、匹配题ID与生成协议、checker无异常；输出成对正确率差、截断、token量和耗时，不给出自动训练许可。新增3项正负例单测通过。
