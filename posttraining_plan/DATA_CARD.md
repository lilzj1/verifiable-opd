# Math/IF 数据契约与审计

本地候选最终数据：`data/posttraining_text_v3/`。未训练、未做公开测试推理。

## 数据来源及用途

所有原始文件固定于 `BytedTsinghua-SIA/Open-MOPD-Data` revision
`9e897efe3257599d4300e2d5ee865a1cc714af87`，下载SHA及大小见原始目录的 `SOURCE_MANIFEST.json`。

| 文件/原上游 | 本项目用途 | 数量 |
|---|---|---:|
| rl_prompt_mix 的 math_dapo_boxed / DAPO-Math-17k | 在线数学问题，无教师答案进入模型输入 | train512、dev64 |
| rl_prompt_mix 的 nemotron_if_rl / Nemotron-RL-instruction_following | 在线可验证指令问题 | train512、dev64 |
| sft/openr1_math_93k / OpenR1-Math-220k | 固定完整数学专家回答 | 128 |
| sft/instruction_nemotron_aligned 第一分片 / Nemotron-Instruction-Following-Chat-v1 | 固定完整单轮指令/聊天专家回答 | 128 |
| AIME24/25 | 冻结测试问题；计划每题4次采样 | 30+30 |
| Google IFEval / AI2 IFBench_test | 冻结严格指令评测 | 541+300 |

IF专家池不是256条全部带硬规则的“金答案”；它是公开聊天/指令SFT示范的一个固定子集。只用第一分片是下载和计算预算决定，不能称为全数据无偏抽样。示范仍可能存在事实/格式噪声，结果需披露。

## 许可与发布边界

- [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)：上游标注 Apache-2.0。
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)：上游标注 Apache-2.0。
- [Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following)：ODC-BY，保留 NVIDIA 归属及许可信息。
- [Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1)：卡片正文区分 chat 子集 ODC-BY 与 structured outputs 的 CC-BY-4.0，不能仅凭卡片顶部标签把全库都叫 CC-BY。当前保留数据来源，不重新声明许可。
- [Google IFEval](https://huggingface.co/datasets/google/IFEval)：Apache-2.0；[AI2 IFBench 仓库](https://github.com/allenai/IFBench) 校验器 Apache-2.0。
- AIME 原题、聚合数据集及 IFBench 数据的完整转授权链并未因此得到证明。这里只做本地研究，不把原始数据、模型权重或第三方全文自动上传 GitHub；发布代码时保留下载来源/版本/署名，另审查再分发许可。

## 数据清理与拆分

1. 先扫描最终测试题用于去重，不生成测试答案、不用测试效果选配置。
2. seed17+样本身份hash决定候选顺序；每域dev64、train512。训练每4条按2数学+2指令排列，shuffle关闭。
3. NFC/空白规范化，但不抹掉IF约束的大小写、数字和标点；精确hash加区分大小写char(3,5)TF-IDF筛查≥.90的近重复。
4. v2额外复核相似度最高20对，发现同一基础任务被换约束后低于.90：埃及名字摘要、David咖啡馆故事、Ravager脚本三类。从train排除5条，并按原固定顺序补足；dev/test保持字节不变。
5. v3独立重新拟合的train↔dev/test审计：精确重叠0、≥.90重叠0、最高约.773；剩余高相似对主要共享IF约束模板，但实际基础问题不同。报告在 `SPLIT_AUDIT.json`。这是启发式去污染，不是不存在语义泄漏的证明。
6. train online和expert精确重叠0。公开模型在更早训练中是否见过dev/test仍不确定，不能称为模型从未见过的测试。

## 专家回答质量与编码

- 单轮完整assistant回答；多轮IF候选不直接截成独立示范。
- 回答>4096 token或prompt>2048排除，不截半个推理过程冒充完整示范。
- 排除上游供应商自我身份回答（如“我是 NVIDIA/Nemotron”），未改写这些答案。
- 同一公开chat template编码完整对话，确认prompt是前缀，然后提取回答；监督含公开EOS，不含EOS后的模板换行，不重复移位标签。
- 纯模型输入只有 system/user 的 role/content。checker、答案、IF约束标签放独立元数据，不拼入模型输入。
- JSONL按字面换行分割，不使用会切断字符串内Unicode行分隔符的 `splitlines()`。

## 环境隔离

评测依赖在 `.text_runtime`：math-verify/ANTLR4.13；训练使用原始verl兼容Python及ANTLR4.9。二者不能直接混装，已实测会破坏OmegaConf反序列化。NumPy固定2.2.6，避免Numba不支持2.5而使vLLM启动失败。

固定校验器：Open-MOPD提交`4809a96cf85a869106ff0ff3f37d0a51e12010ae`，IFBench提交`1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`。公共校验器只用于测试；本地dev采用来源对应的Nemotron verifier。
