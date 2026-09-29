# CPU 准备状态

日期：2026-09-24。状态：CPU_PREP_COMPLETE / WAIT_FOR_HGPO。

最新尝试：GPU 5/6的现有多教师资源布局校验未通过，已按“不强行”停止，详见 [TWO_GPU_ATTEMPT.md](TWO_GPU_ATTEMPT.md)。没有进行真实GPU加载或显存实测。学生权重下载未完成并已停止，保留可续传缓存；GPU进程状态未改变。

## 最新：配置歧义已处理，GPU验证仍待进行

详见 [CONFIG_COMPATIBILITY.md](CONFIG_COMPATIBILITY.md)。需要纠正下方初判：旧版200万并不等于已证明加载错误，作者公开安装脚本指定4.57.6也实际使用200万。现在以此为参照生成独立适配配置，新旧字段一致，原始配置/模型不覆盖。三版本CPU固定小模型回归及本地vLLM配置加载器通过，22项单测通过。真实权重、tokenizer、GPU引擎、显存与分布式更新仍未验证。

新建了两个仅供CPU对照的venv，未修改共享环境；它们不是已兼容verl的训练环境。下方预检报告保留为历史证据，其初始“嵌套字段应优先”结论已更正。

## 历史：双卡预检发现歧义（2026-09-24 10:51 UTC）

用户允许先尝试 GPU 5、6，不以损害训练效果换取运行。GPU 5 的旧 EvidenceDesk 服务此前已停止；GPU 6 还有两个既有小进程，未触碰。磁盘空闲约216GiB，三份候选权重尚未下载。

本次下载了固定 revision 的三份小型 config.json，用现有 Transformers 4.57.3 实际解析，得到：三者声明 `rope_parameters.rope_theta=5000000`，本地 `SmolLM3Config.rope_theta` 却均为默认 `2000000`。本地 `modeling_rope_utils._compute_default_rope_parameters` 读取后者，因此未经适配直接运行会改变位置编码。学生配置记录保存版本5.3.0，两教师配置记录4.57.6；**不能只检查版本号或注册表**。

证据：[预检报告](../artifacts/two_gpu_preflight_v1/report.json)，同目录保留三份原始配置和其哈希。

可复核入口：`posttraining_plan/audit_candidate_compatibility.py`。执行须使用 `CUDA_VISIBLE_DEVICES=''` 和独立、尚不存在的输出目录。预期风险退出码2，不是脚本崩溃。

状态为 `BLOCKED_UNCHANGED_RUNTIME`：**尚未做GPU显存/速度实测，不能据此判定两卡不够；四卡也不能解决这个软件兼容问题。** 没有升级共享环境、修改模型配置、下载大权重或启动训练。

下一步先在隔离适配路径核对旧新配置语义（包括RoPE、NoPE层、停止token、tokenizer和chat template），通过原始参数一致性及前向/logprob验证后，再做双卡profile。不要简单改掉一个字段便宣布全部兼容，也不要以减短响应、量化学生或改LoRA偷偷绕过正式训练目标。

## 本轮完成

- 读取本地 replay、trainer、单轮 loop、多教师配置及 checkpoint 轮换实现，形成逐文件迁移表。
- 补齐数据契约、更新口径、教师路由、mask/全局分母、评测与轮换验收细则。
- 添加非启动配置草案，默认 `allow_training=false`，15 项正式前待冻结字段明确保留。
- 添加纯标准库规划检查和合成测试，不依赖 torch、模型、网络或 GPU。

## 实测

在 OPD项目目录执行：

```text
python -m unittest discover -s posttraining_plan -p 'test_*.py' -v
Ran 15 tests ... OK

python posttraining_plan/check_contracts.py
退出码 0；CPU_DRAFT_CHECK_ONLY_NOT_TRAINING_READY
候选评测：每检查点 1081 次响应，四组共 4324 次。
```

覆盖草案内部一致性、资源池数量、上下文上限、S2无CE、输入字段白名单、未知域拒绝、跨split簇泄漏、重复ID、专家记录完整性、评测次数。

限制：簇ID由外部去重步骤提供，测试不证明真实数据已去重；输入白名单不能识别藏在用户正文里的答案。尚未下载并审计真实新数据，也未验证新模型/教师/logprob/训练更新/分布式恢复。

## 已发现，待实施时解决

1. 旧 ReplayPool 与专家 loop 绑定算术工具任务。
2. 默认 `teacher_model` 项在多教师模式被删除，需要两个非默认条目。
3. 旧 replay 配置拒绝 alpha=0，需要纯OPD快路径和独立等价性测试。
4. 旧 trainer 仅 replay 分支强制整批一次更新，S2/S3必须统一实际更新口径。
5. 旧 checkpoint 检查可能漏过“三类分片都缺同一rank”，需加强后才能启用新实验自动轮换。

## 下一步

按 IMPLEMENTATION_HANDOFF 的 P0～P3先实现资产审计、真实数据适配和文本专家编码；GPU任务须等待HGPO释放资源并核实归属。真实教师与更新smoke/profile通过后再冻结正式预算。本轮未启动任何后台等待器或自动训练接力。

本轮未修改旧训练器和共享环境，未下载模型、占卡、停止任务或删除权重。
