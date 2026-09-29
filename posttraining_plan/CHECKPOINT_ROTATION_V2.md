# Checkpoint 轮换优化：后台安全切换已部署，实际阶段看状态文件

2026-09-27。用户要求优化后续保存，不影响正在运行的训练。

## 当前部署状态

- 新模块：`verifiable_opd/posttraining/rotation_v2.py`。
- 测试：`posttraining_plan/test_rotation_v2.py`，只操作临时合成 checkpoint。
- **未替换当前公共模块，未修改冻结协议，未重启 S2 训练或评测。**
- 用户已授权监控与自动接入，tmux `opd_rotation_handoff_v2` 运行 `watch_rotation_handoff.py`。进度见 `artifacts/rotation_handoff_status.json`；实际换班前，S3 不算已启动或已实测保存。
- 后台在 S2 dev 子进程启动后暂停旧的等待控制器（仅该 PID），不暂停评测子进程。评测完成后从 Linux `/proc` 读取真实子进程退出状态，确认退出 0、结果完整、评测配置和模型未变，再终止旧控制器、记录接力凭据并启动新版控制器。
- 新版控制器复用同一阶段台账，不重跑 S2；只将 S3 的启动命令替换为 `run_text_formal_v2.py`。S1 及所有评测命令不变。任何既有 S3 启动记录都使切换拒绝执行。
- 版本差异记录在 `artifacts/text_rotation_v2_amendment.json`，原冻结协议和源码哈希不变。失败不调参、不重复启动；安全接力失败时恢复旧等待控制器并留下诊断状态。
- 实施时已核对现有协议的全部 `source_sha256`，没有不匹配项。

## 优化边界

旧版生成删除计划与执行删除前分别对全部 checkpoint 重新进行字节哈希校验。
新版在轮换锁内执行一次计划，缓存完整校验后的文件快照；以后遇到同一份未变更 checkpoint，只检查文件集、inode、设备号、大小、权限、纳秒级 mtime/ctime 和清单哈希，不重复读取全部大文件。

新 checkpoint 第一次参与轮换仍进行一次完整 SHA256 校验；原 commit 的 CRC 与 SHA 校验不变。因此不是“所有保存只读一次”，而是减少历史文件和 plan/apply 的重复扫描。实际节省时间尚未在 GPU 正式作业测量。

按用户追加要求，新部署明确设置 `automatic_keep=1`，正常情况下只保留最新一份完整恢复点（包括模型、优化器、scheduler/RNG 等状态）。保持每 10 步保存；新权重先保存校验再删除旧权重，保存期间临时同时有新旧两份，不能先删后存。相比 keep=2，稳定占用及保存峰值均减少约一份 checkpoint（当前约 35.4GiB）。仅在有明确 `protected_steps` 时额外保留保护项，不能静默突破保护。没有上一版本兜底是单份保留的代价。

已有 keep=2 标记含义不变，当前运行目录不自动迁移。真实 restore 始终使用原 `verify_before_restore` 全量哈希；不以缓存证明可恢复性。

缓存损坏或缺失退回完整扫描，权重修改/替换导致重新全量校验，缺失分片、软链接、过期 latest 指针等均拒绝删除。

限制：只适用于受信任本地单写者目录，所有合法写入必须遵守检查点完成后不再修改及轮换锁约定。元数据快照不能检测不改变元数据的静默磁盘位翻转，也不是防恶意篡改机制。恢复时全量校验仍必要。

## 安全启用步骤（不能对当前进程热替换）

以下步骤已实现为独立控制器/启动器，不需要人工抢时间替换源文件。S3 内 Ray worker 通过显式 `worker_process_setup_hook` 安装仅本进程的 v2 入口；真实 restore 使用原全量校验。各阶段保留 `io_amendment.json`、`resolved_with_io_hook.yaml` 和 `rotation_hook/*.json` 证明入口选择。

验证：轮换/接力相关 19 项 CPU 测试通过；真实 CPU-only Ray worker 返回 `verifiable_opd.posttraining.rotation_v2`；S3 启动器预检返回 `VALIDATED_NOT_EXECUTED`。这不是 S3 GPU 训练已完成的证据。S1 保持原 final-only 保存。

1. 等当前训练子进程完成；检查控制器，不能在下一子进程已经启动后修改公共模块。
2. 在明确作业边界使用新版本控制器/启动器；保留原冻结协议及结果，记录独立的保存机制修订清单和源文件哈希，不直接改旧哈希冒充原版本。
3. 新启动器显式选择 v2 轮换入口及 `automatic_keep=1`。原 `enable_automatic_rotation` 和冻结协议校验要求 keep=2，接入时须提供对应的新版本初始化/协议入口，不能直接给旧运行标记改数字。先跑 CPU 测试，并用临时合成 checkpoint 验证轮换。当前正式 checkpoint 不用于破坏性测试。
4. 保持训练/采样/奖励/更新预算配置不变；每个新作业记录选择的轮换版本。训练耗时比较注明 I/O 实现不同。
5. 确认新进程使用新入口后才能标记“已接入”。如果无法建立安全边界，继续原方案，不为此打断当前训练。

测试命令：

```bash
/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python -m unittest discover -s posttraining_plan -p 'test_rotation*.py' -v
```
