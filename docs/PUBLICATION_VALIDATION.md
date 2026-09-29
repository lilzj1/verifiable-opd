# 发布验证 · 2026-09-29

- 在独立 `github_publish/verifiable-opd` 快照中运行 Math/IF CPU 回归：**111 tests / OK**。
- CPU 测试使用本机既有训练解释器与已核验上游源码；未声称从空白机器全新安装环境后重新验证，也未为发布重跑 GPU 训练。
- `scripts/check_results.py`：8组 dev/test 汇总及8份配对报告一致性通过。它不是从原始回答重新评分。
- 文件扫描：检查常见 GitHub/API 密钥、私钥标记、权重/数据库/日志/大文件及链接；没有发现匹配项。自动扫描不构成绝对无泄露保证。
- `SOURCE_MANIFEST.json` 记录从真实项目复制文件的来源和SHA256；新增发布文档与夹具统一纳入 `PUBLICATION_MANIFEST.json`。
- 本仓库不包含 checkpoint、模型、原始数据、逐题回答或虚拟环境；三个兼容测试配置只是公开模型的 config.json，来源见固定资产 revision。
- 实验代码未为发布改写，原机器控制器路径保持为历史证据；新机器迁移要求见 `REPRODUCE.md`。模型/数据需要另外获取，历史 smoke/restore 准入证据不可伪造复用。
- 旧模运算阶段为历史实验，不把其成功率混入 Math/IF 最终结果。
