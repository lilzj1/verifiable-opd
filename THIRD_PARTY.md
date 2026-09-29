# 来源、许可与贡献边界

- verl: https://github.com/verl-project/verl ，提交 `7aed6b230776f963fa09509c10d9c3a767d1102c`，上游 Apache-2.0。训练框架与原生蒸馏实现来自上游。
- Open-MOPD: https://github.com/BytedTsinghua-SIA/Open-MOPD ，提交 `4809a96cf85a869106ff0ff3f37d0a51e12010ae`。作为方法、公开模型/数据与检查器参考。已检出的版本没有根目录 LICENSE，不擅自重新授权或直接打包其源码；使用者须核查各子目录和数据源许可。
- IFBench: https://github.com/allenai/IFBench ，提交 `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`，上游 Apache-2.0。
- Math/IF 模型与数据均来自 `posttraining_candidate_assets.json` 中固定的 Hugging Face 仓库及 revision；仅发布获取脚本和实验汇总，不再分发权重/样本。

`scripts/fetch_upstreams.py` 将上述依赖检出到被 Git 忽略的 `upstream/`。引用代码保留原文件头与来源；本仓库不以统一许可证覆盖第三方权利。项目自有源码尚未另行授予开源许可证，公开可读不代表允许任意再分发；需要使用许可请联系仓库作者。

项目贡献为双域路由、专家 CE 回放的分布式接入、兼容性/恢复/评测工程与有限预算实证，不宣称发明 OPD、SFT 回放或第三方 benchmark。
