# GPU 5/6 尝试结果：现有资源布局不支持，未强行接入

2026-09-24。状态：STOPPED_BEFORE_GPU_LAUNCH。

用户允许尝试双卡，但不强行改变训练目标或影响训练效果。本轮只检查当前资源和真实框架的资源约束，不重写教师调度。

## 实际完成

- GPU 5 空闲约47.5GiB，GPU 6约46.7GiB；后者的两个既有小进程未动。
- 使用项目固定的 `upstream/verl-v0.8.0`，调用真实 `DistillationConfig` 校验器，验证两位独立教师、各TP=1、各1副本。
- 教师资源池为1张卡：**拒绝**，两教师world_size之和为2，与池大小1不符。
- 教师资源池为2张卡：配置校验通过，但还需学生资源池至少1张卡。当前计划学生2张+教师2张，因此是四卡布局；这里不证明三卡训练显存足够。

证据：[topology_report.json](../artifacts/two_gpu_attempt_v1/topology_report.json)。校验用的是明确不存在的模型占位路径，测试发生在权重加载之前，**不能称真实模型加载或GPU显存profile通过/失败**。

## 为什么这次停止

两位教师共卡、按域串行换教师、CPU/offload评分等可能不改变数学目标，但都需要额外工程接入、路由/性能/数值验证，当前后端不能仅靠修改GPU数量实现。不能删掉一位教师或换成单教师后，仍称原定的多教师能力整合实验。

因此选择保留四卡方案，停止这次双卡长训尝试。不是证明“两张48GB在任何实现下都做不了”，也不是发现双卡本身损害训练效果。

## 本轮下载与资源收尾

曾启动固定revision学生模型的有界下载，尚未下载完权重，随后明确停止了本轮下载进程。缓存约66MiB，包含配置/tokenizer等小文件及未完成权重，可续传，未删除。没有下载教师大权重，没有后台自动启动器。

收尾复核：GPU 5占用7MiB、GPU 6占用864MiB，与尝试前一致；未启动GPU模型或训练进程，HGPO及其他进程未动。配置兼容适配与既有测试不变。

## 可复核命令

从OPD项目目录执行，输出文件须尚不存在：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  PYTHONPATH=upstream/verl-v0.8.0:. \
  /home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python \
  posttraining_plan/probe_two_gpu_topology.py \
  --output artifacts/two_gpu_attempt_v2/topology_report.json
```

后续四卡可用时续传资产，按教师验证→真实更新smoke→profile推进。单卡/双卡推理验证仍可另行进行，但不等于多教师训练已跑通。
