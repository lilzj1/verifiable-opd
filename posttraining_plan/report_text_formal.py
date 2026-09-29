"""Write results only from complete frozen comparisons; never synthesize a gain."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--controller',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((ROOT/'artifacts/text_formal_protocol_v1/protocol.json').read_text())
    rows=['# Math/IF 双教师 OPD 与固定 SFT 回放：冻结对照结果','',
          '本轮为真实 verl 分布式训练，不沿用旧模运算实验结论。', '',
          'S0 为公开 MixSFT 共同起点；S1 为预先按 S2 profile 目标 token 预算规划的 Continue-SFT；'
          'S2 为纯 sampled-token k1 PG OPD；S3 为相同 OPD 加 alpha=0.1 固定专家 CE 回放。', '',
          'S2/S3 各 100 次更新，训练 seed=17；第10步同样暂停并验证恢复，再续至100。'
          '不宣称与未中断生成逐位等价，也不是论文原始 top-k 全分布蒸馏的严格复现。', '',
          '| 评测 | 组别 | 数学宏平均 | 指令遵循宏平均 |','|---|---|---:|---:|']
    for split in ('dev','test'):
        for group in ('S0','S1','S2','S3'):
            folder=ROOT/f'artifacts/text_formal_eval_v1_{group.lower()}_{split}'
            summary=json.loads((folder/'summary.json').read_text())
            if summary['phase']!='EVALUATION_COMPLETE':raise ValueError('Incomplete evaluation')
            values=summary['domain_macro']
            rows.append(f"| {split} | {group} | {100*values['math']:.2f}% | {100*values['if']:.2f}% |")
    rows+=['','## S3 相对 S2 的变化','']
    for split in ('dev','test'):
        report=json.loads((args.controller/f'comparison_{split}_S2_S3.json').read_text())
        for domain,item in report['paired']['domains'].items():
            lo,hi=item['paired_question_bootstrap_95_percentile_pp']
            rows.append(f"- {split}/{domain}：{item['delta_percentage_points']:+.2f} 个百分点；题目配对 bootstrap 95% 区间 [{lo:+.2f}, {hi:+.2f}]。")
    rows+=['','## 预算与局限','',
           f"- S1 冻结预算：{protocol['s1_budget']['updates']} 次更新，{protocol['s1_budget']['scheduled_target_tokens']} 个目标 token（含专家重复使用）。",
           '- AIME24/25 每题四次采样取平均，不是 pass@4，也不是作者 avg@64；IFEval/IFBench 每题一次，严格规则判分。',
           '- 两域基准先分别计算，再宏平均；检查器失败不计作模型答错。测试仅在所有最终 checkpoint 确定后使用。',
           '- 单训练 seed；题目 bootstrap 不是多次独立训练。公开测试可能存在预训练污染。',
           '- 固定专家池两域各128条；在线训练两域各512个任务槽位。短预算结果不能证明普遍有效。',
           '- S3 额外专家 CE 计算成本需计入；同更新数不等于同 FLOPs。生成长度会影响实际 token 与墙钟时间。',
           '- 负提升或置信区间跨零时如实报告，不包装为稳定提升。',
           '', f'逐题回答、token、截断和耗时见 artifacts/text_formal_eval_v1_*；配对报告见 {args.controller.relative_to(ROOT)}。','']
    path=ROOT/'TEXT_FORMAL_RESULTS.md'
    with path.open('x') as stream:stream.write('\n'.join(rows))


if __name__=='__main__':main()
