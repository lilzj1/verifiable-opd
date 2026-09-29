"""Validate all 384 outputs before deciding whether bounded training smoke is justified."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from verifiable_opd.posttraining.checkpoint_protocol import file_sha
from verifiable_opd.posttraining.evaluation import canonical_sha,summarize,paired_report

ROLES=('shared_student_initialization','math_teacher','instruction_following_teacher')


def analyze():
    source=ROOT/'data/posttraining_text_v3/dev.jsonl'
    rows=[json.loads(s) for s in source.read_text().split('\n') if s.strip()]
    by_id={r['example_id']:r for r in rows}
    predictions={}
    runs={}
    metrics={}
    expected_ids=[r['example_id'] for domain in ('math','if') for r in rows if r['domain']==domain]
    for role in ROLES:
        root=ROOT/'artifacts'/f'teacher_matrix_fp16_v1_{role}'
        exit_file=root.with_name(root.name+'_exit_code.txt')
        if not exit_file.exists():
            return {'phase':'RUNNING_OR_INCOMPLETE','pending_role':role,'allow_smoke':False,'allow_formal':False}
        if exit_file.read_text().strip()!='0':
            return {'phase':'INFRASTRUCTURE_FAILURE','failed_role':role,'exit_code':exit_file.read_text().strip(),
                    'allow_smoke':False,'allow_formal':False}
        protocol=json.loads((root/'protocol.json').read_text())
        summary=json.loads((root/'summary.json').read_text())
        if (protocol['ids']!=expected_ids or protocol['dev_sha256']!=file_sha(source)
                or protocol['dtype']!='float16' or protocol['temperature']!=.6
                or protocol['top_k']!=20 or protocol['top_p']!=.95
                or protocol['max_response_tokens']!=8192 or protocol['seed']!=17
                or summary['phase']!='PROBE_COMPLETE'):
            raise ValueError('Matrix differs from registered protocol or has checker failures')
        raw=[json.loads(s) for s in (root/'predictions.jsonl').read_text().split('\n') if s.strip()]
        cooked=[dict(record,sample_index=0,prompt_sha256=canonical_sha(by_id[record['example_id']]['messages']))
                for record in raw]
        predictions[role]=cooked
        metrics[role]=summarize(rows,cooked,'dev')
        runs[role]={'directory':str(root),'responses_sha256':file_sha(root/'predictions.jsonl'),
                    'protocol_sha256':file_sha(root/'protocol.json'),
                    'wall_seconds':summary['wall_seconds']}
    student,math_teacher,if_teacher=ROLES
    scores={role:value['domain_macro'] for role,value in metrics.items()}
    criteria={'math_teacher_over_student':scores[math_teacher]['math']>scores[student]['math'],
              'if_teacher_over_student':scores[if_teacher]['if']>scores[student]['if'],
              'math_specialization':scores[math_teacher]['math']>scores[if_teacher]['math'],
              'if_specialization':scores[if_teacher]['if']>scores[math_teacher]['if']}
    pairs={role:paired_report(rows,predictions[student],predictions[role],'dev') for role in ROLES[1:]}
    return {'phase':'QUALITY_SUPPORTS_BOUNDED_SMOKE' if all(criteria.values()) else 'QUALITY_NOT_ESTABLISHED',
            'allow_smoke':all(criteria.values()),'allow_formal':False,'criteria':criteria,
            'metrics':metrics,'teacher_vs_student_paired':pairs,'runs':runs,
            'dev_manifest_sha256':file_sha(source),
            'limits':['64 tasks per domain; previous 16-task probes overlap',
                      'Point improvement is not statistical significance',
                      'Report truncation/cost before freezing formal budget',
                      'One IF dev task has contradictory original constraint metadata']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=analyze()
    if args.output:
        with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
