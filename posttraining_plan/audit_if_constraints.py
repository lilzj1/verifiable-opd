"""Flag objectively contradictory paragraph metadata; never filter samples."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def main():
    report={'action':'AUDIT_ONLY_NO_DATA_CHANGES','splits':{}}
    for split,name in [('train','online_train.jsonl'),('dev','dev.jsonl')]:
        rows=[json.loads(s) for s in (ROOT/'data/posttraining_text_v3'/name).read_text().split('\n') if s.strip()]
        flags=[]
        for row in rows:
            checker=row['checker']
            if checker['type']!='nemotron_if':continue
            for key,args in zip(checker['instruction_id_list'],checker['kwargs'],strict=True):
                args=args or {}
                if key=='length_constraints:nth_paragraph_first_word' and (
                    args.get('nth_paragraph',1)>args.get('num_paragraphs',10**9)):
                    flags.append({'example_id':row['example_id'],'instruction_id':key,'kwargs':args,
                                  'reason':'Requested paragraph index exceeds requested total paragraphs'})
        report['splits'][split]={'if_rows':sum(r['domain']=='if' for r in rows),'flags':flags}
    output=ROOT/'artifacts/if_constraint_metadata_v1.json'
    with output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
