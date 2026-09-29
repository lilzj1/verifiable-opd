"""Exact/near split audit, schema audit and review candidates; no model inference."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'posttraining_plan')]
from build_text_data import task_text


def main():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    folder=ROOT/'data/posttraining_text_v3'
    rows={name:[json.loads(s) for s in (folder/name).read_text().split('\n') if s.strip()]
          for name in ('online_train.jsonl','expert_train.jsonl','dev.jsonl','test_manifest.jsonl')}
    for name in ('dev.jsonl','test_manifest.jsonl'):
        if (folder/name).read_bytes() != (ROOT/'data/posttraining_text_v1'/name).read_bytes():
            raise ValueError('Expert filtering must not change online/dev/test selection')
    training=rows['online_train.jsonl']+rows['expert_train.jsonl']
    heldout=rows['dev.jsonl']+rows['test_manifest.jsonl']
    exact={r['dedup_cluster_id'] for r in training}&{r['dedup_cluster_id'] for r in heldout}
    assert not exact, 'Cross-split exact contamination'
    # Refit over final rows as an independent second near-overlap screen.
    vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),lowercase=False,dtype=np.float32,max_features=250000)
    matrix=vec.fit_transform([task_text(r['messages']) for r in training+heldout])
    pairs=[]
    scores=matrix[:len(training)]@matrix[len(training):].T
    for i in range(len(training)):
        dense=scores.getrow(i).toarray().ravel()
        j=int(dense.argmax())
        pairs.append({'similarity':float(dense[j]),'train_id':training[i]['example_id'],
                      'heldout_id':heldout[j]['example_id'],'heldout_split':heldout[j]['split'],
                      'train_prompt':task_text(training[i]['messages']),
                      'heldout_prompt':task_text(heldout[j]['messages'])})
    pairs.sort(key=lambda r:-r['similarity'])
    report={'phase':'REVIEW_REQUIRED','exact_overlaps':len(exact),'max_similarity':pairs[0]['similarity'],
        'above_090':sum(p['similarity']>=.9 for p in pairs),'review_top20':pairs[:20],
        'dev_unchanged':True,'test_predictions_accessed':False,
        'online_expert_exact_overlap':len({r['dedup_cluster_id'] for r in rows['online_train.jsonl']}&
                                        {r['dedup_cluster_id'] for r in rows['expert_train.jsonl']}),
        'manifest_sha256':hashlib.sha256((folder/'DATA_MANIFEST.json').read_bytes()).hexdigest()}
    (folder/'SPLIT_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print({k:v for k,v in report.items() if k!='review_top20'})


if __name__=='__main__':
    main()
