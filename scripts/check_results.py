"""Dependency-free consistency checks of published aggregates (not raw re-scoring)."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    completion=json.loads((ROOT/'results/completion.json').read_text())
    assert completion['phase']=='TRAINING_AND_FROZEN_EVALUATIONS_COMPLETE'
    for split in ('dev','test'):
        for g in ('s0','s1','s2','s3'):
            d=json.loads((ROOT/f'results/{g}_{split}/summary.json').read_text())
            assert d['phase']=='EVALUATION_COMPLETE'
            assert d['questions']==(128 if split=='dev' else 901)
            assert d['responses']==(128 if split=='dev' else 1081)
            assert sum(b['responses'] for b in d['benchmarks'].values())==d['responses']
            for b in d['benchmarks'].values():
                assert 0 <= b['accuracy'] <= 1 and 0 <= b['truncation_rate'] <= 1
                assert b['generated_tokens']>0
            assert math.isclose(d['overall_domain_macro'],sum(d['domain_macro'].values())/2)
            print(split,g,'math',round(d['domain_macro']['math']*100,2),
                  'IF',round(d['domain_macro']['if']*100,2),'wall_s',round(d['wall_seconds']))
    for p in (ROOT/'results/comparisons').glob('*.json'):
        d=json.loads(p.read_text())
        for domain, v in d['paired']['domains'].items():
            delta=100*(d['candidate']['domain_macro'][domain]-d['baseline']['domain_macro'][domain])
            assert math.isclose(delta,v['delta_percentage_points'],abs_tol=1e-10),p
    print('Aggregate consistency passed. Raw responses are not distributed; this is not fresh model evaluation.')

if __name__=='__main__':
    main()
