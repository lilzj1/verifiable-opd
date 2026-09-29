import copy
import unittest
from verifiable_opd.posttraining.evaluation import canonical_sha, expected_samples, summarize, paired_report


class EvaluationTests(unittest.TestCase):
    def setup_records(self):
        rows=[dict(example_id=f'q{i}',split='test',domain='math' if i<2 else 'if',
                   messages=[{'role':'user','content':f'question {i}'}],
                   source_path=f'benchmark_{i}',checker={'type':'unused'}) for i in range(4)]
        records=[dict(example_id=key,sample_index=sample,domain=row['domain'],
                      prompt_sha256=canonical_sha(row['messages']),evaluation={'score':int(sample==0)},
                      response_token_ids=[1,2],finish_reason='stop')
                 for (key,sample),row in expected_samples(rows,'test').items()]
        return rows,records

    def test_math_not_pass_at_four(self):
        rows,records=self.setup_records()
        result=summarize(rows,records,'test')
        self.assertEqual(result['responses'],10)
        self.assertEqual(result['domain_macro'],{'math':.25,'if':1.})
        self.assertEqual(result['overall_domain_macro'],.625)

    def test_reject_missing_duplicate_changed_prompt_and_checker_error(self):
        rows,records=self.setup_records()
        variants=[records[:-1],records+[records[0]]]
        for field,value in [('prompt_sha256','wrong'),('checker_error','timeout')]:
            changed=copy.deepcopy(records)
            changed[0][field]=value
            variants.append(changed)
        for variant in variants:
            with self.assertRaises(ValueError):summarize(rows,variant,'test')

    def test_paired_bootstrap_clusters_repeated_responses(self):
        rows,records=self.setup_records()
        report=paired_report(rows,records,copy.deepcopy(records),'test',draws=100)
        self.assertEqual(report['domains']['math']['questions'],2)
        self.assertEqual(report['domains']['math']['paired_question_bootstrap_95_percentile_pp'],[0,0])


if __name__=='__main__':unittest.main()
