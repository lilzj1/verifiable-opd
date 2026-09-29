import unittest
from verifiable_opd.posttraining.evaluation import decode_settings


class EvalDecodeTests(unittest.TestCase):
    def test_source_style_domain_temperature_long_context(self):
        cfg=dict(max_prompt_tokens=2048,max_response_tokens=30000,max_model_len=32768,
                 temperature=1.,temperature_by_domain={'math':.6,'if':1.},top_p=.95,top_k=-1)
        self.assertEqual(decode_settings(cfg,'math')['temperature'],.6)
        self.assertEqual(decode_settings(cfg,'if')['temperature'],1.)
        self.assertEqual(decode_settings(cfg,'if')['max_tokens'],30000)
        cfg['max_model_len']=8192
        with self.assertRaises(ValueError):
            decode_settings(cfg,'math')

    def test_legacy_explicit_common_temperature(self):
        cfg=dict(max_prompt_tokens=2048,max_response_tokens=8192,temperature=.6,top_p=.95)
        self.assertEqual(decode_settings(cfg,'if')['max_model_len'],10241)
        self.assertEqual(decode_settings(cfg,'if')['temperature'],.6)


if __name__=='__main__':
    unittest.main()
