"""Evaluation contracts and paired, question-clustered summaries.

No generation, model loading, or training. Infrastructure/checker failures are
not silently converted into wrong answers. Truncated/model-invalid answers ARE
counted as model outcomes when the checker successfully returns a score.
"""
import hashlib
import json
import random
import statistics


def decode_settings(config, domain):
    """Resolve frozen domain decoding; do not silently shorten long reasoning."""
    if domain not in {'math', 'if'}:
        raise ValueError('Unknown evaluation domain')
    response = config['max_response_tokens']
    if response not in (4096, 8192, 16384, 30000):
        raise ValueError('Unregistered evaluation response budget')
    context = config.get('max_model_len', config['max_prompt_tokens'] + response + 1)
    if not config['max_prompt_tokens'] + response + 1 <= context <= 32768:
        raise ValueError('Frozen evaluation context does not fit prompt and response')
    temperatures = config.get('temperature_by_domain')
    if temperatures is not None and set(temperatures) != {'math', 'if'}:
        raise ValueError('Specify both domain temperatures explicitly')
    temperature = temperatures[domain] if temperatures is not None else config['temperature']
    if temperature < 0:
        raise ValueError('Negative evaluation temperature')
    return dict(max_model_len=context, max_tokens=response, temperature=temperature,
                top_p=config['top_p'], top_k=config.get('top_k', -1))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def expected_samples(rows, split):
    if split not in {'dev', 'test'}:
        raise ValueError('Only frozen dev/test evaluation')
    seen = set()
    expected = {}
    for row in rows:
        key = row['example_id']
        if key in seen or row['split'] != split or row['domain'] not in {'math', 'if'}:
            raise ValueError('Duplicate task, split mismatch, or unknown domain')
        seen.add(key)
        count = 4 if split == 'test' and row['domain'] == 'math' else 1
        for sample in range(count):
            expected[(key, sample)] = row
    if not expected:
        raise ValueError('Empty evaluation')
    return expected


def checked_predictions(rows, predictions, split):
    expected = expected_samples(rows, split)
    found = {}
    for record in predictions:
        key = (record['example_id'], record['sample_index'])
        if key in found or key not in expected:
            raise ValueError('Duplicate or unexpected prediction')
        source = expected[key]
        if record['domain'] != source['domain'] or record['prompt_sha256'] != canonical_sha(source['messages']):
            raise ValueError('Prediction identity/prompt mismatch')
        if record.get('checker_error') or record.get('infrastructure_error'):
            raise ValueError('Resolve infrastructure/checker errors; do not count as model failures')
        if record['evaluation']['score'] not in (0, 1, 0., 1.):
            raise ValueError('Expected binary strict outcome')
        found[key] = record
    if set(found) != set(expected):
        raise ValueError(f'Missing predictions: {len(expected)-len(found)}')
    return found


def task_scores(rows, predictions, split):
    found = checked_predictions(rows, predictions, split)
    return {row['example_id']: statistics.mean(
        found[(row['example_id'], sample)]['evaluation']['score']
        for sample in range(4 if split=='test' and row['domain']=='math' else 1)) for row in rows}


def summarize(rows, predictions, split):
    found = checked_predictions(rows, predictions, split)
    tasks = task_scores(rows, predictions, split)
    buckets = {}
    for row in rows:
        # Public source path distinguishes AIME24/25. Checker type alone cannot.
        name = row['checker']['type'] if split=='dev' else row['source_path']
        buckets.setdefault(name, []).append(row)
    metrics = {}
    for name, subset in buckets.items():
        ids = {row['example_id'] for row in subset}
        records = [v for (key, _), v in found.items() if key in ids]
        flags = [flag for record in records for flag in record['evaluation'].get('instruction_flags', [])]
        metrics[name] = {'questions': len(subset), 'responses':len(records),
            'accuracy':statistics.mean(tasks[row['example_id']] for row in subset),
            'truncation_rate':statistics.mean(r['finish_reason']=='length' for r in records),
            'generated_tokens':sum(len(r['response_token_ids']) for r in records),
            'instruction_strict_micro':statistics.mean(flags) if flags else None}
    domains = {}
    for domain in ('math','if'):
        names = {row['checker']['type'] if split=='dev' else row['source_path'] for row in rows if row['domain']==domain}
        if names:
            domains[domain] = statistics.mean(metrics[name]['accuracy'] for name in names)
    return {'scope':split, 'questions':len(rows), 'responses':len(found), 'benchmarks':metrics,
            'domain_macro':domains,'overall_domain_macro':statistics.mean(domains.values()),
            'math_sampling':'question mean over four samples on test, NOT pass@4'}


def paired_report(rows, baseline, candidate, split, draws=2000, seed=17):
    if draws < 100:
        raise ValueError('At least 100 registered bootstrap draws')
    left=task_scores(rows, baseline, split)
    right=task_scores(rows, candidate, split)
    rng=random.Random(seed)
    result={}
    for domain in ('math','if'):
        subset=[r for r in rows if r['domain']==domain]
        if not subset:
            continue
        # Preserve benchmark weighting; resample QUESTIONS within each benchmark.
        buckets={}
        for row in subset:
            name=row['checker']['type'] if split=='dev' else row['source_path']
            buckets.setdefault(name,[]).append(right[row['example_id']]-left[row['example_id']])
        point=statistics.mean(statistics.mean(values) for values in buckets.values())
        samples=sorted(statistics.mean(statistics.mean(rng.choices(values,k=len(values)))
                       for values in buckets.values()) for _ in range(draws))
        result[domain]={'delta_percentage_points':100*point,
            'paired_question_bootstrap_95_percentile_pp':[100*samples[int(.025*draws)],100*samples[int(.975*draws)-1]],
            'questions':len(subset)}
    return {'domains':result,'draws':draws,'seed':seed,
            'limitation':'Conditional on these checkpoints and training seed; not multi-training-seed uncertainty'}
