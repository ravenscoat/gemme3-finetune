"""Source-specific fidelity summaries. None of these scores establishes correctness."""
from collections import Counter
import json
import math
import random
import statistics

from dataset_catalog import core, data, parse_output, signature, source_warnings, text_at, value_at, validate_record


def tv(a, b):
    if not a or not b: return None
    x, y = Counter(signature(v) for v in a), Counter(signature(v) for v in b)
    return .5 * sum(abs(x[k]/len(a)-y[k]/len(b)) for k in x.keys() | y.keys())


def numeric(a, b):
    import numpy as np
    from scipy.stats import ks_2samp, wasserstein_distance
    x = [v for v in a if type(v) in [int,float] and math.isfinite(v)]
    y = [v for v in b if type(v) in [int,float] and math.isfinite(v)]
    result = {'generated_nonmissing':len(x), 'reference_nonmissing':len(y),
              'generated_missing_rate':sum(v is None for v in a)/len(a) if a else None,
              'reference_missing_rate':sum(v is None for v in b)/len(b) if b else None}
    if not x or not y: return result
    for label, values in [('generated',x),('reference',y)]:
        result[label] = {'min':min(values), 'p25':float(np.percentile(values,25)), 'median':statistics.median(values),
                         'p75':float(np.percentile(values,75)), 'max':max(values), 'mean':statistics.mean(values)}
    test = ks_2samp(x,y)
    result.update(ks_statistic=float(test.statistic), ks_pvalue=float(test.pvalue),
                  wasserstein=float(wasserstein_distance(x,y)),
                  generated_outside_reference_range_rate=sum(v<min(y) or v>max(y) for v in x)/len(x))
    return result


def fidelity(key, records, reference):
    source = data(key)
    paths = source['schema'].get('categorical_fields', [])
    if key == 'reliability': paths = paths + ['expected_output.routing_decision','expected_output.priority','input.context.policy_type','input.context.state']
    categorical = {}
    for path in paths:
        a, b = [value_at(r,path) for r in records], [value_at(r,path) for r in reference]
        categorical[path] = {'total_variation':tv(a,b),
                             'generated_counts':dict(Counter(str(v) for v in a)), 'reference_counts':dict(Counter(str(v) for v in b))}
    continuous = {}
    for path in source['schema'].get('numeric_fields', []):
        continuous[path] = numeric([r[path] for r in records], [r[path] for r in reference])
    for field in ['request','reply']:
        path = source['spec'].get(field)
        if path: continuous[field+'_characters'] = numeric([len(text_at(r,path)) for r in records], [len(text_at(r,path)) for r in reference])
    if key == 'reliability':
        continuous['required_action_count'] = numeric([len(r['expected_output']['required_actions']) for r in records], [len(r['expected_output']['required_actions']) for r in reference])
        pairs = {(r['category'],r['subcategory']) for r in reference}
        pair_rate = sum((r['category'],r['subcategory']) in pairs for r in records)/len(records) if records else None
    else: pair_rate = None
    result = {'records':len(records), 'reference_records':len(reference), 'categorical':categorical, 'numeric':continuous,
              'known_category_subcategory_pair_rate':pair_rate}
    if key == 'mimic_demo':
        import numpy as np
        from scipy.stats import spearmanr
        fields = source['schema']['numeric_fields']
        def correlations(rows):
            values = {}
            for i,a in enumerate(fields):
                for b in fields[i+1:]:
                    pairs = [(r[a],r[b]) for r in rows if r[a] is not None and r[b] is not None]
                    if len(pairs)<3 or len({p[0] for p in pairs})<2 or len({p[1] for p in pairs})<2: continue
                    rho=float(spearmanr(*zip(*pairs)).statistic)
                    if math.isfinite(rho): values[a+' / '+b]={'rho':rho,'n':len(pairs)}
            return values
        gen, ref = correlations(records), correlations(reference)
        result['spearman_correlations'] = {'generated':gen,'reference':ref,
            'mean_absolute_difference':float(np.mean([abs(gen[k]['rho']-ref[k]['rho']) for k in gen.keys()&ref.keys()])) if gen.keys()&ref.keys() else None}
    return result


def assess_label(key, raw, output):
    source = data(key)
    train = [core(key,r) for r in source['train']]
    known = {signature(r) for r in train}
    requests = {signature(value_at(r,source['spec']['request'])) for r in train} if source['spec']['request'] else set()
    replies = {signature(value_at(r,source['spec']['reply'])) for r in train} if source['spec']['reply'] else set()
    parsed, failures, shape, strict = 0, [], [], 0
    rows = []
    for item in raw:
        try:
            record = parse_output(item['text'])
            parsed += 1
            errors = validate_record(key, record)
            if errors: raise ValueError('; '.join(errors))
            notes = source_warnings(key, record)
            strict += not notes
            row = {'sample_id':item['sample_id'], 'record':record, 'source_warnings':notes,
                   'exact_training_record_match':signature(record) in known,
                   'request_seen':signature(value_at(record,source['spec']['request'])) in requests if requests else None,
                   'response_seen':signature(value_at(record,source['spec']['reply'])) in replies if replies else None}
            rows.append(row)
            shape.append(record)
        except (ValueError, TypeError, KeyError) as exc:
            failures.append({'sample_id':item['sample_id'],'error':str(exc)})
    output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    n=len(raw)
    z=1.96; denominator=1+z*z/n
    center=(len(rows)/n+z*z/(2*n))/denominator
    margin=z*math.sqrt((len(rows)/n)*(1-len(rows)/n)/n+z*z/(4*n*n))/denominator
    def rate(field):
        vals=[r[field] for r in rows if r[field] is not None]
        return sum(vals)/len(vals) if vals else None
    return {'attempted':n,'json_parseable':parsed,'structure_valid':len(rows),'source_constraint_valid':strict,
            'structure_valid_rate':len(rows)/n, 'structure_valid_wilson95':[max(0,center-margin),min(1,center+margin)],
            'failures':failures, 'token_limit_count':sum(r.get('hit_token_limit',False) for r in raw),
            'exact_training_record_match_rate':rate('exact_training_record_match'), 'training_request_match_rate':rate('request_seen'),
            'training_reply_match_rate':rate('response_seen'), 'unique_record_rate':len({signature(r) for r in shape})/len(shape) if shape else None,
            'source_warning_counts':dict(Counter(w for r in rows for w in r['source_warnings'])),
            'fidelity_vs_training':fidelity(key,shape,train)}, rows
