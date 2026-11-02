"""One frozen-model test assessment. No optimization, filtering or model selection."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

from evaluate_bitext import distributions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/final-assessment'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def main():
    selected = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    adapter = Path(selected['adapter'])
    test_path = ROOT/'data/bitext/prepared/test.jsonl'
    generation = ROOT/'outputs'/selected['generation_run']
    for path, expected in [(adapter/'adapter_model.safetensors', selected['adapter_sha256']),
                           (test_path, selected['test_split_sha256']),
                           (generation/'tuned_generations.jsonl', selected['generated_outputs_sha256'])]:
        if sha(path) != expected:
            raise RuntimeError(f'Frozen artifact changed: {path}')
    rows = [json.loads(line) for line in test_path.read_text(encoding='utf-8').splitlines()]
    schema = json.loads((ROOT/'data/bitext/prepared/schema.json').read_text())
    prompt = (ROOT/'data/bitext/prepared/generation_prompt.txt').read_text(encoding='utf-8').strip()
    if hashlib.sha256(prompt.encode()).hexdigest() != selected['generation_protocol']['prompt_sha256']:
        raise RuntimeError('Frozen prompt changed')
    result_path = OUT/'test_record_losses.jsonl'
    previous = [json.loads(line) for line in result_path.read_text(encoding='utf-8').splitlines()] if result_path.exists() else []
    if [r['source_line'] for r in previous] != [r['source_line'] for r in rows[:len(previous)]] or len(previous) > len(rows):
        raise RuntimeError('Invalid saved test progress')
    started = time.time()
    if len(previous) < len(rows):
        from unsloth import FastModel
        import torch
        model, processor = FastModel.from_pretrained(model_name=str(adapter), max_seq_length=1024,
                                                     load_in_4bit=True, full_finetuning=False)
        tokenizer = getattr(processor, 'tokenizer', processor)
        FastModel.for_inference(model)
        model.eval()
        prefix = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=False, add_generation_prompt=True)
        prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        with result_path.open('a', encoding='utf-8') as stream:
            for i in range(len(previous), len(rows)):
                row = rows[i]
                answer = json.dumps({field:row[field] for field in schema['fields']}, ensure_ascii=False)
                text = tokenizer.apply_chat_template([{'role':'user','content':prompt},
                    {'role':'assistant','content':answer}], tokenize=False, add_generation_prompt=False)
                ids = tokenizer(text, add_special_tokens=False)['input_ids']
                if len(ids) > 1024 or ids[:len(prefix_ids)] != prefix_ids:
                    raise RuntimeError('Test record exceeds context or prompt masking changed; do not truncate/drop it')
                inputs = torch.tensor([ids], device='cuda')
                labels = inputs.clone()
                labels[:, :len(prefix_ids)] = -100
                with torch.inference_mode():
                    output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels, use_cache=False)
                    loss = float(output.loss.float().cpu())
                if not math.isfinite(loss):
                    raise RuntimeError('Non-finite test loss')
                item = {'source_line':row['source_line'], 'intent':row['intent'],
                        'target_tokens':len(ids)-len(prefix_ids), 'mean_token_nll':loss}
                previous.append(item)
                stream.write(json.dumps(item)+'\n')
                stream.flush()
                if (i+1) % 50 == 0 or i+1 == len(rows):
                    progress = {'completed':i+1, 'total':len(rows), 'elapsed_this_invocation_seconds':time.time()-started}
                    dump('progress.json', progress)
                    print(json.dumps(progress), flush=True)
                del output, inputs, labels
    total_tokens = sum(r['target_tokens'] for r in previous)
    token_nll = sum(r['mean_token_nll']*r['target_tokens'] for r in previous)/total_tokens
    by_intent = defaultdict(list)
    for item in previous:
        by_intent[item['intent']].append(item['mean_token_nll'])
    generated = [json.loads(line) for line in (generation/'tuned_unfenced_manual_review.jsonl').read_text(encoding='utf-8').splitlines()]
    core = json.loads((generation/'evaluation_unfenced.json').read_text())['tuned']
    result = {'frozen_selection':selected, 'test_records':len(rows), 'target_tokens':total_tokens,
              'mean_record_loss':sum(r['mean_token_nll'] for r in previous)/len(previous),
              'token_weighted_nll':token_nll, 'token_weighted_perplexity':math.exp(token_nll),
              'per_intent_mean_record_loss':{k:sum(v)/len(v) for k,v in sorted(by_intent.items())},
              'generated_attempts':core['attempted'], 'generated_valid':core['valid'],
              'generated_vs_test_distributions':distributions(generated, rows),
              'scope':'Full reserved test evaluated once after user model selection. No weight updates. Generated records were frozen before test access.',
              'limitations':['Loss measures prediction of source records, not response correctness.',
                             'Distribution distances compare 500 generations to test data; finite-sample variation remains.',
                             'Tag frequencies do not validate tag semantics.',
                             'Source copying is reported separately in development assessment.',
                             'AI review is not insurance-expert certification. No universal pass threshold is claimed.']}
    dump('final_test_results.json', result)
    distances = result['generated_vs_test_distributions']
    lines = ['# Final frozen-model test assessment', '',
             'Selected model: Gemma 3 4B QLoRA, full-pass checkpoint 3,900. The user selected it before the reserved test was assessed. Generation settings and outputs were frozen by hash.', '',
             f'- Test records assessed: {len(rows):,}.',
             f'- Mean per-record completion loss: {result["mean_record_loss"]:.4f}.',
             f'- Token-weighted completion NLL: {token_nll:.4f}; perplexity: {math.exp(token_nll):.3f}.',
             f'- Frozen generated outputs: {core["valid"]}/{core["attempted"]} valid after outer-fence normalization.',
             f'- Intent TV distance from test: {distances["intent_total_variation"]:.4f}.',
             f'- Category TV distance from test: {distances["category_total_variation"]:.4f}.', '',
             'The model predicted held-out records under the same prompt and completion-only loss mask used for training. No optimizer was created and no training occurred. All records were retained; overlength/masking failures would stop the assessment.', '',
             'Per-intent losses, length and tag distances, model/output hashes, and denominators are in outputs/final-assessment/final_test_results.json.', '',
             *['- '+line for line in result['limitations']], '',
             'These results supplement the original/pilot/full-pass development comparison; they do not retroactively change the selected model.', '']
    (ROOT/'reports/final_test_assessment.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Final frozen-model test assessment complete.', flush=True)


if __name__ == '__main__':
    main()
