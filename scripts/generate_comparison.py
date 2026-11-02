"""Resume a generation-only comparison of a saved adapter and its frozen base.

No training, prompt changes, quotas, filtering, or replacement of failed samples.
Each label uses the same batch boundaries and batch seed schedule.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def completed_rows(path, total):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(rows) > total or [r['sample_id'] for r in rows] != list(range(len(rows))):
        raise RuntimeError(f'Non-contiguous or excessive sample IDs in {path}')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', default='bitext-4b-pilot-windows')
    parser.add_argument('--run', default='bitext-4b-evaluation-1000-windows')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--reuse-base-from', help='Reuse a completed base comparison after protocol verification')
    parser.add_argument('--max-batches-per-label', type=int, default=0,
                        help='Bounded smoke test; rerun without this to continue the same manifest')
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error('samples and batch size must be positive')
    if any(Path(name).name != name for name in (args.run, args.source_run)) or args.run == args.source_run:
        parser.error('Use distinct single-directory source and evaluation run names')
    source = ROOT / 'outputs' / args.source_run
    out = ROOT / 'outputs' / args.run
    out.mkdir(parents=True, exist_ok=True)
    adapter = source / 'adapter'
    prompt = (ROOT / 'data/bitext/prepared/generation_prompt.txt').read_text(encoding='utf-8').strip()
    training = json.loads((source / 'training_summary.json').read_text(encoding='utf-8'))
    if hashlib.sha256(prompt.encode()).hexdigest() != training['prompt_sha256']:
        raise RuntimeError('Prompt differs from the completed training run')
    protocol = {
        'source_run': args.source_run, 'samples_per_label': args.samples,
        'batch_size': args.batch_size, 'batch_seed_start': 202610300,
        'prompt_sha256': training['prompt_sha256'],
        'adapter_sha256': fingerprint(adapter / 'adapter_model.safetensors'),
        'source_manifest_sha256': fingerprint(Path(training['model']) / 'source_manifest.json'),
        'max_length': training['arguments']['max_length'],
        'temperature': 1.0, 'top_p': .95, 'top_k': 64, 'quotas': False,
        'seed_scope': 'batch; same boundaries/seeds for base and tuned, not per-example RNG',
        'purpose': 'Generation-only development evaluation; final test unused',
    }
    reuse_path = None
    if args.reuse_base_from:
        if Path(args.reuse_base_from).name != args.reuse_base_from:
            raise RuntimeError('Reuse source must be a single run-directory name')
        reuse_dir = ROOT / 'outputs' / args.reuse_base_from
        prior = json.loads((reuse_dir / 'generation_protocol.json').read_text(encoding='utf-8'))
        for key in ('samples_per_label', 'batch_size', 'batch_seed_start', 'prompt_sha256',
                    'source_manifest_sha256', 'max_length', 'temperature', 'top_p', 'top_k', 'quotas', 'seed_scope'):
            if prior[key] != protocol[key]:
                raise RuntimeError(f'Cannot reuse base: protocol differs for {key}')
        reuse_path = reuse_dir / 'base_generations.jsonl'
        prior_rows = completed_rows(reuse_path, args.samples)
        if len(prior_rows) != args.samples:
            raise RuntimeError('Reused base outputs must be complete')
        protocol['reused_base'] = {'run': args.reuse_base_from,
                                 'sha256': fingerprint(reuse_path)}
    manifest = out / 'generation_protocol.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != protocol:
            raise RuntimeError('Protocol changed; use a new run directory')
    else:
        if any(out.glob('*_generations.jsonl')):
            raise RuntimeError('Refusing to resume generation files without a protocol')
        manifest.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    if reuse_path:
        destination = out / 'base_generations.jsonl'
        if destination.exists():
            if fingerprint(destination) != fingerprint(reuse_path):
                raise RuntimeError('Existing base output differs from reuse source')
        else:
            destination.write_bytes(reuse_path.read_bytes())
    completed = {label: completed_rows(out / f'{label}_generations.jsonl', args.samples)
                 for label in ('base', 'tuned')}
    if all(len(rows) == args.samples for rows in completed.values()):
        print('Both labels already complete; nothing regenerated.', flush=True)
        return

    # Offline flags and all cache/temp directories are supplied by the Windows runner.
    from unsloth import FastModel
    import torch
    from transformers import set_seed

    model, processor = FastModel.from_pretrained(
        model_name=str(adapter), max_seq_length=protocol['max_length'],
        load_in_4bit=True, full_finetuning=False)
    tokenizer = getattr(processor, 'tokenizer', processor)
    FastModel.for_inference(model)
    model.eval()
    prompt_text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    prompt_length = len(tokenizer(prompt_text, add_special_tokens=False)['input_ids'])
    max_new = min(600, protocol['max_length'] - prompt_length)
    if max_new < 300:
        raise RuntimeError('Insufficient generation context')
    torch.cuda.reset_peak_memory_stats()
    for label in ('base', 'tuned'):
        count = len(completed[label])
        if count == args.samples:
            continue
        batches = 0
        with (out / f'{label}_generations.jsonl').open('a', encoding='utf-8') as stream:
            # Recompute an interrupted batch with its original seed/size; append only missing IDs.
            for start in range((count // args.batch_size) * args.batch_size, args.samples, args.batch_size):
                size = min(args.batch_size, args.samples - start)
                seed = protocol['batch_seed_start'] + start // args.batch_size
                set_seed(seed)
                inputs = tokenizer([prompt_text] * size, add_special_tokens=False,
                                   return_tensors='pt', padding=True).to('cuda')
                began = time.time()
                context = model.disable_adapter() if label == 'base' else contextlib.nullcontext()
                with context, torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=max_new, do_sample=True,
                        temperature=protocol['temperature'], top_p=protocol['top_p'], top_k=protocol['top_k'],
                        pad_token_id=tokenizer.pad_token_id, use_cache=True)
                torch.cuda.synchronize()
                elapsed = time.time() - began
                for offset in range(size):
                    sample_id = start + offset
                    if sample_id < count:
                        continue
                    token_ids = generated[offset, inputs['input_ids'].shape[1]:].tolist()
                    eos = model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else (eos or [])
                    ended = next((i for i, token in enumerate(token_ids) if token in eos), None)
                    if ended is not None:
                        token_ids = token_ids[:ended + 1]
                    row = {'sample_id': sample_id, 'batch_seed': seed,
                           'text': tokenizer.decode(token_ids, skip_special_tokens=True),
                           'batch_seconds': elapsed, 'batch_size': size,
                           'new_tokens': len(token_ids),
                           'hit_token_limit': ended is None and len(token_ids) >= max_new}
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                os.fsync(stream.fileno())
                count = start + size
                batches += 1
                status = {'label': label, 'completed': count, 'target': args.samples,
                          'last_batch_seconds': elapsed,
                          'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3}
                (out / 'progress.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
                print(json.dumps(status), flush=True)
                if args.max_batches_per_label and batches >= args.max_batches_per_label:
                    break
    print('Generation invocation finished. Inspect counts before claiming completion.', flush=True)


if __name__ == '__main__':
    main()
