"""Native-Windows QLoRA training for one prepared synthetic-record dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from unsloth import FastModel
import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from json_io import write_json


def dump(path, value):
    return write_json(path, value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['insuranceqa', 'reliability', 'mimic_demo'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--eval-limit', type=int, default=128)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('Epoch count must be positive')
    prepared = ROOT / 'data' / args.dataset / 'prepared'
    out = ROOT / 'outputs' / args.run
    if (out / 'training_summary.json').exists():
        raise RuntimeError('This run is already complete; use another name')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise RuntimeError('Existing run found: explicitly resume a saved checkpoint or use another name')
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'dataset': args.dataset, 'base': str(ROOT / '.cache/models/gemma3-4b-training'),
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'},
                'input_hashes': {name: sha(prepared / name) for name in ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'schema.json', 'generation_prompt.txt']},
                'seed': 20261031, 'source': 'Independent adapter initialized from original Gemma base; not the Bitext adapter'}
    if args.resume and json.loads((out / 'protocol.json').read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Resume configuration or data differ from the original protocol')
    dump(out / 'protocol.json', protocol)
    started = time.time()
    set_seed(20261031)
    model, processor = FastModel.from_pretrained(model_name=protocol['base'], max_seq_length=args.max_length,
                                                load_in_4bit=True, full_finetuning=False)
    tokenizer = getattr(processor, 'tokenizer', processor)
    schema = json.loads((prepared / 'schema.json').read_text(encoding='utf-8'))
    prompt = (prepared / 'generation_prompt.txt').read_text(encoding='utf-8').strip()
    prompt_text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    prefix = tokenizer(prompt_text, add_special_tokens=False)['input_ids']

    def encode(split, limit=0):
        rows = [json.loads(line) for line in (prepared / (split + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        if limit:
            rows = rows[:limit]
        encoded, rejected = [], []
        for i, row in enumerate(rows):
            answer = json.dumps({k: row[k] for k in schema['fields']}, ensure_ascii=False)
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}], tokenize=False, add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
            if len(ids) > args.max_length:
                rejected.append({'source_id': row['source_id'], 'tokens': len(ids)})
                continue
            if ids[:len(prefix)] != prefix:
                raise RuntimeError('Prompt masking boundary differs from the chat template')
            encoded.append({'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': [-100] * len(prefix) + ids[len(prefix):]})
            if i % 3000 == 0:
                print(f'Tokenizing {split}: {i}/{len(rows)}', flush=True)
        if not encoded or len(rejected) > .05 * len(rows):
            raise RuntimeError(f'{split}: no data or over 5% exceed context; inspect tokenization before training')
        return Dataset.from_list(encoded), {'considered': len(rows), 'retained': len(encoded), 'rejected_overlength': rejected,
                                            'max_tokens': max(len(r['input_ids']) for r in encoded)}

    train, train_audit = encode('train')
    validation, validation_audit = encode('validation', args.eval_limit)
    dump(out / 'tokenization.json', {'train': train_audit, 'validation': validation_audit, 'prompt_tokens': len(prefix)})
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True, r=8, lora_alpha=16, lora_dropout=0,
        bias='none', random_state=20261031, use_gradient_checkpointing='unsloth')
    FastModel.for_training(model)
    model.config.use_cache = False

    def collate(batch):
        length = ((max(len(row['input_ids']) for row in batch) + 7) // 8) * 8
        return {key: torch.tensor([row[key] + [pad] * (length - len(row[key])) for row in batch])
                for key, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}

    class Progress(TrainerCallback):
        def on_step_end(self, arguments, state, control, **kwargs):
            write_json(out / 'progress.json', {'dataset': args.dataset, 'phase': 'training', 'global_step': state.global_step,
                'target_steps': state.max_steps, 'epoch': state.epoch, 'elapsed_seconds': round(time.time()-started, 1),
                'best_validation_loss': state.best_metric}, best_effort=True)
            if state.global_step >= state.max_steps:
                control.should_evaluate = True
                control.should_save = True
            return control

    training_args = TrainingArguments(output_dir=str(out / 'checkpoints'), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, per_device_eval_batch_size=1,
        learning_rate=1e-4, warmup_ratio=.05, weight_decay=.01, lr_scheduler_type='linear',
        bf16=True, fp16=False, optim='adamw_8bit', logging_steps=5,
        eval_strategy='steps', eval_steps=args.checkpoint_every, save_strategy='steps', save_steps=args.checkpoint_every,
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        report_to='none', seed=20261031, data_seed=20261031, gradient_checkpointing=True,
        remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=training_args, train_dataset=train, eval_dataset=validation,
                      data_collator=collate, callbacks=[Progress()])
    torch.cuda.reset_peak_memory_stats()
    if args.resume:
        before = json.loads((out / 'validation_before.json').read_text(encoding='utf-8'))
    else:
        before = trainer.evaluate()
        dump(out / 'validation_before.json', before)
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    after = trainer.evaluate()
    model.save_pretrained(out / 'adapter')
    processor.save_pretrained(out / 'adapter')
    selection = {'dataset': args.dataset, 'adapter': str(out / 'adapter'),
                 'adapter_sha256': sha(out / 'adapter/adapter_model.safetensors'),
                 'best_checkpoint': trainer.state.best_model_checkpoint, 'selection_metric': 'validation completion-only loss',
                 'base_validation_loss': before['eval_loss'], 'selected_validation_loss': after['eval_loss'],
                 'test_used_for_selection': False, 'input_hashes': protocol['input_hashes']}
    dump(out / 'selection.json', selection)
    dump(out / 'training_summary.json', {'protocol': protocol, 'train_metrics': result.metrics,
        'validation_before': before, 'validation_after': after, 'best_checkpoint': trainer.state.best_model_checkpoint,
        'global_steps': trainer.state.global_step, 'elapsed_seconds': time.time()-started,
        'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
        'scope': 'Validation-selected adapter saved. Generation fidelity and final test assessment run separately.'})
    write_json(out / 'progress.json', {'dataset': args.dataset, 'phase': 'complete', 'global_step': trainer.state.global_step,
                               'target_steps': trainer.state.max_steps, 'best_validation_loss': after['eval_loss']}, best_effort=True)
    print('Saved selected adapter:', out / 'adapter', flush=True)


if __name__ == '__main__':
    main()
