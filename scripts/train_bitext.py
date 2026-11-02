"""Local QLoRA pilot or checkpoint continuation with optional generation."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Import Unsloth before Transformers so its patches are applied.
from unsloth import FastModel
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, set_seed
from training_schedule import continuation_scale


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/gemma-3-4b-it")
    parser.add_argument("--run", default="bitext-4b-pilot")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--stress-test", action="store_true", help="Use longest training records for the memory trial")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--eval-per-intent", type=int, default=0)
    parser.add_argument("--logging-every", type=int, default=5)
    args = parser.parse_args()
    set_seed(20261029)
    out = ROOT / "outputs" / args.run
    if (out / 'training_summary.json').exists() or (out / 'checkpoints').exists():
        raise RuntimeError('Use a new run directory; existing training results are preserved')
    resume_step = 0
    if args.resume:
        if args.train_limit or args.stress_test:
            raise RuntimeError('Resume must preserve the complete original training dataset/order')
        if args.samples:
            raise RuntimeError('Continuation requires --samples 0; no generation during training')
        resume_step = json.loads((args.resume / 'trainer_state.json').read_text())['global_step']
        if args.steps <= resume_step:
            raise RuntimeError('--steps must be the final total, greater than checkpoint step')
    out.mkdir(parents=True, exist_ok=True)
    prepared = ROOT / "data/bitext/prepared"
    prompt = (prepared / "generation_prompt.txt").read_text(encoding="utf-8").strip()
    if args.resume:
        original = json.loads((args.resume.parent.parent / 'training_summary.json').read_text(encoding='utf-8'))
        if original['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest():
            raise RuntimeError('Resume prompt differs from the source training run')
        if original['arguments']['max_length'] != args.max_length or original['model'] != args.model:
            raise RuntimeError('Resume base model or context differs from the source run')
    started = time.time()
    model, processor = FastModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_length,
        load_in_4bit=True, full_finetuning=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    fields = json.loads((prepared / "schema.json").read_text())["fields"]

    def encode(split, limit=0):
        rows = [json.loads(line) for line in (prepared / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        if split == "train" and args.stress_test:
            rows.sort(key=lambda row: sum(len(row[field]) for field in fields), reverse=True)
        if limit:
            rows = rows[:limit]
        if split == 'validation' and args.eval_per_intent:
            counts = {}
            chosen = []
            for row in rows:
                intent = row['intent']
                if counts.get(intent, 0) < args.eval_per_intent:
                    chosen.append(row)
                    counts[intent] = counts.get(intent, 0) + 1
            rows = chosen
        encoded, rejected = [], 0
        lengths = []
        for row_index, row in enumerate(rows):
            if row_index % 5000 == 0:
                print(f'Tokenizing {split}: {row_index}/{len(rows)}', flush=True)
            answer = json.dumps({field: row[field] for field in fields}, ensure_ascii=False)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                tokenize=False, add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            # Avoid training on a partial/corrupt JSON record due to truncation.
            if len(ids) > args.max_length:
                rejected += 1
                continue
            if ids[:len(prompt_ids)] != prompt_ids:
                raise RuntimeError("Chat-template prompt prefix changed; response masking would be wrong")
            labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
            encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
            lengths.append(len(ids))
        if not encoded:
            raise RuntimeError("No complete records fit the configured sequence length")
        return Dataset.from_list(encoded), {"considered": len(rows), "retained": len(encoded),
                                            "rejected_overlength": rejected,
                                            "max_tokens": max(lengths)}

    train, train_audit = encode("train", args.train_limit)
    validation, val_audit = encode("validation", args.eval_limit)
    dump(out / "tokenization.json", {"train": train_audit, "validation_subset": val_audit,
                                     "prompt_tokens": len(prompt_ids)})
    if train_audit["rejected_overlength"] > .01 * train_audit["considered"]:
        raise RuntimeError("Over 1% of training rows exceed context: increase context before training")

    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=8, lora_alpha=16, lora_dropout=0, bias="none", random_state=20261029,
        use_gradient_checkpointing="unsloth",
    )
    if args.resume:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, load_file(str(args.resume / 'adapter_model.safetensors')))
        dump(out / 'continuation_protocol.json', {
            'checkpoint': str(args.resume.resolve()), 'start_step': resume_step,
            'final_total_steps': args.steps, 'additional_steps': args.steps - resume_step,
            'optimizer_rng_and_data_position': 'Restored by Trainer checkpoint resume',
            'scheduler': 'New continuation schedule: 50-step linear warmup from checkpoint zero LR, then decay to final step',
            'note': 'Not equivalent to a single uninterrupted long schedule; pilot is preserved',
            'validation_per_intent': args.eval_per_intent})

    def generate(label):
        FastModel.for_inference(model)
        model.eval()
        target = out / f"{label}_generations.jsonl"
        max_tokens = min(600, args.max_length - len(prompt_ids))
        if max_tokens < 300:
            raise RuntimeError("Insufficient generation context for complete records")
        with target.open("w", encoding="utf-8") as stream:
            for i in range(args.samples):
                seed = 20261029 + i
                set_seed(seed)
                inputs = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to("cuda")
                context = model.disable_adapter() if label == "base" else contextlib.nullcontext()
                start = time.time()
                with context, torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                         temperature=1., top_p=.95, top_k=64,
                                         pad_token_id=tokenizer.pad_token_id,
                                         use_cache=True)
                text = tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                stream.write(json.dumps({"sample_id": i, "seed": seed, "text": text,
                                         "seconds": time.time()-start,
                                         "new_tokens": int(ids.shape[1]-inputs["input_ids"].shape[1]),
                                         "hit_token_limit": int(ids.shape[1]-inputs["input_ids"].shape[1]) >= max_tokens}, ensure_ascii=False)+"\n")
                stream.flush()
                print(f"{label} sample {i+1}/{args.samples}", flush=True)

    if args.samples:
        generate("base")
    FastModel.for_training(model)
    model.config.use_cache = False

    def collate(batch):
        length = ((max(len(row["input_ids"]) for row in batch)+7)//8)*8
        return {key: torch.tensor([row[key]+[token]*(length-len(row[key])) for row in batch])
                for key, token in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100))}

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), max_steps=args.steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        per_device_eval_batch_size=1, learning_rate=1e-4,
        warmup_ratio=.05, weight_decay=.01, lr_scheduler_type="linear",
        bf16=True, fp16=False, optim="adamw_8bit", logging_steps=args.logging_every,
        eval_strategy="steps", eval_steps=args.checkpoint_every or max(25, args.steps//4),
        save_strategy="steps", save_steps=args.checkpoint_every or max(25, args.steps//4),
        save_total_limit=10 if args.resume else 2,
        load_best_model_at_end=bool(args.resume), metric_for_best_model='eval_loss', greater_is_better=False,
        report_to="none", seed=20261029, data_seed=20261029,
        gradient_checkpointing=True, remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    class ContinuationTrainer(Trainer):
        def create_scheduler(self, num_training_steps, optimizer=None):
            if not args.resume:
                return super().create_scheduler(num_training_steps, optimizer)
            if self.lr_scheduler is None:
                def scale(step):
                    return continuation_scale(step, resume_step, num_training_steps)
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer, scale)
                self._created_lr_scheduler = True
            return self.lr_scheduler

    trainer = ContinuationTrainer(model=model, args=training_args, train_dataset=train,
                      eval_dataset=validation, data_collator=collate)
    if args.resume:
        from transformers import TrainerCallback
        source_checkpoint = str(args.resume.resolve())
        class EvaluateFinalStep(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                # Trainer restores interval fields from the old short-run state.
                # Apply the explicitly requested continuation intervals afterward.
                state.logging_steps = args.logging_steps
                state.eval_steps = args.eval_steps
                state.save_steps = args.save_steps
                # The saved pilot remains eligible if continuation never improves.
                state.best_metric = before['eval_loss']
                state.best_model_checkpoint = source_checkpoint
                state.best_global_step = resume_step
                return control
            def on_step_end(self, args, state, control, **kwargs):
                dump(out / 'progress.json', {
                    'global_step': state.global_step, 'target_steps': state.max_steps,
                    'additional_steps_completed': state.global_step - resume_step,
                    'epoch': state.epoch, 'best_validation_loss': state.best_metric,
                    'best_checkpoint': state.best_model_checkpoint})
                if state.global_step >= state.max_steps:
                    control.should_evaluate = True
                    control.should_save = True
                return control
        trainer.add_callback(EvaluateFinalStep())
    torch.cuda.reset_peak_memory_stats()
    before = trainer.evaluate()
    dump(out / "validation_before.json", before)
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    after = trainer.evaluate()
    model.save_pretrained(out / "adapter")
    processor.save_pretrained(out / "adapter")
    dump(out / "training_summary.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, "model": args.model, "train_metrics": result.metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "continuation_accounting": {
            "starting_step": resume_step,
            "completed_additional_steps": trainer.state.global_step - resume_step,
            "additional_example_presentations": (trainer.state.global_step - resume_step) * 8,
            "note": "Trainer resumed-run throughput and loss use cumulative global steps; do not interpret them as fresh-run measurements." if args.resume else "Fresh run",
        },
        "validation_before": before, "validation_after": after,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
        "elapsed_seconds": time.time()-started,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "generation": {"temperature": 1., "top_p": .95, "top_k": 64,
                       "samples": args.samples, "quotas": False},
        "scope": "Development training; no final-test evaluation or claim of distributional success",
    })
    if args.samples:
        generate("tuned")
    print("Finished:", out, flush=True)


if __name__ == "__main__":
    main()
