"""QLoRA fine-tuning.

    python -m ftserve.train_qlora --config configs/qlora_qwen7b.yaml

Prints progress every `logging_steps`, evaluates on the validation set every
`eval_steps`, keeps the best checkpoint by validation loss, and saves the final
LoRA adapter to <output_dir>/adapter. Prints "[train] DONE" at the end.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps for a quick smoke run")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)

    from ftserve.data import PadCollator, load_jsonl, tokenize_all

    set_seed(cfg["train"]["seed"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- tokenizer + data
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    sys_prompt = cfg["data"]["system_prompt"].strip()
    max_len = cfg["data"]["max_seq_len"]
    train_rows = load_jsonl(cfg["data"]["train_file"])
    eval_rows = load_jsonl(cfg["data"]["eval_file"])
    train_ds, dropped_t = tokenize_all(train_rows, tok, sys_prompt, max_len)
    eval_ds, dropped_e = tokenize_all(eval_rows, tok, sys_prompt, max_len)
    print(f"[train] train={len(train_ds)} (dropped {dropped_t} too long)  val={len(eval_ds)} "
          f"(dropped {dropped_e})", flush=True)

    # ---------------- 4-bit base model
    q = cfg["quantization"]
    compute_dtype = getattr(torch, q["compute_dtype"])
    if compute_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[train] GPU has no bf16, switching to float16", flush=True)
        compute_dtype = torch.float16
    bnb = BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["quant_type"],
                             bnb_4bit_use_double_quant=q["double_quant"], bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb,
                                                 torch_dtype=compute_dtype, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"])

    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], bias="none",
                                             task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[train] trainable params {trainable / 1e6:.1f}M of {total / 1e9:.2f}B "
          f"({100 * trainable / total:.2f}%)", flush=True)

    # ---------------- training
    t = cfg["train"]
    steps_per_epoch = math.ceil(len(train_ds) / (t["per_device_batch_size"] * t["grad_accum_steps"]))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * t["epochs"]
    print(f"[train] {steps_per_epoch} steps/epoch, ~{total_steps} steps total", flush=True)

    class Progress(TrainerCallback):
        def __init__(self):
            self.start = time.time()

        def on_log(self, a, state, control, logs=None, **kw):
            if not logs:
                return
            elapsed = time.time() - self.start
            eta = elapsed / max(state.global_step, 1) * (state.max_steps - state.global_step)
            shown = {k: round(v, 4) for k, v in logs.items() if isinstance(v, float)}
            print(f"[train] step {state.global_step}/{state.max_steps}  {shown}  "
                  f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)

    bf16 = compute_dtype == torch.bfloat16
    targs = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=t["epochs"], max_steps=args.max_steps,
        per_device_train_batch_size=t["per_device_batch_size"], per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum_steps"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"], warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"], bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit",
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=t["eval_steps"], save_strategy="steps", save_steps=t["save_steps"],
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=t["logging_steps"], report_to="none", seed=t["seed"],
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=PadCollator(tok.pad_token_id),
                      callbacks=[Progress(), EarlyStoppingCallback(t["early_stopping_patience"])])
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    metrics = trainer.evaluate()
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] best eval_loss {metrics.get('eval_loss'):.4f}; adapter saved to {adapter_dir}", flush=True)
    print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
