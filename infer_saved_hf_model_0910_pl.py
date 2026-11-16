#!/usr/bin/env python3
"""PyTorch Lightning + DataLoader inference for checkpoints/saved_hf_model_0910.

NMR -> SMILES, top-K beams, on the NMRMind test set (or any JSON with
C_nmr/NMR + cano_smiles).  The preprocessing and generation parameters are
identical to infer_saved_hf_model_0910.py; the speedup comes from:

  - DataLoader workers prefetching/encoding the next batches
  - a larger generation batch
  - optional bf16/fp16 mixed precision on the 5090

Example (5090 is CUDA_VISIBLE_DEVICES=1):
  CUDA_VISIBLE_DEVICES=1 python infer_saved_hf_model_0910_pl.py \
      --input data/nmrmind_original/test.json \
      --output output/infer_0910_nmrmind_test_top10_pl.json \
      --num_beams 10 --num_return_sequences 10 \
      --batch_size 64 --num_workers 12 --precision bf16-mixed

Benchmark first (100 samples):
  ... --max_inputs 100 --batch_size 64
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "third_party", "bart_infer"))
from my_datasets.nmr_datasets import NmrDataset  # noqa: E402

PAD_ID = 1
SMILES_BOS = 187
SMILES_EOS = 188

STRIP_TOKENS = [
    "<SMILES>", "</SMILES>", "<pad>", "</s>", "<s>", "<unk>", "<mask>",
    "<fragment>", "</fragment>", "<C13-NMR>", "</C13-NMR>",
    "<molecular_formula>", "</molecular_formula>",
]
CORE_PATTERNS = [
    Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=CC=CC=C12"),
    Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=NC=CC=C12"),
]


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def parse_nmr(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        vals = value
    elif isinstance(value, str):
        s = value.strip()
        try:
            parsed = json.loads(s)
            vals = parsed if isinstance(parsed, list) else s.replace("[", "").replace("]", "").replace(",", " ").split()
        except Exception:
            vals = s.replace("[", "").replace("]", "").replace(",", " ").split()
    else:
        return []
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def clean_smiles(text):
    for t in STRIP_TOKENS:
        text = text.replace(t, "")
    return text.strip()


def canon(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None


def has_core(mol):
    if mol is None:
        return False
    return any(p is not None and mol.HasSubstructMatch(p) for p in CORE_PATTERNS)


class NMRTestDataset(Dataset):
    def __init__(self, path, max_inputs=0):
        data = json.load(open(path))
        if isinstance(data, dict):
            for key in ("data", "records", "items"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"input must be a JSON list: {path}")
        if max_inputs > 0:
            data = data[:max_inputs]
        self.records = data
        self.nmr_ds = NmrDataset()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        nmr = parse_nmr(rec.get("C_nmr", rec.get("NMR")))
        item = self.nmr_ds.fill_item(nmr)
        return {
            "index": idx,
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "reference_smiles": rec.get("cano_smiles", rec.get("canonical_smiles", "")),
            "input_nmr": nmr,
        }


def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention = [], []
    for x in batch:
        n = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [PAD_ID] * n)
        attention.append(x["attention_mask"] + [0] * n)
    return {
        "index": torch.tensor([x["index"] for x in batch], dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "reference_smiles": [x["reference_smiles"] for x in batch],
        "input_nmr": [x["input_nmr"] for x in batch],
    }


class NMR0910Module(pl.LightningModule):
    def __init__(self, model_dir, num_beams=10, num_return_sequences=10,
                 max_length=512, length_penalty=0.4):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.model.eval()

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].to(self.device)
        attention = batch["attention_mask"].to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_length=self.max_length,
                num_beams=self.num_beams,
                num_return_sequences=self.num_return_sequences,
                bos_token_id=SMILES_BOS,
                pad_token_id=PAD_ID,
                eos_token_id=SMILES_EOS,
                decoder_start_token_id=SMILES_BOS,
                forced_eos_token_id=SMILES_EOS,
                length_penalty=self.length_penalty,
            )
        nr = self.num_return_sequences
        texts = [clean_smiles(self.tokenizer.decode(seq, skip_special_tokens=False)) for seq in out]
        preds = [texts[i * nr:(i + 1) * nr] for i in range(len(batch["index"]))]
        return {
            "index": batch["index"].tolist(),
            "reference_smiles": list(batch["reference_smiles"]),
            "input_nmr": list(batch["input_nmr"]),
            "predictions": preds,
        }


def compute_metrics(flat, ks=(1, 3, 5, 10)):
    top = {k: 0 for k in ks}
    valid1 = 0
    valid_rate = {k: 0.0 for k in ks}
    core_pos = 0
    core_tp = {k: 0 for k in ks}
    core_neg = 0
    core_fp = {k: 0 for k in ks}

    for r in flat:
        ref_mol = Chem.MolFromSmiles(r["reference_smiles"]) if r["reference_smiles"] else None
        ref_c = canon(r["reference_smiles"])
        preds = r["predictions"]
        pred_mols = [Chem.MolFromSmiles(s) for s in preds]
        pred_cs = [canon(s) for s in preds]
        ref_core = has_core(ref_mol)
        if ref_core:
            core_pos += 1
        else:
            core_neg += 1

        for k in ks:
            ok = ref_c is not None and any(c == ref_c for c in pred_cs[:k])
            top[k] += int(ok)
            valid_rate[k] += sum(1 for m in pred_mols[:k] if m is not None) / max(1, min(k, len(pred_mols)))
            if ref_core:
                core_tp[k] += int(any(has_core(m) for m in pred_mols[:k]))
            else:
                core_fp[k] += int(any(has_core(m) for m in pred_mols[:k]))

        valid1 += int(pred_mols[0] is not None)

    n = max(1, len(flat))
    return {
        "n": len(flat),
        "core_pos": core_pos,
        "core_neg": core_neg,
        "top_n": {str(k): top[k] / n for k in ks},
        "valid_at_1": valid1 / n,
        "valid_rate_at_n": {str(k): valid_rate[k] / n for k in ks},
        "core_recall_at_n": {str(k): (core_tp[k] / core_pos if core_pos else 0.0) for k in ks},
        "core_fp_rate_at_n": {str(k): (core_fp[k] / core_neg if core_neg else 0.0) for k in ks},
    }


def main():
    p = argparse.ArgumentParser(description="Lightning + DataLoader inference for saved_hf_model_0910")
    p.add_argument("--model_dir", default=os.path.join(ROOT, "checkpoints/saved_hf_model_0910"))
    p.add_argument("--input", default=os.path.join(ROOT, "data/nmrmind_original/test.json"))
    p.add_argument("--output", default=os.path.join(ROOT, "output/infer_0910_nmrmind_test_top10_pl.json"))
    p.add_argument("--metrics_output", default=None)
    p.add_argument("--num_beams", type=int, default=10)
    p.add_argument("--num_return_sequences", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--length_penalty", type=float, default=0.4)
    p.add_argument("--max_inputs", type=int, default=0, help="limit samples for benchmark (0=all)")
    p.add_argument("--precision", default="bf16-mixed", choices=["32", "bf16-mixed", "16-mixed"])
    p.add_argument("--devices", type=int, default=1)
    args = p.parse_args()

    if args.num_return_sequences > args.num_beams:
        raise SystemExit("--num_return_sequences must be <= --num_beams")
    if args.metrics_output is None:
        args.metrics_output = args.output + ".metrics.json"

    pl.seed_everything(42, workers=True)
    ds = NMRTestDataset(args.input, max_inputs=args.max_inputs)
    log(f"dataset {args.input}: {len(ds)} samples")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate,
    )
    module = NMR0910Module(
        args.model_dir,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_length=args.max_length,
        length_penalty=args.length_penalty,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    t0 = time.time()
    batches = trainer.predict(module, dataloaders=loader, return_predictions=True)
    elapsed = time.time() - t0
    log(f"predict done: {len(batches)} batches in {elapsed:.1f}s "
        f"({len(ds)/elapsed:.2f} samples/s)")

    flat = []
    for b in batches:
        for i in range(len(b["index"])):
            flat.append({
                "index": int(b["index"][i]),
                "input_nmr": b["input_nmr"][i],
                "reference_smiles": b["reference_smiles"][i],
                "predictions": b["predictions"][i],
            })

    out_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(flat, f, ensure_ascii=False)
    log(f"wrote {out_path} ({len(flat)} samples)")

    metrics = compute_metrics(flat)
    metrics["elapsed_seconds"] = elapsed
    metrics["samples_per_second"] = len(ds) / elapsed if elapsed else 0
    metrics_path = args.metrics_output if os.path.isabs(args.metrics_output) else os.path.join(ROOT, args.metrics_output)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log("metrics: " + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
