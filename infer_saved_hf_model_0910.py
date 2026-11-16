#!/usr/bin/env python3
"""NMR -> SMILES inference for checkpoints/saved_hf_model_0910.

The checkpoint is a HuggingFace BartForConditionalGeneration (103M params)
trained with the CMGNet/BART tokenizer.  This script reproduces the original
remote infer.py preprocessing:

  NMR  -> input_ids = [189] + [191 + round((ppm+50)*10)] + [190]
  output tokens     = <SMILES> ... </SMILES>  (187 ... 188)

Special ids: 1 <pad>, 187 <SMILES>, 188 </SMILES>, 189 <C13-NMR>, 190 </C13-NMR>.

Note: saved_hf_model_0910's generation_config forces </SMILES>, so this
checkpoint is used for NMR->SMILES.  The reverse direction is not exposed here.

Examples
--------
# one NMR list, 10 beams
python infer_saved_hf_model_0910.py \
    --nmr "175.4,164.8,152.66,152.1,141.7,134.8,133.5,132.5,132.4,128.8,122.9,120.9,118.8,114.4,114.3,108.9,39.7,10.1" \
    --num_beams 10 --num_return_sequences 10

# JSON list of records with C_nmr/NMR (optionally cano_smiles as reference)
python infer_saved_hf_model_0910.py \
    --input data/test_0401/filtered_with_f_split_fixed_stereo.json \
    --output output/infer_0401_with_f_split.json \
    --num_beams 10 --num_return_sequences 10
"""
import argparse
import json
import os
import sys
import time

import torch
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


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def parse_nmr(value):
    """Accept list/tuple, JSON string, or comma/space separated string."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        vals = value
    elif isinstance(value, str):
        s = value.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                vals = parsed
            else:
                vals = s.replace("[", "").replace("]", "").replace(",", " ").split()
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


def load_records(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        for key in ("data", "records", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"input must be a JSON list (or {{'data': [...]}}): {path}")
    return data


def pad_batch(items, pad_id=PAD_ID):
    max_len = max(len(ids) for ids, _ in items)
    input_ids, attention = [], []
    for ids, mask in items:
        n = max_len - len(ids)
        input_ids.append(ids + [pad_id] * n)
        attention.append(mask + [0] * n)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention, dtype=torch.long)


def run_nmr2smiles(model, tokenizer, nmrs, args, device):
    ds = NmrDataset()
    items = []
    for nmr in nmrs:
        item = ds.fill_item(nmr)
        items.append((item["input_ids"], item["attention_mask"]))

    all_preds = []
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        input_ids, attention = pad_batch(batch)
        input_ids = input_ids.to(device)
        attention = attention.to(device)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_length=args.max_length,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                bos_token_id=SMILES_BOS,
                pad_token_id=PAD_ID,
                eos_token_id=SMILES_EOS,
                decoder_start_token_id=SMILES_BOS,
                forced_eos_token_id=SMILES_EOS,
                length_penalty=args.length_penalty,
            )
        nr = args.num_return_sequences
        texts = [clean_smiles(tokenizer.decode(seq, skip_special_tokens=False)) for seq in out]
        for i in range(len(batch)):
            all_preds.append(texts[i * nr:(i + 1) * nr])
        log(f"  batch {start//args.batch_size + 1}/{(len(items)-1)//args.batch_size + 1}: "
            f"{len(batch)} inputs, {time.time()-t0:.1f}s")
    return all_preds


def main():
    p = argparse.ArgumentParser(description="NMR -> SMILES inference for saved_hf_model_0910")
    p.add_argument("--model_dir", default=os.path.join(ROOT, "checkpoints/saved_hf_model_0910"))
    p.add_argument("--input", help="JSON list of records with C_nmr/NMR")
    p.add_argument("--nmr", help="single comma/space separated NMR shift list")
    p.add_argument("--output", help="output JSON path (default: output/infer_<time>.json)")
    p.add_argument("--num_beams", type=int, default=10)
    p.add_argument("--num_return_sequences", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--length_penalty", type=float, default=0.4)
    p.add_argument("--max_inputs", type=int, default=0, help="limit number of inputs (0 = all)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    p.add_argument("--half", action="store_true", help="use float16 on CUDA")
    args = p.parse_args()

    if args.num_return_sequences > args.num_beams:
        raise SystemExit("--num_return_sequences must be <= --num_beams")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    log(f"loading {args.model_dir} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    if args.half and device.type == "cuda":
        model = model.half()
    model.to(device).eval()

    if args.nmr is not None:
        records = [{"C_nmr": parse_nmr(args.nmr)}]
    elif args.input:
        records = load_records(args.input)
    else:
        raise SystemExit("provide --input or --nmr")
    if args.max_inputs > 0:
        records = records[:args.max_inputs]

    nmrs = [parse_nmr(r.get("C_nmr", r.get("NMR"))) for r in records]
    refs = [r.get("cano_smiles", r.get("canonical_smiles", "")) for r in records]
    for i, n in enumerate(nmrs):
        if not n:
            log(f"warning: empty NMR at index {i}")

    preds = run_nmr2smiles(model, tokenizer, nmrs, args, device)
    results = []
    for i, r in enumerate(records):
        results.append({
            "index": i,
            "input_nmr": nmrs[i],
            "reference_smiles": refs[i],
            "predictions": preds[i],
        })

    if args.output is None:
        os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
        args.output = os.path.join(
            ROOT, "output", f"infer_saved_hf_model_0910_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"wrote {out_path} ({len(results)} inputs)")

    for r in results[:3]:
        log(f"  ref={r['reference_smiles'][:60]!r} preds={r['predictions'][:2]}")


if __name__ == "__main__":
    main()
