#!/usr/bin/env python3
"""Comment R2.4 -- MC-Dropout uncertainty of the CMGNet generator.

The classifier (XGBoost) is deterministic; MC-Dropout is applied only to the
BART-based CMGNet generator.  Dropout in this checkpoint is functional
(p = 0.1), so the whole module is placed in train() mode during generation.

For each selected spectrum we run T stochastic passes (beam = 5, 5 returned
sequences per pass) and summarise molecular-level variability:
  * distinct canonical SMILES / passes and modal top-1 frequency
  * distinct Bemis-Murcko scaffolds
  * validity rate
  * mean pairwise cosine of CReSS SMILES embeddings (no fingerprint similarity)
  * frequency of passes whose top-5 contains the quinolone core
  * SD of the top-1 sequence score (reported as a sequence-level score, NOT as
    a calibrated probability that the structure is correct)

Outputs
  reports/mc_dropout_generator.json
  reports/fig_mc_dropout.png / .pdf
"""
import json
import os
import sys
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RDLogger.DisableLog("rdApp.*")
import os as _os
B = _os.environ.get('NMR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
B = B.rstrip('/') + '/'
sys.path.insert(0, B)
MODEL_DIR = B + "checkpoints/bart_base_0910"
OUT = B + "reports/"
T = 20
BEAMS = 5
RETURN = 5
MAX_LEN = 128
START, END, NUMBER_START = 189, 190, 191
BOS, EOS = 187, 188
CORE_SMARTS = [
    Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=CC=CC=C12"),
    Chem.MolFromSmiles("O=C1C(C(O)=O)=CNC2=NC=CC=C12"),
]


def log(m):
    print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def nmr_ids(nmr):
    vals = sorted({int(round((v + 50) * 10)) for v in nmr})
    vals = [min(max(v, 0), 3399) for v in vals]
    return [START] + [v + NUMBER_START for v in vals] + [END]


def canon(s):
    s = s.replace("<SMILES>", "").replace("</SMILES>", "").strip()
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def has_core(mol):
    return any(mol.HasSubstructMatch(sm) for sm in CORE_SMARTS)


def choose_spectra(d):
    y, p = d["y"], d["p_raw"]
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    order = pos[np.argsort(p[pos])]
    sel = []
    sel += [(int(order[0]), "FN #12 (p=0.012)")]
    sel += [(int(i), "borderline positive") for i in order[1:5]]
    conf_pos = order[-8:]
    sel += [(int(i), "confident positive") for i in conf_pos]
    fp = neg[p[neg] > 0.4]
    sel += [(int(i), "borderline negative (FP)") for i in fp[:4]]
    rng = np.random.default_rng(0)
    tn = rng.choice(neg[p[neg] < 1e-3], 4, replace=False)
    sel += [(int(i), "confident negative") for i in tn]
    # de-duplicate
    seen, out = set(), []
    for i, g in sel:
        if i not in seen:
            seen.add(i)
            out.append((i, g))
    return out


def main():
    d = dict(np.load(B + "reports/calib_data/external_apr18peak.npz",
                     allow_pickle=True))
    import json as _json
    recs = _json.load(open(B + "data/nmrmind_cmgnet_1004/"
                           "external_test_april18peak.json"))

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).cuda()
    # use the checkpoint reported in the manuscript (1029_v2), not the 0910 HF export
    ckpt = torch.load(B + "models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt",
                      map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    info = model.load_state_dict(sd, strict=False)
    log(f"checkpoint loaded: missing={len(info.missing_keys)} "
        f"unexpected={len(info.unexpected_keys)}")
    model.eval()
    spectra = choose_spectra(d)
    log(f"{len(spectra)} spectra, T={T}, beams={BEAMS}")

    rows = []
    for k, (idx, group) in enumerate(spectra):
        nmr = recs[idx]["NMR"]
        ids = torch.tensor(nmr_ids(nmr)).unsqueeze(0).cuda()
        am = torch.ones_like(ids)
        per_pass, per_score, top1s = [], [], []
        model.train()
        for t in range(T):
            with torch.no_grad():
                o = model.generate(
                    input_ids=ids, attention_mask=am, max_length=MAX_LEN,
                    num_beams=BEAMS, num_return_sequences=RETURN,
                    bos_token_id=BOS, pad_token_id=1, eos_token_id=EOS,
                    decoder_start_token_id=BOS, output_scores=True,
                    return_dict_in_generate=True)
            seqs = [canon(tok.decode(s, skip_special_tokens=True))
                    for s in o.sequences]
            per_pass.append(seqs)
            score = float(o.sequences_scores[0].item())
            per_score.append(score)
            top1s.append(seqs[0] if seqs and seqs[0] else "")
        model.eval()
        with torch.no_grad():
            o = model.generate(
                input_ids=ids, attention_mask=am, max_length=MAX_LEN,
                num_beams=BEAMS, num_return_sequences=RETURN,
                bos_token_id=BOS, pad_token_id=1, eos_token_id=EOS,
                decoder_start_token_id=BOS, output_scores=True,
                return_dict_in_generate=True)
        det = [canon(tok.decode(s, skip_special_tokens=True)) for s in o.sequences]

        all_seqs = [s for pp in per_pass for s in pp if s]
        uniq = sorted(set(all_seqs))
        top1_mode = max(set([s for s in top1s if s]),
                        key=[s for s in top1s if s].count) if any(top1s) else ""
        mode_freq = (sum(s == top1_mode for s in top1s) / T) if top1_mode else 0.0
        mols = [Chem.MolFromSmiles(s) for s in uniq]
        mols = [m for m in mols if m is not None]
        scaf = {MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols}
        core_pass = 0
        for pp in per_pass:
            if any(s and (Chem.MolFromSmiles(s) is not None) and
                   has_core(Chem.MolFromSmiles(s)) for s in pp):
                core_pass += 1
        rows.append(dict(
            index=int(idx), group=group,
            p_xgb=float(d["p_raw"][idx]), y=int(d["y"][idx]),
            n_peaks=len(nmr), nmr=[float(x) for x in nmr],
            n_candidates=len(all_seqs), n_unique=len(uniq),
            unique_frac=len(uniq) / max(1, len(all_seqs)),
            n_scaffolds=len(scaf),
            top1_mode=top1_mode, top1_mode_freq=float(mode_freq),
            valid_frac=float(len(mols) / max(1, len(uniq))),
            core_pass_freq=float(core_pass / T),
            det_top1=det[0] if det else "",
            det_top1_has_core=bool(det and det[0] and
                                   Chem.MolFromSmiles(det[0]) is not None and
                                   has_core(Chem.MolFromSmiles(det[0]))),
            score_mean=float(np.mean(per_score)),
            score_sd=float(np.std(per_score, ddof=1)) if T > 1 else 0.0,
            pass_smiles=per_pass,
        ))
        log(f"{k+1}/{len(spectra)} idx {idx} {group}: uniq {len(uniq)}, "
            f"top1 mode {mode_freq:.2f}, scaffolds {len(scaf)}, "
            f"core passes {core_pass}/{T}")

    # CReSS cosine dispersion over generated molecules
    from infer import ModelInference
    mi = ModelInference(B + "models/2_5_w_model/8.json",
                        B + "models/2_5_w_model/8.pth", "cuda")
    for r in rows:
        smis = [s for s in set(s for pp in r["pass_smiles"] for s in pp if s)]
        if len(smis) < 2:
            r["cress_cos_mean"] = 1.0
            r["cress_cos_sd"] = 0.0
            continue
        embs = []
        for i in range(0, len(smis), 128):
            e = mi.smiles_encode(smis[i:i + 128])
            embs.append(e.detach().cpu().numpy())
        E = np.concatenate(embs, 0)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        C = E @ E.T
        iu = np.triu_indices(len(E), 1)
        r["cress_cos_mean"] = float(C[iu].mean())
        r["cress_cos_sd"] = float(C[iu].std())

    json.dump(dict(T=T, beams=BEAMS, return_sequences=RETURN,
                   dropout_p=0.1, spectra=rows),
              open(OUT + "mc_dropout_generator.json", "w"), indent=2)
    log("saved reports/mc_dropout_generator.json")

    # quick summary
    import collections
    for g in sorted(set(r["group"] for r in rows)):
        sub = [r for r in rows if r["group"] == g]
        log(f"{g:26s} n={len(sub)} unique={np.mean([r['n_unique'] for r in sub]):.1f} "
            f"mode_freq={np.mean([r['top1_mode_freq'] for r in sub]):.2f} "
            f"scaffolds={np.mean([r['n_scaffolds'] for r in sub]):.1f} "
            f"core_freq={np.mean([r['core_pass_freq'] for r in sub]):.2f} "
            f"cos={np.mean([r['cress_cos_mean'] for r in sub]):.3f}")


if __name__ == "__main__":
    main()
