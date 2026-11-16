#!/bin/bash
# 复跑 10-28 tf457 两腿(4090, py312_v2 老栈), 配置逐项对齐 hparams.yaml
set -x
cd /mnt/data2/NMR_Core_Mol_Detection
export CUDA_VISIBLE_DEVICES=0
PY=~/venvs/py312_tf457/bin/python

COMMON="--do_test --gpus 1 --seed 42 \
 --model_dir ./checkpoints/saved_hf_model_0910 \
 --use_ring_system \
 --batch_size 32 --max_length 512 --num_workers 12 \
 --test_file ./data/nmrmind_cmgnet_1004/external_test.json \
 --ckpt models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt \
 --num_beams 10"

$PY main_cmgnet.py $COMMON --decode_mode frag \
  --save_path output/repro1029_4090_b32_frag_beams10 \
  > logs/repro1029_frag.log 2>&1
echo "FRAG exit=$?" >> logs/repro1029_frag.log

$PY main_cmgnet.py $COMMON --decode_mode smiles \
  --save_path output/repro1029_4090_b32_smiles_beams10 \
  > logs/repro1029_smiles.log 2>&1
echo "SMILES exit=$?" >> logs/repro1029_smiles.log
