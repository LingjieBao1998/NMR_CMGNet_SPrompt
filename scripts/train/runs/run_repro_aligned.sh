#!/bin/bash
set -x
cd /mnt/data2/NMR_Core_Mol_Detection
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/venvs/py312_tf457/bin/python
COMMON="--do_test --gpus 1 --seed 42 \
 --model_dir ./checkpoints/saved_hf_model_0910 \
 --use_ring_system \
 --batch_size 32 --max_length 512 --num_workers 12 \
 --ckpt models/cmgnet/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt \
 --num_beams 10"

$PY main_cmgnet.py $COMMON --decode_mode frag   --test_file ./data/aligned4500_currentNMR.json --save_path output/alignedA_cur_frag_beams10   > logs/alignedA_cur_frag.log 2>&1
$PY main_cmgnet.py $COMMON --decode_mode smiles --test_file ./data/aligned4500_currentNMR.json --save_path output/alignedA_cur_smiles_beams10 > logs/alignedA_cur_smiles.log 2>&1
$PY main_cmgnet.py $COMMON --decode_mode frag   --test_file ./data/aligned4500_aprilNMR.json   --save_path output/alignedB_apr_frag_beams10   > logs/alignedB_apr_frag.log 2>&1
$PY main_cmgnet.py $COMMON --decode_mode smiles --test_file ./data/aligned4500_aprilNMR.json   --save_path output/alignedB_apr_smiles_beams10 > logs/alignedB_apr_smiles.log 2>&1
echo ALL_DONE > logs/aligned_done.marker
