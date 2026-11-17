#!/bin/bash
# XGBoost-CReSS on NMRMind, seeds 0/2006/42. Uses 4090 (CUDA id 0).
set -u
cd /mnt/data2/NMR_Core_Mol_Detection

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

/home/lingjiebao/anaconda3/envs/py312/bin/python -u train_xgb_cress_nmrmind.py 2>&1 | tee -a run_xgb_cress_nmrmind.log
