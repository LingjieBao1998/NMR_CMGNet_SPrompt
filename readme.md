# NMR_Core_Mol_Detection

## core molecule
core_smiles
```
OC(C1=C([R2])N([R1])C2=[R]([R8])C([R7])=C([R6])C([R5])=C2C1=O)=O
```
> while `R` is `C` or `N` 

<img src="assets/core_mol.svg"></img>

while `R` is `C`
```
OC(C1=CNC2=CC=CC=C2C1=O)=O
```
<img src="assets/core_mol_c.svg"></img>

while `R` is `N`
```
OC(C1=CNC2=NC=CC=C2C1=O)=O
```
<img src="assets/core_mol_n.svg"></img>


## data
### 数据处理
#### nmrshiftdb2_2024
```python
python nmrshift_db_data_process.py
```

#### nmrshiftdb2_2024
```python
python nmrshift_db_data_process.py --file_path "/mnt/e/image_caption/NMRDataBase/data/nmrshiftdb2_2018/mol_nmrshift_nmrshiftdb2_2018_C_train.pkl" --save_path "./data/mol_nmrshift_nmrshiftdb2_2018_C_train_processed.sdf"
```

#### multimodal_spectroscopic_dataset
```python
python parquet_parser.py 
```

### 过滤 (core molecule query)
#### 层次一（子结构匹配（指纹查询））
<img src="assets/nmrshiftdb_2024_query.png"></img>

其中只有下面的分子有化学位移

<img src="assets/nmrshiftdb_2024_query_2.png"></img>

#### 层次二 (子结构查询（smart查询）)

## model

### 外部测试集效果（2026-10-26）

外部测试集 4,502 条 / 母核正例 64（`data/nmrmind_cmgnet_1004/external_test.json`）。
结果为 3 个 seed（42 / 0 / 2006）间的 mean ± std（ddof=1）。

#### XGBoost-CReSS 检测头（阈值 0.5）

| 训练配置 | Precision | Recall | F1 |
|---|---|---|---|
| exp_lit（仅实验语料 ~5 万，正 66） | 1.000 ± 0.000 | 0.193 ± 0.033 | 0.322 ± 0.046 |
| no_pubchem（new_db，无 PubChem 正例） | 0.993 ± 0.013 | 0.688 ± 0.016 | 0.812 ± 0.012 |
| ex_train（new_db + 1209 正例 + ex_train） | 0.934 ± 0.000 | 0.891 ± 0.000 | 0.912 ± 0.000 |
| nmrmind（NMRMind 训练集） | 0.974 ± 0.023 | 0.385 ± 0.009 | 0.552 ± 0.012 |

> 外部召回主要由 7 万条 PubChem 来源正例驱动；exp_lit 的 P=1.0 是阈值收得极紧（仅报 10~14 个样本）。

#### CMGNet 生成式（top-N 内任一候选含母核判阳）

frag 表示（smiles 略差，详见聚合报告）：

| 训练配置 | 指标 | top-1 | top-3 | top-5 | top-10 |
|---|---|---|---|---|---|
| exp30ep（仅实验语料，best） | P | 0.333 ± 0.577 | 0.667 ± 0.577 | 0.667 ± 0.577 | 0.667 ± 0.577 |
| | R | 0.005 ± 0.009 | 0.016 ± 0.016 | 0.016 ± 0.016 | 0.016 ± 0.016 |
| | F1 | 0.010 ± 0.018 | 0.031 ± 0.030 | 0.031 ± 0.030 | 0.031 ± 0.030 |
| extrain5ep（last） | P | 0.941 ± 0.037 | 0.917 ± 0.055 | 0.899 ± 0.071 | 0.883 ± 0.086 |
| | R | 0.588 ± 0.063 | 0.682 ± 0.039 | 0.688 ± 0.041 | 0.688 ± 0.041 |
| | F1 | 0.723 ± 0.055 | 0.782 ± 0.044 | 0.779 ± 0.052 | 0.773 ± 0.057 |
| 1209extrain（seed42=1ep重训） | P | 0.910 ± 0.027 | 0.837 ± 0.011 | 0.788 ± 0.034 | 0.770 ± 0.042 |
| | R | 0.943 ± 0.009 | 0.964 ± 0.009 | 0.964 ± 0.009 | 0.969 ± 0.016 |
| | F1 | 0.926 ± 0.010 | 0.896 ± 0.008 | 0.867 ± 0.022 | 0.858 ± 0.032 |

> 1209extrain 的 seed42 采用 1ep 重训版（`train_newdb_1209_extrain_5ep_seed42_retrain`）：短训外部召回显著优于原版 3ep last（frag top-1 F1 0.938 vs 0.835）。
> 纯实验语料（exp30ep / exp_lit）在外部集近乎失效（R≈0.02~0.19），换 new_db 语料后 frag top-1 F1 提升 +0.92。

详细报告：`output/xgb_cress_external_test_summary_20261026.md`、`output/cmgnet_topn_external_summary_20261026.md`、`output/exp_corpus_summary_20261026.md`、`output/xgb_seed_aggregate_stats.md`

## script

## 爬虫
```bash
python query_nmr_prediction.py
```

```bash
python query_nmr_prediction_test.py
```

## 计算误差
### 阳性样本
```bash
python Hungarian_Matching/NMR_Hungarian_Matching.py
```

```
        mae       mape    mae_max   mape_max     mae_all   mape_all
count  58.000000  58.000000  58.000000  58.000000   58.000000  58.000000
mean    2.880257   0.047976  11.965259   0.284981   44.683534   0.765805
std     1.292837   0.027676   9.513915   0.217229   23.255267   0.514590
min     1.016667   0.007839   3.900000   0.038885   12.200000   0.094073
25%     2.092708   0.028006   7.125000   0.101827   31.257500   0.389005
50%     2.562964   0.040586   9.425000   0.209302   36.535000   0.602113
75%     3.288162   0.062992  12.150000   0.445412   49.292500   0.975071
max     8.364118   0.131439  51.300000   1.026316  142.190000   2.234469
```

<img src="/mnt/d/wsl_backend/git_download/NMR_Core_Mol_Detection/assets/Hungarian_Matching.png"></img>

### NMRShiftDB(TODO)
```bash
python NMR_Hungarian_Matching2.py
```



## 补充知识
### 
