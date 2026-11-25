# Data

本目录包含 CMGNet-SPrompt 使用的训练/验证/测试数据、辅助评估数据和说明文件。  
大文件 LMDB 采用 pickle dict 记录格式，可直接被 `main_cmgnet.py` 和 `train_xgb_cress_new_dataset.py` 读取。

## 本次上传包含的数据

| 类别 | 路径 | 记录数 | 正样本 | 说明 |
|---|---|---:|---:|---|
| 训练-实验 | `experimental/exp_lit.lmdb` | 49,984 | 55 | 从原始 `new_db.lmdb` 中筛出的文献实验谱（见“来源”） |
| 训练-实验额外 | `experimental/ex_train.lmdb`<br>`experimental/ex_train.json`<br>`experimental/ex_train.ring.json` | 24,520 | 11 | nmrshiftdb extra train split；与评估集无 SMILES 重叠 |
| 训练-合成 | `synthetic/synth_predicted.lmdb` | 1,488,432 | 2,461 | 预测 13C 谱训练语料 |
| 训练-PubChem | `pubchem/pubchem_1209.lmdb` | 77,328 | 77,328 | PubChem 正样本，已去除 90 条评估重叠 |
| 训练-NMRMind | `nmrmind/train.lmdb` | 2,137,771 | 3,288 | NMRMind 训练集 |
| 验证 | `validation/validation.lmdb`<br>`validation/validation.json` | 40,667 | 37 | 项目校准/验证集，仅用于 validation/calibration，不用于训练 |
| 验证-NMRMind | `nmrmind/validation.lmdb`<br>`nmrmind/validation.json` | 55,167 | 81 | NMRMind 验证集 |
| 测试 | `external_test/external_test.lmdb`<br>`external_test/external_test.json`<br>`external_test_pos_only.json` | 4,502 | 64 | April-18 外部测试集（4,438 负 + 64 正） |
| 测试-NMRMind | `nmrmind/test.lmdb`<br>`nmrmind/test.json` | 55,150 | 87 | NMRMind 测试集 |
| F-split 评估 | `f_split/raw_data.json`<br>`f_split/clean_data.json` | 48 + 48 | 48 + 48 | 同一批 48 个正样本的两种峰表：raw（更多峰）/clean（C–F 合并后更少峰） |
| 模拟谱 | `sim5/fluoroquinolones_nmr.json` | 5 | 5 | 5 条氟喹诺酮模拟 13C 谱，用于小规模 demo/对比 |
| 元数据 | `manifest.json`<br>`README.md` | - | - | 数据清单与说明 |

## 本次不上传/不纳入数据包

| 路径 | 原因 |
|---|---|
| `nmrsolver/` | NMR-Solver 提交/结果缓存，非原始训练/验证/测试数据 |
| `external_test/sprompt_votes/` | SPrompt 推理投票缓存，复现辅助文件 |
| `_source/` | 旧工作区 symlink，不是自包含数据 |

## 各数据集说明

### 1. `experimental/exp_lit.lmdb`（训练-实验）

- **来源**：从原始 `new_db.lmdb` 中按 `source ∈ {NMR BANK, c13_naproc, bmrb, Spektraris_NMR, GlycoNMR, hmdb}` 筛选。
- **数量**：49,984 条，55 条阳性。
- **用途**：实验谱训练语料。
- **说明**：原始库中 NMRium 上传集合（1,657,910 条）因 provenance 未独立核验，**不纳入训练语料**；但单独的 `validation/` 拆分来自 NMRium，仅用于校准/验证。

### 2. `experimental/ex_train.*`（训练-实验额外）

- **来源**：nmrshiftdb extra train split。
- **数量**：24,520 条，11 条阳性。
- **文件**：
  - `ex_train.json`：原始 JSON。
  - `ex_train.ring.json`：补充 ring-system 字段后的 JSON。
  - `ex_train.lmdb`：由 ring JSON 构建的 LMDB。
- **用途**：额外实验训练数据；与评估集做过 SMILES 去重检查。

### 3. `synthetic/synth_predicted.lmdb`（训练-合成）

- **来源**：原始 `new_db.lmdb` 中 `source ∈ {multi, napacd, acd_lotusv9, pnmrnp2}`。
- **数量**：1,488,432 条，2,461 条阳性。
- **用途**：合成/预测谱训练语料。
- **许可注意**：

| source | 记录数 | 许可状态 |
|---|---:|---|
| `multi` | 789,028 | 旧记录中未记录 licence，发布前需确认 |
| `napacd` | 313,343 | 未记录 |
| `acd_lotusv9` | 194,550 | CC BY 4.0（LOTUS v9 + ACD/Labs 预测 13C，Zenodo 10.5281/zenodo.8175939） |
| `pnmrnp2` | 191,511 | 未记录 |

如不能确认全部来源，可用 `scripts/build_lmdb/filter_lmdb_by_source.py` 构建清理子集，并用 `export_source_index.py` 导出不含谱图的索引。

### 4. `pubchem/pubchem_1209.lmdb`（训练-PubChem 正样本）

- **来源**：PubChem 正样本 `new_data_1209.filtered`，去除 90 条与评估集重叠记录。
- **数量**：77,328 条，77,328 条阳性。
- **用途**：正样本增强训练。

### 5. `validation/validation.*`（验证）

- **来源**：clean `new_dataset` validation split，`source = NMRium`。
- **数量**：40,667 条，37 条阳性。
- **用途**：**仅用于校准/验证**，不用于训练。
- **说明**：这不是 NMRMind validation 的子集；NMRMind validation/test 是另一套上游拆分。

### 6. `external_test/external_test.*`（测试）

- **来源**：April-18 外部测试集。
- **数量**：4,502 条，64 条阳性，4,438 条阴性。
- **文件**：
  - `external_test.json`：完整 JSON。
  - `external_test.lmdb`：LMDB。
  - `external_test_pos_only.json`：仅 64 个阳性。
- **用途**：外部测试/最终评估。
- **说明**：外部阳性不用于阈值选择；τ=0.50 为先验固定。记录 #57 已从 3-ethyl ester 修正为 3-carboxylic acid（15 → 14 peaks），修正已反映在 JSON/LMDB。

### 7. `f_split/`（F-split 评估）

- **数量**：48 + 48，均为阳性。
- **文件**：
  - `raw_data.json`：更多峰（`nmr_type=without_f_split`）。
  - `clean_data.json`：更少峰，C–F 合并后（`nmr_type=with_f_split`）。
- **用途**：评估 F 裂分/峰表处理对模型的影响。

### 8. `sim5/fluoroquinolones_nmr.json`（模拟谱 demo）

- **数量**：5 条，均为阳性氟喹诺酮模拟 13C 谱。
- **用途**：小规模模型对比/demo。

### 9. `nmrmind/`（NMRMind 训练/验证/测试）

- **来源**：NMRMind 上游拆分。
- **数量**：
  - `train.lmdb`：2,137,771 条，3,288 条阳性
  - `validation.lmdb` / `validation.json`：55,167 条，81 条阳性
  - `test.lmdb` / `test.json`：55,150 条，87 条阳性
- **用途**：NMRMind 配置下的训练、验证和测试。
- **说明**：上传前请确认 NMRMind 数据的再分发许可/条款。该拆分与本项目 `validation/`、`external_test/` 是不同的数据集定义，不应混用。

## LMDB schema

每条记录为 pickled dict：

```text
key               int
canonical_smiles  str        (cano_smiles 为别名)
C_nmr             list[float]  13C shifts, ppm
source            str
split             str
label             np.ndarray([0|1])   是否含喹诺酮母核
ecfp              np.ndarray(2048,)   Morgan FP, r=2
maccs             np.ndarray(167,)
element_list      np.ndarray(12,)     C,H,O,N,P,S,Cl,Br,I,B,F,Si counts
function_list     np.ndarray(92,)     reserved (zeros; thermo optional)
ring_system       list[str]           去除末端碳后的环系统
extra_ring_system list[str]           匹配到的喹诺酮母核 SMARTS
ori_key           str                 原始索引（JSON 构建的 split）
```

`function_list` 默认全零；如需启用，安装 `thermo` 并在 `scripts/build_lmdb/json_to_lmdb.py` 中填充。

## 重建命令

```bash
# 1. split the 3.20 M source corpus by source category (~10 min)
python scripts/build_lmdb/split_newdb_by_source.py \
    --src data/_source/new_db.lmdb \
    --exp_out data/experimental/exp_lit.lmdb \
    --synth_out data/synthetic/synth_predicted.lmdb \
    --overwrite

# 2. JSON -> featurised LMDB (validation / external test / ex_train)
python scripts/build_lmdb/json_to_lmdb.py \
    --json data/validation/validation.json \
    --out  data/validation/validation.lmdb --split valid

python scripts/build_lmdb/json_to_lmdb.py \
    --json data/external_test/external_test.json \
    --out  data/external_test/external_test.lmdb --split external_test

python scripts/build_lmdb/json_to_lmdb.py \
    --json data/experimental/ex_train.ring.json \
    --out  data/experimental/ex_train.lmdb --split train
```

`pubchem/pubchem_1209.lmdb` 和 NMRMind LMDB 通常从上游发布复制；其记录已包含 ring systems、ECFP/MACCS 和 label。

## 泄漏控制

- External test、validation、test 与所有训练语料按 exact canonical SMILES 和 stereo-stripped SMILES 去重，0 重叠。
- `pubchem_1209` 已按同样规则过滤（77,418 → 77,328，去除 90 条）。
- 外部正样本不用于阈值选择；τ=0.50 为先验固定。
- External record #57 已从 3-ethyl ester 修正为 3-carboxylic acid（15 → 14 peaks）。
