#!/usr/bin/env python
# 从 new_db.lmdb (1029_ring 版) 中提取实验数据源, 生成新 lmdb.
# 排除计算/预测源: multi, napacd, acd_lotusv9, pnmrnp2;
# 排除 NMRium (15495052 upload 数据集, 213.8万条, 出处不明, 用户确认不要).
# 保留: NMR BANK/c13_naproc/bmrb/Spektraris/GlycoNMR/hmdb (纯文献实验库, 约5万条).
# value 原样字节拷贝 (保留 ring_system/ecfp/maccs 等全部字段), key 重新顺序编号.
# 结束时打印全量 source 精确计数, 供核对排除/保留清单是否正确.
#
# 注意: 源库 pickle 为 numpy 2.x 写出, 本机 py312 (numpy 1.26.4) 读需 numpy._core 别名 shim.
# 用法: tmux new-session -d -s extract_exp 'python data_process/extract_experimental_newdb_1024.py > logs/extract_experimental_1024.log 2>&1'

import os
import sys
import time
import pickle
import lmdb

import numpy as _np
import numpy.core as _nc
sys.modules['numpy._core'] = _nc
sys.modules['numpy._core.numeric'] = _nc.numeric
sys.modules['numpy._core.multiarray'] = _nc.multiarray

SRC_LMDB = 'data/NMR_other_data_1020_add_ring_system_1029/new_db.lmdb'
DST_LMDB = '/mnt/data1/nmr_data/newdb_experimental_lit_1024/new_db_exp_lit.lmdb'
EXCLUDE_SOURCES = {'multi', 'napacd', 'acd_lotusv9', 'pnmrnp2', 'NMRium'}
COMMIT_EVERY = 100_000
REPORT_EVERY = 200_000


def main():
    os.makedirs(os.path.dirname(DST_LMDB), exist_ok=True)
    src_env = lmdb.open(SRC_LMDB, readonly=True, lock=False, subdir=True,
                        readahead=False, meminit=False)
    dst_env = lmdb.open(DST_LMDB, map_size=80 * (1024 ** 3), subdir=True)

    src_cnt = {}
    kept = 0
    t0 = time.time()

    with src_env.begin() as src_txn:
        total = src_txn.stat()['entries']
        print(f'源库条数: {total}', flush=True)
        dst_txn = dst_env.begin(write=True)
        for k, v in src_txn.cursor():
            d = pickle.loads(v)
            s = d.get('source', '<none>') if isinstance(d, dict) else '<notdict>'
            src_cnt[s] = src_cnt.get(s, 0) + 1
            if s not in EXCLUDE_SOURCES:
                dst_txn.put(str(kept).encode(), v)
                kept += 1
                if kept % COMMIT_EVERY == 0:
                    dst_txn.commit()
                    dst_txn = dst_env.begin(write=True)
            done = sum(src_cnt.values())
            if done % REPORT_EVERY == 0:
                rate = done / (time.time() - t0)
                eta = (total - done) / max(rate, 1) / 60
                print(f'[{done}/{total}] 已保留 {kept}  {rate:.0f} 条/s  ETA {eta:.0f} min', flush=True)
        dst_txn.commit()

    print('\n===== 全量 source 精确计数 =====', flush=True)
    for s, c in sorted(src_cnt.items(), key=lambda x: -x[1]):
        tag = '排除(计算/预测)' if s in EXCLUDE_SOURCES else '保留(实验)'
        print(f'{s}: {c} ({c / total:.2%})  [{tag}]', flush=True)
    print(f'\n总条数: {total}  保留: {kept} ({kept / total:.2%})  排除: {total - kept}', flush=True)
    print(f'耗时: {(time.time() - t0) / 60:.1f} min', flush=True)

    n_check = dst_env.begin().stat()['entries']
    print(f'复核: 新库实际条数 = {n_check} (期望 {kept})', flush=True)
    src_env.close()
    dst_env.close()
    print('完成:', DST_LMDB, flush=True)


if __name__ == '__main__':
    main()
