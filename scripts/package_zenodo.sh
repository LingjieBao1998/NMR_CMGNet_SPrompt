#!/usr/bin/env bash
# Build the Zenodo upload bundles described in docs/zenodo_upload.md.
#
#   bash scripts/package_zenodo.sh                 # code archive only
#   INCLUDE_WEIGHTS=1 bash scripts/package_zenodo.sh
#   INCLUDE_DATA=1    bash scripts/package_zenodo.sh   # ~33 GB, no compression
#
# Output: dist/zenodo/<name>.tar.gz + dist/zenodo/SHA256SUMS
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-v1.0.0}"
OUT="${OUT:-$ROOT/dist/zenodo}"
INCLUDE_WEIGHTS="${INCLUDE_WEIGHTS:-0}"
INCLUDE_DATA="${INCLUDE_DATA:-0}"
# Override when packaging the filtered release corpus:
#   SYNTH_LMDB=data/synthetic/synth_predicted_release.lmdb
SYNTH_LMDB="${SYNTH_LMDB:-data/synthetic/synth_predicted.lmdb}"

mkdir -p "$OUT"
cd "$ROOT"

# Deterministic tar metadata (GNU tar)
TAR=(tar --sort=name --mtime="UTC 2026-01-01" --owner=0 --group=0 --numeric-owner)

echo "==> code archive (git-tracked files only)"
git archive --format=tar.gz --prefix="nmr-cmgnet-sprompt-${VERSION}/" \
    -o "$OUT/nmr-cmgnet-sprompt-${VERSION}-code.tar.gz" HEAD

if [[ "$INCLUDE_WEIGHTS" == "1" ]]; then
    echo "==> weights archive (~4.1 GB)"
    "${TAR[@]}" -czf "$OUT/nmr-cmgnet-sprompt-${VERSION}-weights.tar.gz" \
        checkpoints/sprompt_ex1209_1ep_seed0.ckpt \
        checkpoints/sprompt_ex1209_1ep_seed2006.ckpt \
        checkpoints/sprompt_ex1209_5ep_seed42.ckpt \
        checkpoints/train_cmgnet_ring_system_3.2_1029_v2_best.ckpt \
        checkpoints/bart_base_0910 \
        checkpoints/cress \
        models/xgboost
fi

if [[ "$INCLUDE_DATA" == "1" ]]; then
    echo "==> data archive (~33 GB, uncompressed; this takes a while)"
    "${TAR[@]}" -cf "$OUT/nmr-cmgnet-sprompt-${VERSION}-data.tar" \
        --exclude='data/_source' \
        data/experimental/exp_lit.lmdb \
        data/experimental/ex_train.lmdb \
        data/experimental/ex_train.json \
        data/experimental/ex_train.ring.json \
        "$SYNTH_LMDB" \
        data/pubchem/pubchem_1209.lmdb \
        data/validation/validation.lmdb \
        data/validation/validation.json \
        data/external_test/external_test.lmdb \
        data/external_test/external_test.json \
        data/external_test/external_test_pos_only.json \
        data/nmrmind/train.lmdb \
        data/nmrmind/validation.lmdb \
        data/nmrmind/test.lmdb \
        data/nmrmind/validation.json \
        data/nmrmind/test.json \
        data/f_split/with_f_split.json \
        data/f_split/without_f_split.json \
        data/manifest.json \
        data/README.md
fi

echo "==> checksums"
( cd "$OUT" && find . -maxdepth 1 \( -name '*.tar' -o -name '*.tar.gz' \) -printf '%f\n' \
    | sort | xargs -r sha256sum > SHA256SUMS )

echo
ls -lh "$OUT"
echo
cat "$OUT/SHA256SUMS"
