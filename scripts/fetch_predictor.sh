#!/usr/bin/env bash
# Copy the optional nmrshiftdb2 predictor jars into lib/.
#
# The jars are third-party code (AGPL-3.0 + nmrshiftdb2 Database License) and
# are deliberately not tracked by git.  This script only copies them from a
# local checkout; it does not download anything.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGACY_LIB="${LEGACY_LIB:-/mnt/data2/NMR_Core_Mol_Detection/lib}"
DEST="$ROOT/lib"

mkdir -p "$DEST"

declare -A SHA=(
  [cdk-2.3.jar]="bf60002d40b88136f185a96b1f38135e240733360c6b684cfc2bfd64972eb04a"
  [predictorc.jar]="95211fcaf148600057743b657afdaced6cc352cdb3dbf242992b0cd2ff425419"
  [predictorh.jar]="5e29217c7792b0b3651e5520d076e61df7dec3d29f1ec6fdfbf13f48e0cb3156"
)

if [ ! -d "$LEGACY_LIB" ]; then
  cat >&2 <<EOF
legacy lib directory not found: $LEGACY_LIB

Place predictorc.jar, predictorh.jar and cdk-2.3.jar in $DEST manually, or run:
  LEGACY_LIB=/path/to/NMR_Core_Mol_Detection/lib bash scripts/fetch_predictor.sh

The jars are AGPL-3.0 + nmrshiftdb2 Database License; see lib/README.md.
EOF
  exit 1
fi

for jar in predictorc.jar predictorh.jar cdk-2.3.jar; do
  src="$LEGACY_LIB/$jar"
  if [ ! -f "$src" ]; then
    echo "missing: $src" >&2
    exit 1
  fi
  cp -f "$src" "$DEST/$jar"
  got="$(sha256sum "$DEST/$jar" | awk '{print $1}')"
  if [ "$got" != "${SHA[$jar]}" ]; then
    echo "checksum mismatch for $jar: $got" >&2
    exit 1
  fi
  echo "ok $jar"
done

echo "jars copied to $DEST (git-ignored)"
