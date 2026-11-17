#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Regenerate the paper / SI figures (CPU only, ~5 min total).
#
#   bash scripts/plot/make_all_figures.sh
#
# Outputs go to `reports/` (all figure scripts resolve paths relative to the
# repository root).  Inputs are the pre-computed artefacts shipped in
# `reports/calib_data/`, `reports/*.json`, `viz_tmp/*` and `results/predictions/`.
#
# Requires: matplotlib, numpy, scipy, scikit-learn, rdkit, pillow, joblib,
#           xgboost (used for the Figure 2 / PR-ROC / FN#12 panels).
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-python}
export NMR_ROOT=${NMR_ROOT:-$(pwd)}

FIG_SCRIPTS=(
    make_figure_model_analysis_v5.py     # Figure 2 (A-H): embedding, FN#12, CI, ring systems
    make_ci_external_figure.py           # calibrated p + 90% bootstrap CI (forest plot)
    make_xgb_pr_roc.py                   # PR / ROC curves with 95% bootstrap bands
    make_hit_rate_figure.py              # top-10 hit rate / exact@10 dumbbell
    make_3d_hit_chart.py                 # 3-D hit-rate chart
    make_nhetero_systems_figure.py       # accuracy per N-hetero ring system
    make_mc_dropout_figure.py            # MC-dropout uncertainty
    make_p_distribution_figure.py        # per-molecule probability distribution
    make_uncertainty_figure.py           # 3-seed spread + conformal prediction sets
    make_fp_structures_figure.py         # structures of the four false positives
    make_fn12_case_figure.py             # FN #12 case study
    make_fn12_local_figure.py            # FN #12 local decision boundary
    make_paper_figure_fn12.py            # FN #12 composite (paper layout)
    make_r2_qualitative_figure.py        # qualitative model comparison (6 families)
)

fail=0
for s in "${FIG_SCRIPTS[@]}"; do
    if [ -f "reports/$s" ]; then
        printf '\n=== %s\n' "$s"
        "$PY" "reports/$s" || { echo "!! $s failed"; fail=1; }
    fi
done

printf '\nfigures written to: %s/reports\n' "$(pwd)"
[ "$fail" -eq 0 ] || echo "some figures failed (see log above)"
exit "$fail"
