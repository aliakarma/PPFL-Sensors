#!/usr/bin/env bash
# =============================================================================
# scripts/reproduce_all.sh
# =============================================================================
# Single-command full experiment reproduction.
#
# Runs all four experimental conditions, then calls evaluate.py to generate
# comparison plots and summary table.
#
# Usage
# -----
#   bash scripts/reproduce_all.sh              # full run
#   bash scripts/reproduce_all.sh --dry-run    # validate only (no training)
#   bash scripts/reproduce_all.sh --fast-dev   # 3 rounds, 3 clients (quick smoke)
#
# Environment overrides
# ---------------------
#   SEED=123 bash scripts/reproduce_all.sh
#   N_ROUNDS=10 N_CLIENTS=5 bash scripts/reproduce_all.sh
#
# Exit codes
# ----------
#   0  — all experiments completed successfully
#   1  — one or more experiments failed
# =============================================================================

set -euo pipefail

# ── Defaults (overridable via env) ────────────────────────────────
SEED="${SEED:-42}"
N_ROUNDS="${N_ROUNDS:-20}"
N_CLIENTS="${N_CLIENTS:-10}"
RESULTS_DIR="${RESULTS_DIR:-results/logs/}"
PLOT_DIR="${PLOT_DIR:-results/plots/}"

DRY_RUN=false
FAST_DEV=false

# ── Argument parsing ─────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN=true ;;
        --fast-dev)  FAST_DEV=true ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--dry-run] [--fast-dev]"
            exit 1
            ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="results/logs/reproduce_${TIMESTAMP}.log"
FAILED_EXPERIMENTS=()

mkdir -p results/logs results/plots

# ── Step 0: Environment check ─────────────────────────────────────
info "=== Step 0: Environment check ==="

PYTHON_MIN="3.8"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
info "Python version: $python_version"

# Check required packages
check_package() {
    python3 -c "import $1" 2>/dev/null && success "$1 available" || {
        error "$1 not found — run: pip install -r requirements.txt"
        exit 1
    }
}
check_package torch
check_package sklearn
check_package yaml
check_package numpy

success "All required packages available."

# ── Step 1: Dataset preparation ───────────────────────────────────
info "=== Step 1: Dataset check ==="

if [ -d "data/raw/UCI HAR Dataset" ]; then
    success "UCI HAR dataset found."
else
    warn "UCI HAR dataset not found at data/raw/UCI HAR Dataset/"
    warn "Falling back to synthetic data (set dataset.name: synthetic in config)."
fi

# ── Dry-run shortcut ─────────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    info "=== Dry-run mode: validating configs only ==="
    for cfg in configs/default.yaml configs/attack_only.yaml configs/defense_ablation.yaml; do
        python3 main.py --config "$cfg" --dry-run && success "$cfg OK" || {
            error "$cfg failed validation"
            exit 1
        }
    done
    success "All configs valid. Dry-run complete."
    exit 0
fi

# ── Fast-dev flag ─────────────────────────────────────────────────
FAST_DEV_FLAG=""
if [ "$FAST_DEV" = true ]; then
    FAST_DEV_FLAG="--fast-dev"
    warn "fast-dev mode: using 3 clients, 3 rounds, tiny model"
fi

# ── Runner helper ─────────────────────────────────────────────────
run_experiment() {
    local label="$1"; shift
    local cmd="$@"
    info "--- $label ---"
    echo "[$(date)] $label: $cmd" >> "$LOG_FILE"
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        success "$label completed."
    else
        error "$label FAILED. See $LOG_FILE"
        FAILED_EXPERIMENTS+=("$label")
    fi
}

# ── Step 2: Experiment 1 — Baseline (no defense) ─────────────────
info "=== Step 2: Experiment 1 — Baseline FL (no defense) ==="
run_experiment "baseline" \
    python3 main.py \
        --config configs/attack_only.yaml \
        --set logging.seed "$SEED" \
        --set dataset.n_clients "$N_CLIENTS" \
        --set training.rounds "$N_ROUNDS" \
        $FAST_DEV_FLAG

# ── Step 3: Experiment 2 — Gaussian noise ablation ───────────────
info "=== Step 3: Experiment 2 — Gaussian Noise Ablation ==="
for sigma in 0.001 0.01 0.05 0.1 0.5; do
    if [ "$FAST_DEV" = true ] && [ "$(python3 -c "print(1 if float('$sigma') > 0.01 else 0)")" = "1" ]; then
        continue  # fast-dev: only one sigma value
    fi
    run_experiment "noise_sigma_${sigma}" \
        python3 main.py \
            --config configs/defense_ablation.yaml \
            --set logging.seed "$SEED" \
            --set dataset.n_clients "$N_CLIENTS" \
            --set training.rounds "$N_ROUNDS" \
            --set defense.noise.enabled true \
            --set defense.noise.sigma "$sigma" \
            --set defense.clipping.enabled false \
            $FAST_DEV_FLAG
done

# ── Step 4: Experiment 3 — Gradient clipping ablation ────────────
info "=== Step 4: Experiment 3 — Gradient Clipping Ablation ==="
for max_norm in 0.1 0.5 1.0 2.0 5.0; do
    if [ "$FAST_DEV" = true ] && [ "$(python3 -c "print(1 if float('$max_norm') > 0.5 else 0)")" = "1" ]; then
        continue  # fast-dev: only one clipping value
    fi
    run_experiment "clipping_norm_${max_norm}" \
        python3 main.py \
            --config configs/defense_ablation.yaml \
            --set logging.seed "$SEED" \
            --set dataset.n_clients "$N_CLIENTS" \
            --set training.rounds "$N_ROUNDS" \
            --set defense.noise.enabled false \
            --set defense.clipping.enabled true \
            --set defense.clipping.max_norm "$max_norm" \
            $FAST_DEV_FLAG
done

# ── Step 5: Experiment 4 — Combined defense ───────────────────────
info "=== Step 5: Experiment 4 — Combined (noise + clipping) ==="
run_experiment "combined_defense" \
    python3 main.py \
        --config configs/defense_ablation.yaml \
        --set logging.seed "$SEED" \
        --set dataset.n_clients "$N_CLIENTS" \
        --set training.rounds "$N_ROUNDS" \
        --set defense.noise.enabled true \
        --set defense.noise.sigma 0.01 \
        --set defense.clipping.enabled true \
        --set defense.clipping.max_norm 1.0 \
        $FAST_DEV_FLAG

# ── Step 6: Experiment 5 — Ensemble FL ───────────────────────────
info "=== Step 6: Experiment 5 — Ensemble FL ==="
run_experiment "ensemble_fl" \
    python3 main.py \
        --config configs/default.yaml \
        --set logging.seed "$SEED" \
        --set dataset.n_clients "$N_CLIENTS" \
        --set training.rounds "$N_ROUNDS" \
        --set training.aggregation ensemble \
        $FAST_DEV_FLAG

# ── Step 7: Non-IID experiments ───────────────────────────────────
if [ "$FAST_DEV" = false ]; then
    info "=== Step 7: Non-IID partition experiments ==="
    for strategy in dirichlet pathological; do
        run_experiment "noniid_${strategy}" \
            python3 main.py \
                --config configs/default.yaml \
                --set logging.seed "$SEED" \
                --set dataset.n_clients "$N_CLIENTS" \
                --set training.rounds "$N_ROUNDS" \
                --set dataset.partition_strategy "$strategy" \
                $FAST_DEV_FLAG
    done
fi

# ── Step 8: Aggregate + visualise ────────────────────────────────
info "=== Step 8: Aggregating results and generating plots ==="
python3 experiments/evaluate.py \
    --results-dir "$RESULTS_DIR" \
    --plot-dir "$PLOT_DIR" >> "$LOG_FILE" 2>&1 && \
    success "Evaluation complete. Plots saved to $PLOT_DIR" || \
    warn "evaluate.py failed — results still in $RESULTS_DIR"

# ── Final report ─────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    success "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
    echo ""
    echo "  Results : $RESULTS_DIR"
    echo "  Plots   : $PLOT_DIR"
    echo "  Log     : $LOG_FILE"
    echo "═══════════════════════════════════════════════════════"
    exit 0
else
    error "THE FOLLOWING EXPERIMENTS FAILED:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "    ✗  $exp"
    done
    echo ""
    echo "  Full log: $LOG_FILE"
    echo "═══════════════════════════════════════════════════════"
    exit 1
fi
