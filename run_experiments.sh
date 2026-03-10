#!/bin/bash
# =============================================================================
# 🧪 G05 — Pipeline d'Expérimentation AG News / DistilBERT
# Script : run_experiments.sh (VERSION CORRIGÉE)
# 📍 À lancer DEPUIS le dossier : G05_AGNews/
# =============================================================================

# ⚠️ CORRECTION : -n au lieu de -u pour éviter les erreurs sur variables non définies
set -eo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 📦 CONFIGURATION — CHEMINS RELATIFS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT="$(pwd)"
MODE="${MODE:-cpu_mid}"
RUN_TYPE="${RUN_TYPE:-full}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"

# 📁 Chemins relatifs
BACKUP_DIR="./backups"
RESULTS_DIR="./results"
LOGS_DIR="./logs"
NOTEBOOKS_DIR="./notebooks"
SRC_DIR="./src"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
HOSTNAME=$(hostname 2>/dev/null || echo "local")

# ✅ CORRECTION : Initialisation explicite des variables optionnelles
PYTHON_CMD="python3"
USE_RSYNC="false"
REMOTE_DEST=""
CLEANUP_DAYS=""
GENERATE_MATRIX="false"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ─────────────────────────────────────────────────────────────────────────────
# 🛠️ FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

log_info()    { echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') — $*"; }
log_success() { echo -e "${GREEN}[OK]${NC}   $(date '+%H:%M:%S') — $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') — $*"; }
log_error()   { echo -e "${RED}[ERR]${NC}  $(date '+%H:%M:%S') — $*" >&2; }

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

check_dependencies() {
    log_info "Vérification des dépendances..."
    
    # Trouver Python (python3 ou python)
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python non trouvé. Installe Python 3.8+."
        exit 1
    fi
    
    log_info "🐍 Python utilisé : $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
    
    # Vérifier dépendances de base
    for dep in tar cp mkdir find; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep non disponible."
            exit 1
        fi
    done
    
    # rsync optionnel
    if command -v rsync &> /dev/null; then
        log_info "🟢 rsync disponible"
        USE_RSYNC="true"
    else
        log_warn "🟡 rsync non installé — utilisation de cp/tar"
        USE_RSYNC="false"
    fi
    
    log_success "Dépendances vérifiées"
}

setup_directories() {
    log_info "Création de l'arborescence..."
    mkdir -p "$RESULTS_DIR"/{logs,figures,models,checkpoints,notebooks}
    mkdir -p "$LOGS_DIR"
    if [[ "$BACKUP_ENABLED" == "true" ]]; then
        mkdir -p "$BACKUP_DIR"/{configs,results,models}
    fi
    log_success "Répertoires créés"
}

save_config_snapshot() {
    local snapshot_dir="$BACKUP_DIR/configs/$TIMESTAMP"
    mkdir -p "$snapshot_dir"
    
    log_info "Sauvegarde de la configuration..."
    
    [[ -f "$SRC_DIR/config.py" ]] && cp "$SRC_DIR/config.py" "$snapshot_dir/"
    
    cat > "$snapshot_dir/env_snapshot.txt" << EOF
# Environment Snapshot — $TIMESTAMP
PROJECT_ROOT=$PROJECT_ROOT
MODE=$MODE
RUN_TYPE=$RUN_TYPE
BACKUP_ENABLED=$BACKUP_ENABLED
HOSTNAME=$HOSTNAME
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
TRANSFORMERS_VERSION=$($PYTHON_CMD -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "N/A")
CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A")
EOF
    
    [[ -f "$NOTEBOOKS_DIR/G05_full_analysis.ipynb" ]] && \
        cp "$NOTEBOOKS_DIR/G05_full_analysis.ipynb" "$snapshot_dir/"
    
    log_success "Config snapshot → $snapshot_dir"
}

copy_files() {
    local src="$1"
    local dst="$2"
    local pattern="${3:-*}"
    
    mkdir -p "$dst"
    
    if [[ "$USE_RSYNC" == "true" ]]; then
        rsync -av --checksum \
            --include="$pattern" --include='*/' --exclude='*' \
            "$src/" "$dst/" 2>/dev/null || cp -r "$src"/* "$dst/" 2>/dev/null || true
    else
        if [[ "$pattern" == "*" ]]; then
            cp -r "$src"/* "$dst/" 2>/dev/null || true
        else
            find "$src" -type f -name "$pattern" -print0 2>/dev/null | \
            while IFS= read -r -d '' file; do
                local rel_path="${file#$src/}"
                local target_dir="$dst/$(dirname "$rel_path")"
                mkdir -p "$target_dir"
                cp "$file" "$target_dir/" 2>/dev/null || true
            done
        fi
    fi
}

backup_results() {
    local src="$RESULTS_DIR"
    local dst="$BACKUP_DIR/results/$TIMESTAMP"
    
    if [[ ! -d "$src" ]] || [[ -z "$(ls -A "$src" 2>/dev/null)" ]]; then
        log_warn "Aucun résultat à sauvegarder dans $src"
        return
    fi
    
    log_info "Sauvegarde des résultats vers $dst..."
    
    for ext in csv json png pt ipynb; do
        copy_files "$src" "$dst" "*.$ext"
    done
    
    cat > "$dst/MANIFEST.txt" << EOF
Backup Manifest — $TIMESTAMP
Mode: $MODE | Run: $RUN_TYPE | Host: $HOSTNAME
Projet: $PROJECT_ROOT
Méthode: $([ "$USE_RSYNC" == "true" ] && echo "rsync" || echo "cp/tar")
EOF
    
    log_success "Backup terminé : $dst"
}

backup_model() {
    local model_name="${1:-best}"
    local src_dir="$RESULTS_DIR/models"
    local dst_dir="$BACKUP_DIR/models/$TIMESTAMP"
    
    mkdir -p "$dst_dir"
    
    local count=0
    shopt -s nullglob
    for f in "$src_dir"/${model_name}*.pt; do
        local fname=$(basename "$f")
        local fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "0")
        
        if [[ "$fsize" -gt 104857600 ]]; then
            log_info "  🗜️  Compression de $fname (>100Mo)..."
            tar -czf "$dst_dir/${fname}.tar.gz" -C "$(dirname "$f")" "$fname" 2>/dev/null || \
            cp "$f" "$dst_dir/"
        else
            cp "$f" "$dst_dir/"
        fi
        ((count++)) || true
    done
    shopt -u nullglob
    
    if [[ $count -eq 0 ]]; then
        log_warn "Aucun modèle '${model_name}*.pt' trouvé"
    else
        log_success "$count modèle(s) sauvegardé(s) → $dst_dir"
    fi
}



run_notebook_section() {
    local section="${1:-full}"
    local notebook="$NOTEBOOKS_DIR/G05_full_analysis.ipynb"
    
    if [[ ! -f "$notebook" ]]; then
        log_error "Notebook non trouvé : $notebook"
        return 1
    fi
    
    log_info "Exécution du notebook (section: $section, mode: $MODE)..."
    
    # ✅ FORCER PYTHONPATH avant papermill
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export PROJECT_ROOT="$PROJECT_ROOT"
    
    # Afficher pour debug
    log_info "PYTHONPATH=$PYTHONPATH"
    log_info "PROJECT_ROOT=$PROJECT_ROOT"
    log_info "PWD=$(pwd)"
    
    if command -v papermill &> /dev/null; then
        local output_notebook="$RESULTS_DIR/notebooks/G05_executed_${TIMESTAMP}.ipynb"
        mkdir -p "$(dirname "$output_notebook")"
        
        # ✅ Passer PYTHONPATH explicitement à papermill
        PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" \
        papermill "$notebook" "$output_notebook" \
            -p MODE "$MODE" \
            -k python3 \
            --log-output \
            --cwd "$PROJECT_ROOT" \
            2>&1 | tee "$LOGS_DIR/papermill_${TIMESTAMP}.log"
        
        log_success "Notebook exécuté → $output_notebook"
    else
        log_warn "papermill non installé. Exécution fallback..."
        
        $PYTHON_CMD -c "
import nbformat, sys, os
from nbconvert.preprocessors import ExecutePreprocessor

nb_path = os.path.join(os.environ.get('PROJECT_ROOT', '.'), 'notebooks', 'G05_full_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=7200, kernel_name='python3')
try:
    ep.preprocess(nb, {'metadata': {'path': os.path.join(os.environ.get('PROJECT_ROOT', '.'), 'notebooks')}})
    print('✅ Exécution terminée')
except Exception as e:
    print(f'❌ Erreur: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1 | tee "$LOGS_DIR/exec_${TIMESTAMP}.log"
    fi
}




run_random_search() {
    log_info "🚀 Lancement du Random Search (mode=$MODE)..."
    
    local available_gb="unknown"
    if command -v df &> /dev/null; then
        available_gb=$(df -BG "$PROJECT_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "unknown")
    fi
    
    if [[ "$available_gb" != "unknown" ]] && (( available_gb < 10 )); then
        log_warn "⚠️ Espace disque faible : ${available_gb}Go restants"
    fi
    
    local start_time=$(date +%s)
    run_notebook_section "random_search"
    local end_time=$(date +%s)
    local duration=$(( (end_time - start_time) / 60 ))
    
    log_info "⏱️ Durée totale : ${duration} minutes"
    
    if [[ "$BACKUP_ENABLED" == "true" ]]; then
        backup_results
        backup_model "best"
    fi
}

generate_experiment_matrix() {
    log_info "Génération de la matrice d'expériences..."
    local matrix_file="$LOGS_DIR/experiment_matrix_${TIMESTAMP}.csv"
    
    cat > "$matrix_file" << EOF
run_id,mode,optimizer,lr,batch_size,warmup_steps,epochs,timestamp,hostname
EOF
    
    local optimizers=("adamw" "sgd" "adafactor")
    local lrs=("1e-5" "2e-5" "5e-5" "1e-4")
    local batches=("8" "16" "32")
    
    local run_id=1
    for opt in "${optimizers[@]}"; do
        for lr in "${lrs[@]}"; do
            for bs in "${batches[@]}"; do
                echo "$run_id,$MODE,$opt,$lr,$bs,100,3,$TIMESTAMP,$HOSTNAME" >> "$matrix_file"
                ((run_id++)) || true
            done
        done
    done
    
    log_success "Matrice générée : $matrix_file"
}

sync_to_remote() {
    local remote="${1:-}"
    if [[ -z "$remote" ]]; then
        log_warn "Destination distante manquante."
        return
    fi
    
    log_info "Synchronisation vers $remote..."
    
    local tmp_archive="/tmp/g05_backup_${TIMESTAMP}.tar.gz"
    
    tar -czf "$tmp_archive" \
        -C "$PROJECT_ROOT" \
        $(find "$RESULTS_DIR" -type f \( -name "*.csv" -o -name "*.json" -o -name "*.png" -o -name "best_*.pt" \) -printf '%P\n' 2>/dev/null || true) \
        2>/dev/null || true
    
    if [[ -f "$tmp_archive" ]]; then
        if command -v scp &> /dev/null; then
            scp "$tmp_archive" "$remote/" 2>&1 | tee -a "$LOGS_DIR/sync_${TIMESTAMP}.log"
            log_success "Archive transférée → $remote"
        else
            log_warn "scp non disponible. Archive locale : $tmp_archive"
        fi
        rm -f "$tmp_archive"
    else
        log_warn "Aucun fichier à synchroniser"
    fi
}

cleanup_old_backups() {
    local keep_days="${1:-7}"
    log_info "Nettoyage des backups de plus de $keep_days jours..."
    find "$BACKUP_DIR" -type d -mtime +$keep_days -exec rm -rf {} + 2>/dev/null || true
    log_success "Nettoyage effectué"
}

show_help() {
    cat << EOF
🧪 G05 — Script d'Expérimentation (VERSION CORRIGÉE)
📍 À lancer DEPUIS : G05_AGNews/

USAGE: $0 [OPTIONS]

OPTIONS:
  --mode <cpu_low|cpu_mid|gpu>    Mode d'exécution (défaut: cpu_mid)
  --run <setup|data|model|random_search|full>
  --backup <true|false>           (défaut: true)
  --backup-dir <path>             (défaut: ./backups)
  --results-dir <path>            (défaut: ./results)
  --sync <user@host:/path>        Sync vers remote
  --cleanup <days>                Supprimer backups > N jours
  --matrix                        Générer matrice d'expériences
  --help                          Afficher cette aide

EXEMPLES:
  $0 --mode cpu_mid --run full
  $0 --mode gpu --run random_search
  $0 --matrix
  $0 --cleanup 30
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# 🎯 PARSEUR D'ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --run) RUN_TYPE="$2"; shift 2 ;;
        --backup) BACKUP_ENABLED="$2"; shift 2 ;;
        --backup-dir) BACKUP_DIR="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --sync) REMOTE_DEST="$2"; shift 2 ;;
        --cleanup) CLEANUP_DAYS="$2"; shift 2 ;;
        --matrix) GENERATE_MATRIX="true"; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) log_error "Option inconnue: $1"; show_help; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 MAIN
# ─────────────────────────────────────────────────────────────────────────────

main() {
    print_header "G05 — Pipeline d'Expérimentation"
    
    # Vérifier structure
    if [[ ! -d "./notebooks" ]] || [[ ! -d "./src" ]]; then
        log_warn "⚠️ Structure non détectée dans $(pwd)"
        log_info "Lance ce script DEPUIS G05_AGNews/"
    fi
    
    log_info "📁 Projet : $PROJECT_ROOT"
    log_info "⚙️ Mode : $MODE | Run : $RUN_TYPE"
    
    check_dependencies
    setup_directories
    
    [[ "$BACKUP_ENABLED" == "true" ]] && save_config_snapshot
    [[ "$GENERATE_MATRIX" == "true" ]] && generate_experiment_matrix
    
    case $RUN_TYPE in
        setup) log_info "✅ Setup uniquement" ;;
        data) log_info "📦 Chargement données..."; run_notebook_section "data" ;;
        model) log_info "🤖 Chargement modèle..."; run_notebook_section "model" ;;
        random_search) run_random_search ;;
        full|*) log_info "🔄 Exécution complète..."; run_random_search ;;
    esac
    
    if [[ "$BACKUP_ENABLED" == "true" ]]; then
        backup_results
        backup_model "best"
    fi
    
    [[ -n "$REMOTE_DEST" ]] && sync_to_remote "$REMOTE_DEST"
    [[ -n "$CLEANUP_DAYS" ]] && cleanup_old_backups "$CLEANUP_DAYS"
    
    print_header "✅ Pipeline Terminé"
    log_success "📁 Résultats : $RESULTS_DIR"
    log_success "📁 Logs : $LOGS_DIR"
    [[ "$BACKUP_ENABLED" == "true" ]] && log_success "💾 Backups : $BACKUP_DIR/$TIMESTAMP"
    
    if [[ -d "$RESULTS_DIR/figures" ]]; then
        local fig_count=$(find "$RESULTS_DIR/figures" -name "*.png" 2>/dev/null | wc -l)
        log_info "📊 Figures générées : $fig_count"
    fi
}

main "$@"