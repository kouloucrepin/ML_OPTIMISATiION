"""
random_search.py — Moteur de Random Search pour P01 (Benchmark d'Optimiseurs)
G05 | P01

Références :
- Bergstra & Bengio (2012) "Random Search for Hyper-Parameter Optimization"
  JMLR → Random Search est aussi efficace que Grid Search avec ~5x moins de trials
- Li et al. (2017) "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
  → Pruning des trials peu prometteurs (implémenté via early stopping)
- Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization"
  → Justification du choix AdamW comme baseline
"""

import sys
import os
import json
import argparse
import random
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import Config, get_config
from src.data_loader import prepare_data, get_dataloader, set_seed
from src.model_setup import load_fresh_model, build_optimizer, build_scheduler
from optimiseurs.train import train_one_run, evaluate


# ─────────────────────────────────────────────
# HYPERPARAMÈTRES D'UN TRIAL
# ─────────────────────────────────────────────

@dataclass
class TrialConfig:
    """Hyperparamètres pour un trial donné."""
    run_id:       str
    optimizer:    str
    lr:           float
    batch_size:   int
    warmup_steps: int
    num_epochs:   int
    momentum:     float   # SGD uniquement
    weight_decay: float


# ─────────────────────────────────────────────
# ÉCHANTILLONNAGE ALÉATOIRE
# ─────────────────────────────────────────────

def sample_hyperparams(cfg: Config, run_idx: int, forced_optimizer: Optional[str] = None) -> TrialConfig:
    """
    Tire aléatoirement un jeu d'hyperparamètres.

    Bergstra & Bengio (2012) montrent que le LR est l'hyperparamètre
    le plus important → on utilise une échelle log-uniforme pour le LR
    afin de couvrir plusieurs ordres de grandeur de façon homogène.
    """
    sc = cfg.search

    # LR — échelle log (couvre [1e-6, 5e-4] uniformément en log)
    log_lr = np.random.uniform(np.log(sc.lr_min), np.log(sc.lr_max))
    lr = float(np.exp(log_lr))

    # Optimiseur — si forced, on l'impose (pour garantir couverture minimale)
    optimizer = forced_optimizer if forced_optimizer else random.choice(sc.optimizers)

    # Autres hyperparamètres — tirages uniformes discrets
    batch_size   = random.choice(sc.batch_sizes)
    warmup_steps = random.choice(sc.warmup_options)
    num_epochs   = random.randint(sc.epoch_min, sc.epoch_max)
    momentum     = random.choice(sc.momentum_options)
    weight_decay = float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2))))

    return TrialConfig(
        run_id       = f"run_{run_idx:03d}",
        optimizer    = optimizer,
        lr           = round(lr, 8),
        batch_size   = batch_size,
        warmup_steps = warmup_steps,
        num_epochs   = num_epochs,
        momentum     = momentum,
        weight_decay = round(weight_decay, 6),
    )


def generate_trial_list(cfg: Config) -> List[TrialConfig]:
    """
    Génère la liste des trials en garantissant la couverture minimale
    par optimiseur (min_trials_per_optimizer = 3, protocole P01).

    Stratégie :
    1. Réserver 3 trials par optimiseur = 9 trials garantis
    2. Le reste est tiré aléatoirement

    Cela respecte le protocole P01 qui exige ≥3 LR testés par optimiseur.
    """
    trials = []
    idx = 0

    # Phase 1 : couverture garantie
    sc = cfg.search
    for opt in sc.optimizers:
        for _ in range(sc.min_trials_per_optimizer):
            t = sample_hyperparams(cfg, idx, forced_optimizer=opt)
            trials.append(t)
            idx += 1

    # Phase 2 : trials aléatoires supplémentaires
    remaining = sc.n_trials - len(trials)
    for _ in range(max(0, remaining)):
        t = sample_hyperparams(cfg, idx)
        trials.append(t)
        idx += 1

    # Mélange pour ne pas avoir les optimiseurs en blocs
    random.shuffle(trials)
    for i, t in enumerate(trials):
        t.run_id = f"run_{i:03d}"

    return trials


# ─────────────────────────────────────────────
# EXÉCUTION D'UN TRIAL
# ─────────────────────────────────────────────

def run_trial(
    trial: TrialConfig,
    cfg: Config,
    train_dataset,
    val_dataset,
    verbose: bool = True,
) -> Dict:
    """
    Exécute un trial complet : init modèle → entraînement → évaluation.

    Chaque trial charge un nouveau modèle depuis les poids pré-entraînés.
    Cela évite toute contamination entre les runs (Dodge et al., 2020).
    """
    set_seed(cfg.project.seed + hash(trial.run_id) % 1000)
    device = torch.device(cfg.project.device)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Trial {trial.run_id} | optimizer={trial.optimizer} | lr={trial.lr:.2e}")
        print(f"  batch={trial.batch_size} | warmup={trial.warmup_steps} | epochs={trial.num_epochs}")
        print(f"{'='*55}")

    # Chargement modèle frais
    model = load_fresh_model(cfg)

    # DataLoaders
    train_loader = get_dataloader(train_dataset, batch_size=trial.batch_size, shuffle=True)
    val_loader   = get_dataloader(val_dataset,   batch_size=16,               shuffle=False)

    # Optimiseur
    optimizer = build_optimizer(
        model,
        optimizer_name=trial.optimizer,
        lr=trial.lr,
        weight_decay=trial.weight_decay,
        momentum=trial.momentum,
    )

    # Scheduler — total_steps = max_steps du cfg
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=min(trial.warmup_steps, cfg.training.max_steps // 4),
        total_steps=cfg.training.max_steps,
    )

    # Entraînement
    result = train_one_run(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_id=trial.run_id,
        verbose=verbose,
    )

    # Enrichissement avec les hyperparamètres du trial
    result.update(asdict(trial))

    # Sauvegarde du modèle si meilleur résultat
    return result, model


# ─────────────────────────────────────────────
# RANDOM SEARCH PRINCIPAL
# ─────────────────────────────────────────────

def run_random_search(cfg: Config, test_run: bool = False, verbose: bool = True) -> List[Dict]:
    """
    Lance le Random Search complet.

    Args:
        cfg      : configuration du projet
        test_run : si True, ne lance qu'1 trial (validation du pipeline)
        verbose  : affichage détaillé

    Returns:
        Liste des résultats triés par val_f1 décroissant
    """
    set_seed(cfg.project.seed)

    print("\n[SEARCH] Préparation des données...")
    train_dataset, val_dataset, test_dataset = prepare_data(cfg)

    # Génération des trials
    trials = generate_trial_list(cfg)
    if test_run:
        trials = trials[:1]
        print("[SEARCH] Mode test : 1 trial seulement")

    print(f"\n[SEARCH] {len(trials)} trials à exécuter")
    print(f"[SEARCH] Optimiseurs : {cfg.search.optimizers}")

    # Affichage du plan
    from collections import Counter
    opt_counts = Counter(t.optimizer for t in trials)
    for opt, count in opt_counts.items():
        print(f"         {opt:10s} → {count} trials")

    # CSV d'enregistrement des résultats
    csv_path = cfg.search.results_csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    csv_columns = [
        "run_id", "optimizer", "lr", "batch_size", "warmup_steps", "num_epochs",
        "momentum", "weight_decay", "final_val_accuracy", "final_val_f1",
        "final_val_loss", "best_val_f1", "train_time_min", "steps_done"
    ]

    all_results = []
    best_f1_so_far = 0.0
    best_models = {}  # optimizer → model

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()

        for i, trial in enumerate(trials):
            print(f"\n[SEARCH] Trial {i+1}/{len(trials)}")

            try:
                result, model = run_trial(trial, cfg, train_dataset, val_dataset, verbose=verbose)

                # Écriture immédiate dans le CSV (reprise possible si crash)
                writer.writerow({k: result.get(k, "") for k in csv_columns})
                csvfile.flush()

                all_results.append(result)

                # Tracking du meilleur par optimiseur
                opt = trial.optimizer
                if opt not in best_models or result["final_val_f1"] > best_models[opt]["f1"]:
                    best_models[opt] = {
                        "f1":    result["final_val_f1"],
                        "model": model,
                        "run_id": trial.run_id,
                    }
                    # Sauvegarde du meilleur modèle par optimiseur
                    model_path = os.path.join(cfg.project.models_dir, f"best_{opt}.pt")
                    torch.save(model.state_dict(), model_path)
                    print(f"  [SEARCH] Nouveau meilleur {opt} → F1={result['final_val_f1']:.4f} — sauvegardé")

                if result["final_val_f1"] > best_f1_so_far:
                    best_f1_so_far = result["final_val_f1"]
                    torch.save(model.state_dict(), os.path.join(cfg.project.models_dir, "best_overall.pt"))

            except Exception as e:
                print(f"  [ERREUR] Trial {trial.run_id} échoué : {e}")
                continue

    # Tri des résultats
    all_results.sort(key=lambda x: x.get("final_val_f1", 0), reverse=True)

    # Sauvegarde JSON complète (avec historiques de training)
    json_path = os.path.join(cfg.project.logs_dir, "all_results.json")
    with open(json_path, "w") as f:
        # On ne sérialise pas les états du modèle, juste les métriques
        clean_results = []
        for r in all_results:
            cr = {k: v for k, v in r.items() if k not in ("train_loss_history", "val_metrics_history")}
            clean_results.append(cr)
        json.dump(clean_results, f, indent=2)

    # Historiques de training pour les visualisations
    histories_path = os.path.join(cfg.project.logs_dir, "training_histories.json")
    with open(histories_path, "w") as f:
        histories = [
            {
                "run_id": r["run_id"],
                "optimizer": r["optimizer"],
                "lr": r["lr"],
                "train_loss_history": r.get("train_loss_history", []),
                "val_metrics_history": r.get("val_metrics_history", []),
            }
            for r in all_results
        ]
        json.dump(histories, f, indent=2)

    print(f"\n[SEARCH] ✓ Résultats sauvegardés → {csv_path}")
    print(f"[SEARCH] ✓ JSON complet → {json_path}")

    # Résumé final
    print("\n" + "="*55)
    print("  RÉSUMÉ RANDOM SEARCH")
    print("="*55)
    from collections import defaultdict
    by_opt = defaultdict(list)
    for r in all_results:
        by_opt[r["optimizer"]].append(r["final_val_f1"])

    for opt, f1s in by_opt.items():
        print(f"  {opt:10s} | best={max(f1s):.4f} | mean={np.mean(f1s):.4f} | n={len(f1s)}")

    print(f"\n  MEILLEUR GLOBAL : {all_results[0]['optimizer']} | F1={all_results[0]['final_val_f1']:.4f}")
    print("="*55)

    return all_results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Search — G05 P01")
    parser.add_argument("--mode",       type=str, default="cpu_mid",
                        choices=["cpu_low", "cpu_mid", "gpu"],
                        help="Mode matériel : cpu_low | cpu_mid | gpu")
    parser.add_argument("--n_trials",   type=int, default=None,
                        help="Nombre de trials (override la config)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Graine aléatoire")
    parser.add_argument("--test_run",   action="store_true",
                        help="Lance uniquement 1 trial (validation pipeline)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Dossier de sortie")
    parser.add_argument("--verbose",    action="store_true", default=True,
                        help="Affichage détaillé")
    parser.add_argument("--quiet",      action="store_true",
                        help="Affichage minimal")

    args = parser.parse_args()

    cfg = get_config(args.mode)
    cfg.project.seed = args.seed

    if args.n_trials is not None:
        cfg.search.n_trials = args.n_trials

    if args.quiet:
        args.verbose = False

    results = run_random_search(cfg, test_run=args.test_run, verbose=not args.quiet)
