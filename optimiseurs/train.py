"""
train.py — Boucle d'entraînement et d'évaluation
G05 | P01

Références :
- Wolf et al. (2020) "HuggingFace Transformers: State-of-the-art NLP"
- Dodge et al. (2020) "Fine-Tuning Pretrained Language Models: Weight
  Initializations, Data Orders, and Early Stopping"
  → Early stopping et gradient clipping sont essentiels pour la stabilité
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import Config


# ─────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calcule accuracy et F1 macro.

    F1 macro : moyenne non pondérée du F1 par classe.
    Adapté aux datasets équilibrés comme notre sous-ensemble AG News.
    Référence : Manning et al. (2008) 'Introduction to Information Retrieval'
    """
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": round(float(accuracy), 4), "f1": round(float(f1), 4)}


# ─────────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────────

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Évalue le modèle sur un DataLoader.
    Retourne accuracy, f1, et loss moyenne.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            n_batches += 1

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    metrics["loss"] = round(total_loss / max(n_batches, 1), 4)
    return metrics


# ─────────────────────────────────────────────
# BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train_one_run(
    model: nn.Module,
    optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    run_id: str = "run_000",
    verbose: bool = True,
) -> Dict:
    """
    Entraîne le modèle pour un run complet.

    Implémente :
    - Gradient clipping (max_norm=1.0) → stabilité (Zhang et al., 2019)
    - Gradient accumulation → simule des batches plus grands sur CPU
    - Early stopping → évite l'overfitting et économise du temps

    Args:
        model       : modèle DistilBERT
        optimizer   : optimiseur construit par model_setup.build_optimizer()
        scheduler   : scheduler LR
        train_loader: DataLoader d'entraînement
        val_loader  : DataLoader de validation
        cfg         : Config du projet
        run_id      : identifiant du run (pour logs)
        verbose     : affichage progressif

    Returns:
        dict avec métriques finales + historique de training
    """
    device = torch.device(cfg.project.device)
    model.train()

    # Historiques
    train_losses: List[float] = []
    val_metrics_history: List[Dict] = []

    # Early stopping
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    # Compteurs
    global_step = 0
    start_time = time.time()

    accumulation_steps = cfg.training.gradient_accumulation_steps
    max_steps = cfg.training.max_steps
    eval_steps = cfg.training.eval_steps

    if verbose:
        print(f"\n[TRAIN] Run {run_id} | max_steps={max_steps} | eval_steps={eval_steps}")

    optimizer.zero_grad()

    # Boucle sur les epochs (on s'arrête aussi via max_steps)
    epoch = 0
    stop_training = False

    while global_step < max_steps and not stop_training:
        epoch += 1
        epoch_loss = 0.0
        n_batches = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not verbose, leave=False)

        for batch in iterator:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / accumulation_steps
            loss.backward()

            epoch_loss += outputs.loss.item()
            n_batches += 1

            # Gradient accumulation
            if (n_batches % accumulation_steps == 0) or (n_batches == len(train_loader)):
                # Gradient clipping — stabilité numérique
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = epoch_loss / n_batches
                train_losses.append(avg_loss)

                if verbose:
                    iterator.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})

                # Évaluation périodique
                if global_step % eval_steps == 0:
                    val_metrics = evaluate(model, val_loader, device)
                    val_metrics["step"] = global_step
                    val_metrics_history.append(val_metrics)
                    model.train()

                    if verbose:
                        print(f"  [Step {global_step:4d}] val_loss={val_metrics['loss']:.4f} | "
                              f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f}")

                    # Early stopping
                    if val_metrics["f1"] > best_val_f1:
                        best_val_f1 = val_metrics["f1"]
                        patience_counter = 0
                        # Sauvegarde du meilleur état
                        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= cfg.training.early_stopping_patience:
                            if verbose:
                                print(f"  [Early Stopping] Patience épuisée à step {global_step}")
                            stop_training = True
                            break

                if global_step >= max_steps:
                    stop_training = True
                    break

    # Restauration du meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Évaluation finale
    final_val = evaluate(model, val_loader, device)
    elapsed = (time.time() - start_time) / 60

    result = {
        "run_id":            run_id,
        "final_val_accuracy": final_val["accuracy"],
        "final_val_f1":       final_val["f1"],
        "final_val_loss":     final_val["loss"],
        "best_val_f1":        best_val_f1,
        "train_time_min":     round(elapsed, 2),
        "steps_done":         global_step,
        "epochs_done":        epoch,
        "train_loss_history": train_losses,
        "val_metrics_history": val_metrics_history,
    }

    if verbose:
        print(f"\n[TRAIN] Run {run_id} terminé en {elapsed:.1f} min")
        print(f"        val_accuracy={final_val['accuracy']:.4f} | val_f1={final_val['f1']:.4f}")

    return result


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("train.py — module utilitaire, importer depuis random_search.py")
