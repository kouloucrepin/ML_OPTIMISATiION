"""
model_setup.py — Initialisation et configuration de DistilBERT
G05 | M01 | P01

Références :
- Sanh et al. (2019) "DistilBERT, a distilled version of BERT"
  → DistilBERT conserve 97% des performances de BERT avec 40% moins de paramètres
- Sun et al. (2019) "How to Fine-Tune BERT for Text Classification"
  → Recommandations de fine-tuning : LR 2e-5 à 5e-5, 3-4 epochs
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertConfig,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import Config, get_config


# ─────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────

def load_model(cfg: Config) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Charge DistilBERT avec les optimisations adaptées au matériel.

    DistilBERT (Sanh et al., 2019) :
    - 6 couches Transformer (vs 12 pour BERT-base)
    - 66M paramètres
    - Entraîné par distillation de BERT
    - 60% plus rapide, 40% moins de mémoire

    Returns:
        model, tokenizer
    """
    device = torch.device(cfg.project.device)
    print(f"[MODEL] Device : {device}")
    print(f"[MODEL] Chargement de {cfg.model.model_name}...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_config = DistilBertConfig.from_pretrained(
        cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        seq_classif_dropout=cfg.model.hidden_dropout_prob,
        dropout=cfg.model.hidden_dropout_prob,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        config=model_config,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    # Optimisations CPU
    if device.type == "cpu":
        torch.set_num_threads(cfg.model.num_threads)
        print(f"[MODEL] Threads CPU : {cfg.model.num_threads}")

    # Compte des paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Paramètres totaux    : {total_params:,}")
    print(f"[MODEL] Paramètres entraînables : {trainable_params:,}")
    print(f"[MODEL] Modèle chargé ✓")

    return model, tokenizer


def load_fresh_model(cfg: Config) -> nn.Module:
    """
    Charge une nouvelle instance du modèle (sans tokenizer).
    Utilisé pour chaque run du Random Search — garantit que
    les poids sont réinitialisés entre les runs.

    IMPORTANT : Chaque trial repart de zéro depuis les poids pré-entraînés.
    Bonne pratique issue de Dodge et al. (2020).
    """
    device = torch.device(cfg.project.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_config = DistilBertConfig.from_pretrained(
    cfg.model.model_name,
    num_labels=cfg.model.num_labels,
    seq_classif_dropout=cfg.model.hidden_dropout_prob,
    dropout=cfg.model.hidden_dropout_prob,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        config=model_config,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


# ─────────────────────────────────────────────
# QUANTIZATION (option mémoire faible)
# ─────────────────────────────────────────────

def quantize_model(model: nn.Module) -> nn.Module:
    """
    Quantization dynamique INT8 — réduit la mémoire ~4x.

    Recommandé pour CPU avec < 6Go RAM.
    Perte de précision généralement < 1% sur les tâches de classification.
    Référence : Zafrir et al. (2019) 'Q8BERT: Quantized 8Bit BERT'
    """
    from torch.quantization import quantize_dynamic
    model_q = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    print("[MODEL] Quantization INT8 appliquée (mémoire réduite ~4x)")
    return model_q


# ─────────────────────────────────────────────
# OPTIMISEUR
# ─────────────────────────────────────────────

def build_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float = 0.01, momentum: float = 0.9):
    """
    Construit l'optimiseur selon le nom.

    Trois optimiseurs testés (P01) :

    1. AdamW (Loshchilov & Hutter, 2019 'Decoupled Weight Decay Regularization')
       → Standard de facto pour le fine-tuning de Transformers.
       Corrige le bug de weight decay dans Adam classique.

    2. SGD + momentum (Sutskever et al., 2013)
       → Plus difficile à tuner mais peut trouver des minima plus plats
       (Keskar et al., 2017 'On Large-Batch Training for Deep Learning').

    3. Adafactor (Shazeer & Stern, 2018 'Adafactor: Adaptive Learning Rates with
       Sublinear Memory Cost')
       → Très économe en mémoire car n'accumule pas de seconds moments
       complets. Idéal pour CPU avec RAM limitée.

    Args:
        model        : modèle PyTorch
        optimizer_name : 'adamw' | 'sgd' | 'adafactor'
        lr           : learning rate
        weight_decay : régularisation L2 (ignoré pour Adafactor)
        momentum     : momentum SGD

    Returns:
        optimizer
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        # no_decay : on n'applique pas weight_decay sur les biais et LayerNorm
        # Bonne pratique issue de Wolf et al. (2020) HuggingFace
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(params, lr=lr)

    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,  # Nesterov momentum → convergence légèrement plus rapide
        )

    elif optimizer_name == "adafactor":
        try:
            from transformers.optimization import Adafactor
            return Adafactor(
                model.parameters(),
                lr=lr,
                relative_step=False,   # LR fixe depuis notre Random Search
                scale_parameter=False,
                warmup_init=False,
            )
        except ImportError:
            print("[WARN] Adafactor non disponible, fallback AdamW")
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Optimiseur inconnu : {optimizer_name}. Choisir parmi : adamw, sgd, adafactor")


# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup + linear decay.

    Le warmup est essentiel pour les Transformers : démarrer avec un LR
    trop élevé casse les poids pré-entraînés.
    Référence : Devlin et al. (2019) BERT, section fine-tuning.
    """
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_config("cpu_low")
    model, tokenizer = load_model(cfg)

    opt = build_optimizer(model, "adamw", lr=2e-5)
    print(f"Optimiseur : {type(opt).__name__}")

    opt_sgd = build_optimizer(model, "sgd", lr=1e-3)
    print(f"Optimiseur : {type(opt_sgd).__name__}")