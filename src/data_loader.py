"""
data_loader.py — Chargement et préparation des données AG News
G05 | D03 | P01

Références :
- Zhang et al. (2015) "Character-level Convolutional Networks for Text Classification"
  → AG News dataset original
- Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
  → tokenisation strategy max_length=128 pour économie mémoire
"""

import os
import sys
import argparse
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Ajout du dossier racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config, Config


# ─────────────────────────────────────────────
# SEED GLOBALE
# ─────────────────────────────────────────────

def set_seed(seed: int):
    """
    Fixe toutes les graines aléatoires pour la reproductibilité.
    Bonne pratique recommandée par Dodge et al. (2020)
    'Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping'
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────
# DATASET TORCH
# ─────────────────────────────────────────────

class AGNewsDataset(Dataset):
    """
    Dataset PyTorch pour AG News.
    Encapsule les encodages tokenisés et les labels.
    """

    def __init__(self, encodings: Dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────────────────────────
# CHARGEMENT DATASET
# ─────────────────────────────────────────────

def load_ag_news_raw():
    """
    Charge AG News depuis HuggingFace datasets.
    Gestion du cas offline (cache local).
    """
    try:
        from datasets import load_dataset
        print("  Chargement AG News depuis HuggingFace...")
        dataset = load_dataset("ag_news")
        print(f"  Train : {len(dataset['train'])} exemples")
        print(f"  Test  : {len(dataset['test'])} exemples")
        return dataset
    except Exception as e:
        print(f"  [ERREUR] Impossible de charger le dataset : {e}")
        print("  Vérifiez votre connexion internet ou le cache HuggingFace.")
        sys.exit(1)


# ─────────────────────────────────────────────
# SOUS-ÉCHANTILLONNAGE ÉQUILIBRÉ
# ─────────────────────────────────────────────

def balanced_subsample(dataset_split, num_classes: int, n_per_class: int, seed: int) -> Tuple[list, list]:
    """
    Sous-échantillonnage équilibré par classe.

    Pourquoi ? Les datasets déséquilibrés biaisant le F1-score macro,
    nous maintenons une distribution uniforme sur les 4 classes AG News.
    Référence : Japkowicz & Stephen (2002) "The class imbalance problem"

    Args:
        dataset_split : split HuggingFace (train ou test)
        num_classes   : nombre de classes
        n_per_class   : exemples à garder par classe
        seed          : graine aléatoire

    Returns:
        texts, labels : listes équilibrées
    """
    rng = np.random.default_rng(seed)

    texts_by_class = {c: [] for c in range(num_classes)}
    for example in dataset_split:
        texts_by_class[example["label"]].append(example["text"])

    texts, labels = [], []
    for c in range(num_classes):
        available = texts_by_class[c]
        n = min(n_per_class, len(available))
        indices = rng.choice(len(available), size=n, replace=False)
        for i in indices:
            texts.append(available[i])
            labels.append(c)

    # Mélange final
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)

    return list(texts), list(labels)


# ─────────────────────────────────────────────
# TOKENISATION
# ─────────────────────────────────────────────

def tokenize_texts(texts: list, tokenizer, max_length: int) -> Dict:
    """
    Tokenise une liste de textes avec padding et truncation.

    max_length=128 : compromis vitesse/qualité recommandé pour CPU.
    Sun et al. (2019) 'How to Fine-Tune BERT for Text Classification'
    montrent que 128 tokens suffisent pour la plupart des tâches de
    classification sur des textes courts comme les news.
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,  # Retourne des listes → conversion dans __getitem__
    )


# ─────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────

def prepare_data(cfg: Config) -> Tuple[AGNewsDataset, AGNewsDataset, AGNewsDataset]:
    """
    Pipeline complet : chargement → sous-échantillonnage → tokenisation.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    set_seed(cfg.project.seed)

    print("\n[DATA] Chargement AG News...")
    raw = load_ag_news_raw()

    print(f"[DATA] Sous-échantillonnage (mode: {cfg.project.device})")
    print(f"       Train : {cfg.data.samples_per_class_train} × {cfg.data.num_classes} classes = {cfg.data.samples_per_class_train * cfg.data.num_classes}")
    print(f"       Val   : {cfg.data.samples_per_class_val} × {cfg.data.num_classes} classes = {cfg.data.samples_per_class_val * cfg.data.num_classes}")
    print(f"       Test  : {cfg.data.samples_per_class_test} × {cfg.data.num_classes} classes = {cfg.data.samples_per_class_test * cfg.data.num_classes}")

    # Sous-échantillonnage — on prend val et test depuis le split test officiel
    train_texts, train_labels = balanced_subsample(
        raw["train"], cfg.data.num_classes, cfg.data.samples_per_class_train, cfg.project.seed
    )
    val_texts, val_labels = balanced_subsample(
        raw["test"], cfg.data.num_classes, cfg.data.samples_per_class_val, cfg.project.seed + 1
    )
    test_texts, test_labels = balanced_subsample(
        raw["test"], cfg.data.num_classes, cfg.data.samples_per_class_test, cfg.project.seed + 2
    )

    # Tokenisation
    print(f"[DATA] Tokenisation avec {cfg.model.model_name} (max_length={cfg.data.max_length})...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    train_enc = tokenize_texts(train_texts, tokenizer, cfg.data.max_length)
    val_enc   = tokenize_texts(val_texts,   tokenizer, cfg.data.max_length)
    test_enc  = tokenize_texts(test_texts,  tokenizer, cfg.data.max_length)

    train_dataset = AGNewsDataset(train_enc, train_labels)
    val_dataset   = AGNewsDataset(val_enc,   val_labels)
    test_dataset  = AGNewsDataset(test_enc,  test_labels)

    print(f"[DATA] Datasets prêts ✓")

    # Sauvegarde des stats
    stats = {
        "train_size": len(train_dataset),
        "val_size":   len(val_dataset),
        "test_size":  len(test_dataset),
        "num_classes": cfg.data.num_classes,
        "class_names": cfg.data.class_names,
        "max_length":  cfg.data.max_length,
        "seed":        cfg.project.seed,
    }
    stats_path = os.path.join(cfg.project.logs_dir, "data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[DATA] Stats sauvegardées → {stats_path}")

    return train_dataset, val_dataset, test_dataset


def get_dataloader(dataset: AGNewsDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Crée un DataLoader PyTorch."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chargement et préparation AG News")
    parser.add_argument("--mode",       type=str, default="cpu_mid", choices=["cpu_low", "cpu_mid", "gpu"])
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/logs")
    args = parser.parse_args()

    cfg = get_config(args.mode)
    cfg.project.seed = args.seed

    train, val, test = prepare_data(cfg)
    print(f"\n  Train : {len(train)} exemples")
    print(f"  Val   : {len(val)} exemples")
    print(f"  Test  : {len(test)} exemples")
