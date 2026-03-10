"""
config.py — Configuration centralisée du projet G05
Dataset      : AG News (D03)
Modèle       : DistilBERT (M01)
Problématique: P01 - Benchmark d'Optimiseurs
Méthode      : Random Search
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────
# SOUS-CONFIGS
# ─────────────────────────────────────────────

@dataclass
class ProjectConfig:
    project_name: str = "G05_AGNews_DistilBERT_P01"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir:    str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir: str = field(init=False)
    logs_dir:    str = field(init=False)
    figures_dir: str = field(init=False)
    models_dir:  str = field(init=False)

    def __post_init__(self):
        self.results_dir = os.path.join(self.root_dir, "results")
        self.logs_dir    = os.path.join(self.root_dir, "results", "logs")
        self.figures_dir = os.path.join(self.root_dir, "results", "figures")
        self.models_dir  = os.path.join(self.root_dir, "results", "models")
        for d in [self.results_dir, self.logs_dir, self.figures_dir, self.models_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass
class DataConfig:
    dataset_id:   str = "D03"
    dataset_name: str = "ag_news"
    language:     str = "en"
    num_classes:  int = 4
    class_names:  List[str] = field(
        default_factory=lambda: ["World", "Sports", "Business", "Sci/Tech"]
    )
    samples_per_class_train: int = 100
    samples_per_class_val:   int = 50
    samples_per_class_test:  int = 50
    max_length:    int = 128
    text_column:   str = "text"
    label_column:  str = "label"


@dataclass
class ModelConfig:
    model_id:            str   = "M01"
    model_name:          str   = "distilbert-base-uncased"
    num_labels:          int   = 4
    hidden_dropout_prob: float = 0.1
    torch_dtype:         str   = "float32"
    num_threads:         int   = 4


@dataclass
class TrainingConfig:
    max_steps:                   int  = 100
    eval_steps:                  int  = 25
    save_steps:                  int  = 50
    logging_steps:               int  = 10
    gradient_accumulation_steps: int  = 2
    early_stopping_patience:     int  = 3
    output_dir:                  str  = "results/models"
    metric_for_best_model:       str  = "eval_f1"
    greater_is_better:           bool = True


@dataclass
class SearchConfig:
    n_trials:                int        = 20
    optimizers:              List[str]  = field(default_factory=lambda: ["adamw", "sgd", "adafactor"])
    min_trials_per_optimizer: int       = 3
    lr_min:                  float      = 1e-6
    lr_max:                  float      = 5e-4
    batch_sizes:             List[int]  = field(default_factory=lambda: [8, 16])
    warmup_options:          List[int]  = field(default_factory=lambda: [0, 100, 500])
    epoch_min:               int        = 2
    epoch_max:               int        = 3
    momentum_options:        List[float] = field(default_factory=lambda: [0.85, 0.9, 0.95])
    results_csv:             str        = "results/logs/random_search_results.csv"


@dataclass
class LandscapeConfig:
    n_points:       int   = 8
    epsilon:        float = 0.05
    n_samples_eval: int   = 50
    n_directions:   int   = 1


# ─────────────────────────────────────────────
# CLASSE PRINCIPALE
# ─────────────────────────────────────────────

@dataclass
class Config:
    """
    Point d'entrée unique pour toute la configuration.

    Usage :
        from src.config import Config, get_config
        cfg = get_config("cpu_mid")
        print(cfg.model.model_name)
    """
    project:   ProjectConfig  = field(default_factory=ProjectConfig)
    data:      DataConfig     = field(default_factory=DataConfig)
    model:     ModelConfig    = field(default_factory=ModelConfig)
    training:  TrainingConfig = field(default_factory=TrainingConfig)
    search:    SearchConfig   = field(default_factory=SearchConfig)
    landscape: LandscapeConfig = field(default_factory=LandscapeConfig)

    def summary(self):
        print("=" * 55)
        print(f"  PROJET   : {self.project.project_name}")
        print(f"  DEVICE   : {self.project.device}")
        print(f"  DATASET  : {self.data.dataset_name} ({self.data.num_classes} classes)")
        print(f"  MODÈLE   : {self.model.model_name}")
        print(f"  RUNS     : {self.search.n_trials} trials Random Search")
        print(f"  OPTIM.   : {', '.join(self.search.optimizers)}")
        print(f"  MAX STEPS: {self.training.max_steps}")
        print(f"  LR RANGE : [{self.search.lr_min:.0e}, {self.search.lr_max:.0e}]")
        print(f"  SEED     : {self.project.seed}")
        print("=" * 55)


# ─────────────────────────────────────────────
# PRESETS PAR MATÉRIEL
# ─────────────────────────────────────────────

def get_config(mode: str = "cpu_mid") -> Config:
    """
    Retourne une configuration adaptée au matériel disponible.

    Modes :
        cpu_low  — RAM < 8Go  : sous-ensemble réduit, 12 trials
        cpu_mid  — RAM 8-16Go : configuration standard, 20 trials
        gpu      — GPU dispo  : configuration complète, 30 trials

    Usage :
        cfg = get_config("cpu_low")
        bash: python src/random_search.py --mode cpu_low
    """
    cfg = Config()

    if mode == "cpu_low":
        cfg.data.samples_per_class_train = 75
        cfg.data.samples_per_class_val   = 30
        cfg.data.samples_per_class_test  = 30
        cfg.data.max_length              = 64
        cfg.training.max_steps           = 60
        cfg.training.eval_steps          = 15
        cfg.training.early_stopping_patience = 2
        cfg.search.n_trials              = 12
        cfg.search.batch_sizes           = [8]
        cfg.search.warmup_options        = [0, 50]
        cfg.landscape.n_points           = 6
        cfg.landscape.n_samples_eval     = 30
        cfg.model.num_threads            = 2

    elif mode == "cpu_mid":
        pass  # Valeurs par défaut

    elif mode == "gpu":
        cfg.data.samples_per_class_train = 500
        cfg.data.samples_per_class_val   = 200
        cfg.data.samples_per_class_test  = 200
        cfg.data.max_length              = 256
        cfg.training.max_steps           = 500
        cfg.training.eval_steps          = 50
        cfg.training.early_stopping_patience = 5
        cfg.training.gradient_accumulation_steps = 1
        cfg.search.n_trials              = 30
        cfg.search.batch_sizes           = [16, 32]
        cfg.landscape.n_points           = 15
        cfg.landscape.n_samples_eval     = 200
        cfg.model.torch_dtype            = "float16"

    else:
        raise ValueError(f"Mode inconnu : {mode}. Choisir parmi : cpu_low, cpu_mid, gpu")

    # Mise à jour du chemin CSV avec le mode
    cfg.search.results_csv = os.path.join(
        cfg.project.logs_dir, f"random_search_results_{mode}.csv"
    )

    return cfg


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    for mode in ["cpu_low", "cpu_mid", "gpu"]:
        print(f"\n--- Mode : {mode} ---")
        cfg = get_config(mode)
        cfg.summary()
