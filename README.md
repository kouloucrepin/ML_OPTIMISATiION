# G05 — Benchmark d'Optimiseurs pour le Fine-Tuning de DistilBERT sur AG News

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11.5-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-4.35+-FF9D00?style=flat-square&logo=huggingface&logoColor=white)
![Dash](https://img.shields.io/badge/Plotly_Dash-2.x-00A3E0?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-22C55E?style=flat-square)
![Deadline](https://img.shields.io/badge/Deadline-13_mars_2026-F59E0B?style=flat-square)

**Cours : Fine-tuning de Transformers · Enseignante : MBIA NDI Marie Thérèse · Groupe G05 · P01**

</div>

---

## Résumé Exécutif

Ce projet conduit une **étude comparative empirique** de trois optimiseurs de descente de gradient — **AdamW**, **SGD avec momentum de Nesterov** et **Adafactor** — appliqués au fine-tuning du modèle pré-entraîné **DistilBERT** sur la tâche de classification de textes **AG News** (4 classes).

La méthodologie repose sur un protocole de **Random Search** sur l'espace des hyperparamètres, couplé à une analyse de la géométrie des minima via le **loss landscape 1D** et la mesure de **sharpness** (Keskar et al., 2017). L'objectif est de fournir des réponses actionnables aux praticiens NLP confrontés à des choix d'optimisation dans des contextes à ressources contraintes (CPU, mémoire limitée).

> **Question de Recherche (P01) :** Quel optimiseur — AdamW, SGD+Nesterov ou Adafactor — offre le meilleur compromis performance / stabilité / généralisation pour le fine-tuning de DistilBERT sur AG News ?

---

## Table des Matières

1. [Architecture du Projet](#1-architecture-du-projet)
2. [Prérequis & Installation](#2-prérequis--installation)
3. [Méthodologie Expérimentale](#3-méthodologie-expérimentale)
4. [Les Trois Optimiseurs](#4-les-trois-optimiseurs)
5. [Pipeline d'Exécution](#5-pipeline-dexécution)
6. [Dashboard Interactif](#6-dashboard-interactif)
7. [Résultats & Sorties](#7-résultats--sorties)
8. [Hypothèses à Valider](#8-hypothèses-à-valider)
9. [Dépannage](#9-dépannage)
10. [Références](#10-références)

---

## 1. Architecture du Projet

```
G05_AGNews/
│
├── run_experiments.sh              ← Orchestrateur CLI principal
├── requirements_project.txt                ← Dépendances Python (entraînement)
├── requirent_dash.txt              ← Dépendances Python (dashboard)
├── README.md                       ← Ce fichier
├── LICENSE                         ← Licence MIT
├── dashboard_g05_enhanced.py       ← Dashboard interactif Plotly Dash
│
├── assets/
│   └── rapport.pdf                 ← Rapport PDF du projet (téléchargeable)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   ← Configuration centralisée (presets CPU/GPU)
│   ├── data_loader.py              ← Chargement AG News + undersampling équilibré
│   └── model_setup.py              ← DistilBERT + construction des optimiseurs
│
├── optimiseurs/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── random_search.py            ← Moteur Random Search + journalisation
│   ├── train.py                    ← Boucle d'entraînement (gradient clipping, early stopping)
│   └── visualization.py           ← Génération des figures statiques
│
├── notebooks/
│   ├── G05_full_analysis.ipynb    ← Notebook principal d'analyse (papermill)
│   └── start.ipynb                ← Notebook de démarrage / exploration
│
├── logs/                           ← Logs d'exécution papermill + matrice expériences
│   ├── experiment_matrix_<date>.csv
│   └── papermill_<date>.log
│
├── results/                        ← Généré à l'exécution
│   ├── checkpoints/                ← Points de sauvegarde intermédiaires
│   ├── figures/                    ← Visualisations PNG
│   ├── logs/                       ← CSV + JSON des expériences (Random Search)
│   ├── models/                     ← Poids des meilleurs modèles (.pt)
│   └── notebooks/                  ← Notebooks exécutés par papermill
│
└── backups/                        ← Généré si --backup true
    ├── configs/
    ├── models/
    └── results/
```

### Système de Configuration

La classe `Config` dans `src/config.py` centralise tous les hyperparamètres. Trois **presets matériels** sont disponibles :

```python
from src.config import get_config

cfg = get_config("cpu_low")   # RAM < 8 Go  | 12 trials | ~1-2 h
cfg = get_config("cpu_mid")   # RAM 8-16 Go | 20 trials | ~3-5 h  ← défaut
cfg = get_config("gpu")       # GPU ≥ 4 Go  | 30 trials | ~30 min
```

| Sous-config | Rôle | Paramètres clés |
|---|---|---|
| `ProjectConfig` | Chemins, seed, device | `seed=42`, `device="cpu"` |
| `DataConfig` | Dataset et échantillonnage | `samples_per_class=100`, `max_length=128` |
| `ModelConfig` | Architecture | `model_name="distilbert-base-uncased"` |
| `TrainingConfig` | Entraînement | `max_steps=500`, `patience=3`, `grad_clip=1.0` |
| `SearchConfig` | Espace Random Search | `n_trials=20`, `lr_range=[1e-6, 5e-4]` |
| `LandscapeConfig` | Loss landscape | `epsilon=0.5`, `n_points=20` |

> **Organisation des modules :** `src/` contient la configuration et le chargement des données ; `optimiseurs/` contient la logique d'entraînement, le Random Search et les visualisations.

---

## 2. Prérequis & Installation

### Configuration Matérielle

| Mode | RAM | CPU | GPU | Trials | Durée estimée |
|---|---|---|---|---|---|
| `cpu_low` | < 8 Go | 4 cœurs | — | 12 | 1–2 h |
| `cpu_mid` | 8–16 Go | 8 cœurs | — | 20 | 3–5 h |
| `gpu` | 16+ Go | 8 cœurs | ≥ 4 Go VRAM | 30 | ~30 min |

### Logiciels Requis

| Logiciel | Version | Notes |
|---|---|---|
| Python | **3.11.5** | 3.11.x recommandé |
| pip | ≥ 23.0 | Inclus avec Python 3.11 |
| Git | ≥ 2.0 | Versionnement |
| Git Bash | ≥ 2.0 | Windows uniquement — requis pour `.sh` |

### Installation Pas à Pas

```bash
# 1. Se positionner dans le répertoire du projet
cd G05_AGNews

# 2. Créer l'environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Windows PowerShell :
.\venv\Scripts\Activate.ps1
# Windows CMD :
.\venv\Scripts\activate
# Linux / macOS :
source venv/bin/activate

# 4. Mettre à jour pip et installer les dépendances
python -m pip install --upgrade pip
pip install -r requirements_project.txt

# 5. Vérifier l'installation
python -c "import torch, transformers, datasets, dash; print('✅ Installation OK')"

# 6. Rendre le script d'exécution exécutable (Linux/macOS/Git Bash)
chmod +x run_experiments.sh
```

> **⚠️ Note Windows :** Exécutez toujours les scripts `.sh` depuis **Git Bash**, pas depuis PowerShell ou CMD.

---

## 3. Méthodologie Expérimentale

### 3.1 Dataset — AG News

AG News (Zhang et al., 2015) est un benchmark standard de classification de textes anglais composé de 120 000 articles de presse répartis en **4 classes équilibrées** : World, Sports, Business, Sci/Tech.

**Sous-échantillonnage équilibré :** Pour les contraintes CPU, nous appliquons un undersampling stratifié :

| Split | Exemples/classe | Total |
|---|---|---|
| Train | 100 | 400 |
| Validation | 50 | 200 |
| Test | 50 | 200 |

La graine aléatoire est fixée à `42` pour garantir la **reproductibilité** (Dodge et al., 2020). La tokenisation est limitée à **128 tokens** — Sun et al. (2019) montrent que cette longueur est suffisante pour les dépêches de presse courtes.

### 3.2 Modèle — DistilBERT

DistilBERT (Sanh et al., 2019) est une version compressée de BERT par **distillation de connaissances** :

| Propriété | BERT-base | DistilBERT |
|---|---|---|
| Couches Transformer | 12 | 6 |
| Paramètres | 110M | 66M |
| Vitesse d'inférence | 1× | ~1.6× |
| Mémoire | 1× | ~0.6× |
| Score GLUE relatif | 100% | 97% |

Ces caractéristiques en font le modèle optimal pour une exécution **sans GPU**, conformément aux contraintes matérielles du projet.

### 3.3 Random Search

Le choix du **Random Search** plutôt que du Grid Search est motivé par Bergstra & Bengio (2012) : avec un budget de trials fixé, le Random Search est statistiquement aussi efficace avec ~5× moins d'évaluations. Cette efficacité s'explique par la **dominance du learning rate** sur les autres hyperparamètres.

**Espace de recherche :**

| Hyperparamètre | Distribution | Plage / Valeurs |
|---|---|---|
| Learning rate | Log-uniforme | [1×10⁻⁶, 5×10⁻⁴] |
| Optimiseur | Catégorielle | {AdamW, SGD, Adafactor} |
| Batch size | Catégorielle | {8, 16} |
| Warmup steps | Catégorielle | {0, 100, 500} |
| Nombre d'époques | Entier uniforme | [2, 3] |
| Weight decay | Log-uniforme | [1×10⁻⁵, 1×10⁻²] |
| Momentum SGD | Catégorielle | {0.85, 0.90, 0.95} |

> **Garantie P01 :** Avant le tirage aléatoire, **3 trials sont réservés par optimiseur** pour assurer la couverture minimale du protocole.

### 3.4 Techniques d'Entraînement

| Technique | Paramètre | Justification |
|---|---|---|
| **Gradient clipping** | `max_norm=1.0` | Stabilité des Transformers (Zhang et al., 2019) |
| **Gradient accumulation** | `steps=2` | Simule batch 16/32 avec moitié de RAM |
| **Early stopping** | `patience=3` | Prévention overfitting sur petits datasets (Dodge et al., 2020) |
| **Warmup linéaire** | `warmup_steps` variable | Protection des poids pré-entraînés (Devlin et al., 2019) |
| **No weight decay sur biais/LayerNorm** | — | Standard AdamW (Loshchilov & Hutter, 2019) |

### 3.5 Analyse du Loss Landscape

La géométrie des minima est analysée via **perturbation directionnelle 1D avec filter normalization** (Li et al., 2018) :

```
Pour chaque optimiseur θ* :
  1. Tirer une direction aléatoire d dans l'espace des paramètres
  2. Appliquer filter normalization : d_i ← d_i × (‖θ*_i‖ / ‖d_i‖)
  3. Évaluer L(θ* + α·d) pour α ∈ [−ε, +ε], ε=0.5, 20 points
```

La **filter normalization** est indispensable pour corriger le biais d'échelle entre couches. La version 2D (surface 3D) est exclue pour des raisons de coût calcul sur CPU.

**Sharpness** (Keskar et al., 2017) :

```
S(θ*) = (1/N) Σᵢ |L(θ* + ε·dᵢ) − L(θ*)|
```

Une valeur faible indique un minimum plat → meilleure généralisation hors distribution.

---

## 4. Les Trois Optimiseurs

### AdamW — Loshchilov & Hutter (2019)

Standard de facto pour le fine-tuning de Transformers. Corrige un défaut d'Adam classique : dans Adam, le weight decay est absorbé par l'adaptation des moments, le rendant équivalent à une régularisation L2 non standard. AdamW applique le weight decay **directement sur les poids**, indépendamment des moments adaptatifs.

```
θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε) − η · λ · θ_t
```

### SGD + Nesterov Momentum — Sutskever et al. (2013)

Plus difficile à tuner mais présente un avantage théorique important : Keskar et al. (2017) montrent empiriquement que SGD avec petits batches **converge vers des minima plus plats** qu'Adam, ce qui se traduit par une meilleure généralisation. Le momentum de Nesterov anticipe le gradient via un pas de correction :

```
v_{t+1} = μ · v_t − η · ∇L(θ_t + μ · v_t)
θ_{t+1} = θ_t + v_{t+1}
```

### Adafactor — Shazeer & Stern (2018)

Conçu spécifiquement pour les Transformers à mémoire contrainte. Au lieu de stocker les seconds moments complets **O(n×m)**, Adafactor les factorise en matrices rang-1 **O(n+m)**, réduisant drastiquement l'empreinte mémoire. Idéal pour les exécutions CPU de ce projet.

---

## 5. Pipeline d'Exécution

### Démarrage Rapide

```bash
# Exécution complète (défaut : cpu_mid, 20 trials)
./run_experiments.sh

# Mode économique (RAM < 8 Go, 12 trials)
./run_experiments.sh --mode cpu_low

# Mode GPU (30 trials, ~30 min)
./run_experiments.sh --mode gpu
```

### Options Complètes

| Option | Valeurs | Description | Défaut |
|---|---|---|---|
| `--mode` | `cpu_low`, `cpu_mid`, `gpu` | Preset matériel | `cpu_mid` |
| `--run` | `setup`, `data`, `model`, `random_search`, `full` | Étape à exécuter | `full` |
| `--backup` | `true`, `false` | Activer les sauvegardes | `true` |
| `--backup-dir` | `<chemin>` | Dossier de backup | `./backups` |
| `--results-dir` | `<chemin>` | Dossier de résultats | `./results` |
| `--sync` | `user@host:/path` | Synchronisation SSH | — |
| `--cleanup` | `<jours>` | Purge des anciens backups | — |
| `--matrix` | — | Afficher la matrice d'expériences | — |
| `--help` | — | Aide complète | — |

### Scénarios Courants

```bash
# Valider le pipeline avant une longue exécution
./run_experiments.sh --mode cpu_low --run setup

# Exécution complète avec sauvegardes
./run_experiments.sh --mode cpu_mid --run full --backup true

# Exécution GPU production
./run_experiments.sh --mode gpu --run random_search --backup true

# Reprendre après interruption (les trials complétés sont conservés)
./run_experiments.sh --mode cpu_mid --run random_search

# Exécution longue en arrière-plan (Linux/macOS)
nohup ./run_experiments.sh --mode cpu_mid > logs/run.log 2>&1 &

# Avec screen pour reprendre plus tard
screen -S g05 && ./run_experiments.sh --mode cpu_mid
# Détacher : Ctrl+A D | Reprendre : screen -r g05
```

---

## 6. Dashboard Interactif

Un dashboard analytique complet est disponible via **Plotly Dash**, avec support du mode sombre/clair :

```bash
# Installer les dépendances spécifiques au dashboard
pip install -r requirent_dash.txt

# Lancer le dashboard
python dashboard_g05_enhanced.py
# → http://127.0.0.1:8050
```

### Onglets Disponibles

| Onglet | Contenu |
|---|---|
| 📊 Vue d'ensemble | Distribution des classes, Radar Chart comparatif, Matrice de corrélation |
| 📈 Convergence | Courbes train loss + val F1 par optimiseur (meilleur run + tous trials) |
| ⚙️ Hyperparamètres | Scatter LR vs F1, Heatmaps Optimiseur × Plage LR |
| 🔬 Analyses Avancées | Sharpness vs Performance, corrélations hyperparamètres |
| 🏔️ Loss Landscape | Perturbation 1D, sharpness par optimiseur |
| 📦 Distributions | Boxplots F1 / Accuracy / Temps par optimiseur |
| 📋 Tableau Récap | Synthèse complète triable (Best F1, Mean F1, Sharpness, Temps) |

> Le dashboard détecte automatiquement les fichiers de résultats dans `results/logs/`. Les sections non disponibles affichent un état vide explicite.

---

## 7. Résultats & Sorties

### Structure des Fichiers Générés

```
results/
├── logs/
│   ├── random_search_results_<mode>.csv    ← Tous les trials (hyperparamètres + métriques)
│   ├── training_histories.json            ← Courbes complètes (loss + F1 par step)
│   ├── sharpness_scores.json              ← Sharpness par optimiseur
│   └── landscape_results.json            ← Données loss landscape 1D
│
├── figures/
│   ├── 00_data_exploration.png            ← Distribution avant/après undersampling
│   ├── fig1_convergence.png               ← Courbes de convergence (meilleur run)
│   ├── fig1b_all_trials.png               ← Tous les trials superposés
│   ├── fig2_heatmap.png                   ← Heatmap LR × Optimiseur
│   ├── fig2b_lr_scatter.png               ← Scatter LR vs F1 Macro
│   ├── fig3_loss_landscape.png            ← Loss landscape 1D (3 optimiseurs)
│   ├── fig4_sharpness.png                 ← Sharpness vs Performance
│   ├── fig5_summary_table.png             ← Tableau récapitulatif
│   └── fig6_distributions.png            ← Boxplots distributions
│
└── models/
    ├── best_adamw.pt                      ← Meilleur modèle AdamW
    ├── best_sgd.pt                        ← Meilleur modèle SGD
    ├── best_adafactor.pt                  ← Meilleur modèle Adafactor
    └── best_overall.pt                    ← Meilleur modèle global (tous optimiseurs)
```

### Métriques Collectées par Trial

| Colonne CSV | Description |
|---|---|
| `optimizer` | Nom de l'optimiseur |
| `lr` | Learning rate utilisé |
| `batch_size` | Taille de batch |
| `warmup_steps` | Nombre de steps de warmup |
| `num_epochs` | Nombre d'époques |
| `final_val_f1` | F1 Macro sur validation (métrique principale P01) |
| `final_val_accuracy` | Accuracy sur validation |
| `train_time_min` | Durée d'entraînement en minutes |
| `run_id` | Identifiant unique du trial |

---

## 8. Hypothèses à Valider

| ID | Hypothèse | Métrique de Validation | Référence |
|---|---|---|---|
| **H1** | AdamW produit le F1 le plus élevé | `final_val_f1` max | Devlin et al., 2019 |
| **H2** | SGD converge vers des minima plus plats | `sharpness_scores.json` min | Keskar et al., 2017 |
| **H3** | Adafactor présente l'empreinte mémoire la plus faible | Monitoring RAM | Shazeer & Stern, 2018 |
| **H4** | Le learning rate est l'hyperparamètre le plus influent | Corrélation Spearman LR/F1 | Bergstra & Bengio, 2012 |

---

## 9. Dépannage

### Erreurs Fréquentes

**`ModuleNotFoundError: No module named 'src'`**
```bash
# Ajouter le répertoire courant au PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
./run_experiments.sh --mode cpu_mid
```

**`Permission denied: ./run_experiments.sh`**
```bash
chmod +x run_experiments.sh
```

**`CUDA out of memory`**
```bash
# Passer en mode CPU avec batch réduit
./run_experiments.sh --mode cpu_low --backup false
```

**`PapermillExecutionError`**
```bash
# Inspecter les logs détaillés
cat results/logs/papermill_*.log | tail -100

# Déboguer directement dans Jupyter
jupyter notebook notebooks/G05_full_analysis.ipynb
```

### Questions Fréquentes

**Q : Comment modifier le nombre de trials ?**
```python
# src/config.py → fonction get_config()
cfg.search.n_trials = 15  # Valeur personnalisée
```

**Q : Comment reprendre une exécution interrompue ?**  
Les résultats sont sauvegardés trial par trial. Relancez simplement la même commande — les trials déjà complétés sont détectés et ignorés.

**Q : Le dashboard ne charge pas les graphiques.**  
Vérifiez que `results/logs/random_search_results_*.csv` existe. Les graphiques ne s'affichent que si les fichiers de résultats sont présents.

---

## 10. Références

| Auteurs | Titre | Venue | Année |
|---|---|---|---|
| Bergstra & Bengio | Random Search for Hyper-Parameter Optimization | JMLR 13 | 2012 |
| Devlin et al. | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | NAACL | 2019 |
| Dodge et al. | Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping | arXiv:2002.06305 | 2020 |
| Keskar et al. | On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima | ICLR | 2017 |
| Li et al. | Visualizing the Loss Landscape of Neural Nets | NeurIPS | 2018 |
| Loshchilov & Hutter | Decoupled Weight Decay Regularization (AdamW) | ICLR | 2019 |
| Sanh et al. | DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter | NeurIPS Workshop | 2019 |
| Shazeer & Stern | Adafactor: Adaptive Learning Rates with Sublinear Memory Cost | ICML | 2018 |
| Sun et al. | How to Fine-Tune BERT for Text Classification? | CCL | 2019 |
| Sutskever et al. | On the Importance of Initialization and Momentum in Deep Learning | ICML | 2013 |
| Zhang et al. | Character-level Convolutional Networks for Text Classification (AG News) | NeurIPS | 2015 |
| Zhang et al. | Gradient Clipping for Deep Neural Networks | — | 2019 |

---

## Checklist de Démarrage

```bash
# 1. Vérifier Python
python --version          # Doit afficher 3.11.x

# 2. Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\Activate.ps1 sur Windows

# 3. Installer les dépendances d'entraînement
pip install --upgrade pip
pip install -r requirements_project.txt

# 4. Installer les dépendances du dashboard
pip install -r requirent_dash.txt

# 5. Vérifier l'installation complète
python -c "import torch, transformers, datasets, dash, plotly; print('✅ Tout est OK')"

# 6. Rendre le script exécutable
chmod +x run_experiments.sh

# 7. Valider le pipeline (test rapide)
./run_experiments.sh --mode cpu_low --run setup

# 8. Lancer les expériences
./run_experiments.sh --mode cpu_mid --run full

# 9. Visualiser les résultats
python dashboard_g05_enhanced.py  # → http://127.0.0.1:8050
```

---

<div align="center">

**G05 · Cours Fine-tuning de Transformers · MBIA NDI Marie Thérèse**  
AYONTA · BAMOGO · KOULOU  
*Dernière mise à jour : 10 mars 2026 · Version 2.0*

</div>