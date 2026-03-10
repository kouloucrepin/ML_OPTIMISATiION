# 🚀 G05 — Benchmark d'Optimiseurs pour le Fine-Tuning de DistilBERT sur AG News

[![Python](https://img.shields.io/badge/Python-3.11.5-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35+-ff9d00.svg?style=flat-square)](https://huggingface.co/docs/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg?style=flat-square)]()

---

## 📖 Résumé Exécutif

Ce projet de recherche expérimentale vise à répondre à une question fondamentale en optimisation de modèles de langage : **quel optimiseur — AdamW, SGD avec momentum de Nesterov, ou Adafactor — offre les meilleures performances pour le fine-tuning de DistilBERT sur la tâche de classification de textes AG News ?** 

Notre méthodologie repose sur une approche systématique de **Random Search** couplée à une analyse approfondie du **loss landscape**, permettant d'évaluer non seulement la performance finale des modèles, mais également la qualité des minima atteints en termes de platitude et de généralisation. Ce benchmark fournit des insights actionnables pour les praticiens du NLP confrontés à des contraintes matérielles variées (CPU vs GPU, mémoire limitée vs ressources abondantes).

---

## 📋 Table des Matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Prérequis Système](#2-prérequis-système)
3. [Installation](#3-installation)
4. [Architecture du Projet](#4-architecture-du-projet)
5. [Méthodologie Expérimentale](#5-méthodologie-expérimentale)
6. [Utilisation de `run_experiments.sh`](#6-utilisation-de-run_experimentssh)
7. [Résultats Attendus](#7-résultats-attendus)
8. [Dépannage & FAQ](#8-dépannage--faq)
9. [Références Académiques](#9-références-académiques)
10. [Informations du Projet](#10-informations-du-projet)

---

## 1. Vue d'ensemble

### 1.1 Question de Recherche (P01)

La problématique centrale de ce travail s'inscrit dans le contexte plus large de l'optimisation des modèles de transformers pour des tâches de classification de textes. La question de recherche peut être formulée comme suit :

> **Quel optimiseur — AdamW, SGD+momentum ou Adafactor — donne les meilleurs résultats pour le fine-tuning de DistilBERT sur une tâche de classification de textes (AG News, 4 classes) ?**

Cette question revêt une importance pratique considérable, car le choix de l'optimiseur influence non seulement les performances finales du modèle, mais également le temps d'entraînement, la consommation mémoire, et la capacité de généralisation à des données non vues.

### 1.2 Métriques d'Évaluation

Notre protocole d'évaluation repose sur trois dimensions complémentaires :

| Métrique | Description | Justification |
|---|---|---|
| **Performance Finale** | F1 Macro et Accuracy sur validation | Mesure standard de la qualité de classification |
| **Vitesse de Convergence** | Loss et F1 en fonction des steps | Évalue l'efficacité de l'optimisation |
| **Platitude du Minimum** | Sharpness (Li et al., 2018) | Corrélé à la capacité de généralisation (Keskar et al., 2017) |

### 1.3 Protocole Expérimental

Pour chaque optimiseur considéré (AdamW, SGD, Adafactor), nous appliquons le protocole suivant :

```
Pour chaque optimiseur {AdamW, SGD, Adafactor} :
  ├── Tester ≥ 3 learning rates (Random Search, distribution log-uniforme)
  ├── Enregistrer les courbes de convergence complètes
  ├── Calculer le F1 macro sur l'ensemble de validation
  └── Analyser le loss landscape (sharpness via perturbation 1D)
```

Ce protocole garantit une comparaison équitable entre les optimiseurs tout en explorant suffisamment l'espace des hyperparamètres pour identifier les configurations optimales.

---

## 2. Prérequis Système

### 2.1 Configuration Matérielle Recommandée

La configuration matérielle requise dépend du mode d'exécution choisi. Nous proposons trois presets adaptés à différentes contraintes :

| Mode | RAM | CPU | GPU | Trials | Temps Estimé |
|---|---|---|---|---|---|
| **cpu_low** | < 8 Go | 4 cœurs | Non requis | 12 | 1-2 heures |
| **cpu_mid** | 8-16 Go | 8 cœurs | Non requis | 20 | 3-5 heures |
| **gpu** | 16+ Go | 8 cœurs | Requis (4Go+) | 30 | ~30 minutes |

### 2.2 Logiciels Requis

| Logiciel | Version Minimale | Notes |
|---|---|---|
| **Python** | 3.11.5 (3.11.x compatible) | ⚠️ 3.10 et 3.12 non testés |
| **pip** | 23.0+ | Inclus avec Python 3.11+ |
| **Git** | 2.0+ | Pour le versionnement |
| **Git Bash** (Windows) | 2.0+ | Requis pour l'exécution des scripts shell |

### 2.3 Vérification de l'Environnement

Avant de procéder à l'installation, nous recommandons de vérifier que votre environnement satisfait les prérequis suivants :

```bash
# Vérifier la version Python
python --version
# Doit afficher : Python 3.11.5

# Vérifier la version pip
pip --version
# Doit afficher : pip 23.x ou supérieur

# Vérifier la version Git
git --version
# Doit afficher : git version 2.x ou supérieur

# Vérifier l'espace disque disponible
df -h .
# Doit afficher : au moins 5 Go libres
```

---

## 3. Installation

### 3.1 Clonage et Positionnement

La première étape consiste à vous positionner dans le répertoire racine du projet. Si vous avez cloné le dépôt depuis un système de versionnement, naviguez vers le dossier `G05_AGNews` :

```bash
cd G05_AGNews
```

### 3.2 Création de l'Environnement Virtuel

L'utilisation d'un environnement virtuel constitue une pratique essentielle en développement Python. Elle permet d'isoler les dépendances du projet et d'éviter les conflits avec d'autres projets installés sur votre machine. Pour créer un environnement virtuel nommé `venv`, exécutez la commande suivante :

```bash
python -m venv venv
```

Une fois l'environnement créé, vous devez l'activer. La commande d'activation dépend de votre système d'exploitation et de votre shell :

| Système | Commande d'Activation |
|---|---|
| **Windows (PowerShell)** | `.\venv\Scripts\Activate.ps1` |
| **Windows (CMD)** | `.\venv\Scripts\activate` |
| **Linux/macOS** | `source venv/bin/activate` |

Après activation, votre invite de commande devrait afficher le préfixe `(venv)`, indiquant que vous travaillez désormais dans l'environnement isolé.

### 3.3 Installation des Dépendances

Le fichier `requirements.txt` liste l'ensemble des bibliothèques Python nécessaires au bon fonctionnement du projet. Pour les installer toutes en une seule commande, exécutez :

```bash
# Mettre à jour pip en premier lieu
python -m pip install --upgrade pip

# Installer toutes les dépendances
pip install -r requirements.txt
```

Cette étape peut prendre plusieurs minutes, en fonction de votre connexion internet et de la puissance de votre machine. Une fois terminée, vérifiez que les packages principaux sont bien installés :

```bash
python -c "import torch, transformers, datasets; print('✅ Installation réussie')"
```

### 3.4 Préparation du Script d'Exécution

Le script `run_experiments.sh` est le point d'entrée principal du projet. Avant de l'exécuter, assurez-vous qu'il dispose des permissions d'exécution :

```bash
chmod +x run_experiments.sh
```

Cette commande est nécessaire uniquement sur Linux, macOS, ou Git Bash sous Windows.

---

## 4. Architecture du Projet

### 4.1 Structure des Répertoires

L'architecture du projet a été conçue pour favoriser la modularité, la maintenabilité et la reproductibilité des expériences :

```
G05_AGNews/
│
├── run_experiments.sh          ← Script principal d'orchestration (CLI)
├── requirements.txt            ← Dépendances Python
├── README.md                   ← Documentation principale
├── .python-version             ← Version Python (pour pyenv)
├── check_python.py             ← Script de vérification de version
│
├── src/
│   ├── config.py               ← Classe Config centralisée (hyperparamètres)
│   ├── data_loader.py          ← Chargement AG News + sous-échantillonnage
│   ├── model_setup.py          ← DistilBERT + construction des optimiseurs
│   ├── train.py                ← Boucle d'entraînement complète
│   ├── random_search.py        ← Moteur Random Search + sauvegarde
│   └── visualization.py        ← Génération des figures d'analyse
│
├── notebooks/
│   └── G05_full_analysis.ipynb ← Notebook principal d'analyse
│
├── results/                    ← Généré automatiquement lors de l'exécution
│   ├── logs/                   ← Logs, CSV, JSON des expériences
│   ├── figures/                ← Visualisations et graphiques
│   └── models/                 ← Poids des meilleurs modèles (.pt)
│
└── backups/                    ← Généré si --backup true
    ├── configs/                ← Snapshots de configuration
    ├── results/                ← Copies des résultats
    └── models/                 ← Copies des modèles
```

### 4.2 Système de Configuration

La classe `Config` dans `src/config.py` constitue le **point d'entrée unique** pour tous les hyperparamètres du projet. Elle est divisée en sous-configurations indépendantes, chacune responsable d'un aspect spécifique du pipeline :

| Sous-configuration | Rôle | Paramètres Principaux |
|---|---|---|
| `ProjectConfig` | Chemins, seed, device | `seed=42`, `device='cpu'` |
| `DataConfig` | Dataset et échantillonnage | `samples_per_class=100`, `max_length=128` |
| `ModelConfig` | Architecture du modèle | `model_name='distilbert-base-uncased'` |
| `TrainingConfig` | Paramètres d'entraînement | `max_steps=500`, `patience=3` |
| `SearchConfig` | Espace de recherche | `n_trials=20`, `lr_range=[1e-6, 5e-4]` |
| `LandscapeConfig` | Analyse du loss landscape | `epsilon=0.5`, `n_points=20` |

Trois presets matériels sont disponibles via la fonction `get_config(mode)` :

```python
cfg = get_config("cpu_low")   # RAM < 8Go, 12 trials
cfg = get_config("cpu_mid")   # RAM 8-16Go, 20 trials (défaut)
cfg = get_config("gpu")       # GPU disponible, 30 trials
```

---

## 5. Méthodologie Expérimentale

### 5.1 Dataset : AG News

AG News (Zhang et al., 2015) constitue un benchmark standard de classification de textes en anglais, composé de 120 000 articles de presse répartis en **quatre classes équilibrées** : World, Sports, Business, et Sci/Tech. 

Pour adapter ce dataset aux contraintes d'exécution sur CPU, nous appliquons un sous-échantillonnage équilibré : 100 exemples par classe pour l'entraînement (400 au total), et 50 exemples par classe pour la validation et le test. Ce sous-échantillonnage est **équilibré par classe** pour éviter les biais de distribution (Japkowicz & Stephen, 2002). La graine aléatoire est fixée à 42 pour garantir la reproductibilité des résultats (Dodge et al., 2020).

### 5.2 Modèle : DistilBERT

DistilBERT (Sanh et al., 2019) est une version allégée de BERT obtenue par **distillation de connaissances**, présentant les caractéristiques suivantes :

- 6 couches Transformer au lieu de 12 (BERT-base)
- 66 millions de paramètres (vs 110 millions pour BERT-base)
- 60% plus rapide en inférence, 40% moins de mémoire
- Conserve 97% des performances de BERT sur le benchmark GLUE

Ces caractéristiques en font le choix optimal pour une exécution sur **CPU sans GPU**, conformément aux contraintes matérielles de ce projet. La tokenisation est limitée à **128 tokens maximum** : Sun et al. (2019) démontrent que cette longueur suffit pour la classification de textes courts comme les dépêches de presse, avec une réduction significative du temps de calcul.

### 5.3 Random Search vs Grid Search

Le choix du **Random Search** plutôt que du Grid Search est motivé par les travaux de Bergstra & Bengio (2012), qui démontrent théoriquement et empiriquement que le Random Search est aussi efficace que le Grid Search avec environ **5 fois moins de trials**. Cette efficacité s'explique par le fait que les hyperparamètres n'ont pas tous le même impact sur la performance, et le learning rate domine largement les autres paramètres.

**Espace de recherche exploré :**

| Hyperparamètre | Distribution | Plage |
|---|---|---|
| Learning rate | Log-uniforme | [1×10⁻⁶, 5×10⁻⁴] |
| Optimiseur | Catégorielle | AdamW, SGD, Adafactor |
| Batch size | Catégorielle | {8, 16} |
| Warmup steps | Catégorielle | {0, 100, 500} |
| Epochs | Entier uniforme | [2, 3] |
| Weight decay | Log-uniforme | [1×10⁻⁵, 1×10⁻²] |
| Momentum (SGD) | Catégorielle | {0.85, 0.9, 0.95} |

Avant de tirer aléatoirement les combinaisons d'hyperparamètres, nous réservons **3 trials par optimiseur** pour garantir le protocole P01 (au moins 3 learning rates par optimiseur). Les trials restants sont tirés aléatoirement dans l'espace défini ci-dessus.

### 5.4 Techniques d'Entraînement

Plusieurs techniques éprouvées sont implémentées pour stabiliser et optimiser l'entraînement :

**Gradient clipping (max_norm=1.0)** : Essentiel pour la stabilité du fine-tuning de Transformers. Zhang et al. (2019) montrent que le clipping évite les explosions de gradient pendant les premiers steps de l'entraînement.

**Gradient accumulation (steps=2)** : Permet de simuler un batch de taille 16 ou 32 en utilisant seulement 8 ou 16 en mémoire. Cette technique permet de contourner les contraintes de RAM sans sacrifier la qualité de l'optimisation.

**Early stopping (patience=3)** : Arrête l'entraînement si le F1 de validation ne s'améliore pas pendant 3 évaluations consécutives. Dodge et al. (2020) identifient l'early stopping comme la technique la plus efficace pour éviter l'overfitting sur les petits datasets de fine-tuning.

**Warmup linéaire + décroissance linéaire** : Recommandé par Devlin et al. (2019) pour protéger les poids pré-entraînés des mises à jour trop agressives au début de l'entraînement.

**No weight decay sur biais et LayerNorm** : Pratique standard depuis Loshchilov & Hutter (2019). Le weight decay sur ces paramètres dégrade les performances sans bénéfice de régularisation.

### 5.5 Les Trois Optimiseurs (P01)

#### AdamW (Loshchilov & Hutter, 2019)

AdamW constitue le standard de facto pour le fine-tuning de Transformers. Il corrige un bug fondamental d'Adam : dans Adam classique, le weight decay est appliqué *après* la mise à l'échelle par les moments, ce qui revient à une régularisation L2 non standard. AdamW applique le weight decay *directement* sur les poids, indépendamment de l'adaptation du learning rate.

#### SGD + Nesterov Momentum (Sutskever et al., 2013)

Plus difficile à tuner que les méthodes adaptatives, mais Keskar et al. (2017) montrent que SGD avec petit batch tend à trouver des **minima plus plats** qu'Adam, ce qui se traduit par une meilleure généralisation. Le momentum de Nesterov (vs momentum classique) permet une convergence légèrement plus rapide grâce à l'anticipation du gradient.

#### Adafactor (Shazeer & Stern, 2018)

Conçu spécifiquement pour les Transformers sur CPU ou mémoire limitée. Adafactor n'accumule pas les seconds moments complets (comme Adam) mais les reconstruit depuis des matrices de rang inférieur, conduisant à une économie de mémoire **O(n+m)** au lieu de **O(n×m)**. Cet optimiseur est idéal pour les contraintes matérielles de ce projet.

### 5.6 Analyse du Loss Landscape (Li et al., 2018)

Notre méthodologie inclut une analyse du loss landscape via **perturbation directionnelle 1D avec filter normalization** :

1. Choisir une direction aléatoire `d` dans l'espace des paramètres
2. Appliquer la filter normalization : normaliser `d` par la norme de chaque filtre du modèle correspondant (invariance à l'échelle)
3. Évaluer `L(θ + α·d)` pour α ∈ [-ε, ε]

La **filter normalization** (Li et al., 2018, équation 4) est cruciale : sans cette normalisation, la visualisation est biaisée par les couches à grande norme. La version 2D (surface 3D) étant trop coûteuse sur CPU, nous utilisons la version 1D.

La **Sharpness** (Keskar et al., 2017) est calculée comme suit :

```
Sharpness = (1/N) Σ |L(θ + ε·di) - L(θ)|
```

Un sharpness élevé indique un minimum pointu (mauvaise généralisation), tandis qu'un sharpness faible indique un minimum plat (bonne généralisation).

---

## 6. Utilisation de `run_experiments.sh`

### 6.1 Démarrage Rapide

Pour les utilisateurs souhaitant exécuter le pipeline sans personnalisation, les commandes suivantes suffisent :

```bash
# Exécution complète avec paramètres par défaut (cpu_mid, 20 trials)
./run_experiments.sh

# Mode économique (RAM < 8Go, 12 trials, ~1-2 heures)
./run_experiments.sh --mode cpu_low

# Mode GPU (30 trials, ~30 minutes)
./run_experiments.sh --mode gpu
```

### 6.2 Commandes Disponibles

Le script supporte un ensemble complet d'options pour personnaliser l'exécution :

| Option | Valeurs | Description | Défaut |
|---|---|---|---|
| `--mode` | `cpu_low`, `cpu_mid`, `gpu` | Mode matériel | `cpu_mid` |
| `--run` | `setup`, `data`, `model`, `random_search`, `full` | Type d'exécution | `full` |
| `--backup` | `true`, `false` | Activer les sauvegardes | `true` |
| `--backup-dir` | `<chemin>` | Dossier de backup | `./backups` |
| `--results-dir` | `<chemin>` | Dossier de résultats | `./results` |
| `--sync` | `user@host:/path` | Synchronisation vers serveur distant | — |
| `--cleanup` | `<jours>` | Supprimer les backups anciens | — |
| `--matrix` | — | Générer la matrice d'expériences | — |
| `--help` | — | Afficher l'aide complète | — |

### 6.3 Scénarios d'Utilisation Typiques

#### Scénario 1 : Premier Lancement (Validation du Pipeline)

Avant de lancer une exécution complète, nous recommandons de valider que l'ensemble du pipeline fonctionne correctement :

```bash
# Test rapide avec configuration minimale
./run_experiments.sh --mode cpu_low --run setup

# Vérifier que les dossiers sont créés
ls -la results/
ls -la backups/
```

#### Scénario 2 : Exécution Complète Standard

Pour une exécution complète avec tous les trials et sauvegardes :

```bash
# 20 trials, mode CPU intermédiaire, avec backups
./run_experiments.sh --mode cpu_mid --run full --backup true

# Temps estimé : 3-5 heures
# Résultats dans : results/logs/, results/figures/, results/models/
```

#### Scénario 3 : Exécution sur GPU

Si vous disposez d'un GPU compatible CUDA :

```bash
# 30 trials, mode GPU, backups activés
./run_experiments.sh --mode gpu --run random_search --backup true

# Temps estimé : ~30 minutes
```

#### Scénario 4 : Exécution en Arrière-Plan

Pour les exécutions longues, nous recommandons d'utiliser un gestionnaire de sessions :

```bash
# Linux/macOS avec screen
screen -S g05_exp
./run_experiments.sh --mode cpu_mid --run full
# Ctrl+A, puis D pour détacher
# screen -r g05_exp pour reprendre

# Windows avec Git Bash (nohup)
nohup ./run_experiments.sh --mode cpu_mid > logs/run.log 2>&1 &
```

#### Scénario 5 : Reprise d'Exécution Interrompue

Le script sauvegarde automatiquement les résultats au fur et à mesure. Si l'exécution s'interrompt :

```bash
# Relancer simplement
./run_experiments.sh --mode cpu_mid --run random_search

# Les trials déjà complétés sont détectés et non re-exécutés
```

### 6.4 Structure des Sorties

À l'issue de l'exécution, les artefacts suivants sont générés :

```
G05_AGNews/
├── results/
│   ├── logs/
│   │   ├── random_search_results_cpu_mid.csv    ← Tous les trials
│   │   ├── training_histories.json              ← Courbes complètes
│   │   ├── sharpness_scores.json                ← Sharpness par optimiseur
│   │   └── papermill_YYYYMMDD_HHMMSS.log        ← Logs d'exécution
│   ├── figures/
│   │   ├── 00_data_exploration.png              ← Exploration données
│   │   ├── fig1_convergence.png                 ← Courbes de convergence
│   │   ├── fig1b_all_trials.png                 ← Tous les trials
│   │   ├── fig2_heatmap.png                     ← Heatmap LR × Optimiseur
│   │   ├── fig2b_lr_scatter.png                 ← Scatter LR vs F1
│   │   ├── fig3_loss_landscape.png              ← Loss landscape 1D
│   │   ├── fig4_sharpness.png                   ← Sharpness vs Performance
│   │   ├── fig5_summary_table.png               ← Tableau récapitulatif
│   │   └── fig6_distributions.png               ← Boxplots distributions
│   └── models/
│       ├── best_adamw.pt                        ← Meilleur modèle AdamW
│       ├── best_sgd.pt                          ← Meilleur modèle SGD
│       ├── best_adafactor.pt                    ← Meilleur modèle Adafactor
│       └── best_overall.pt                      ← Meilleur modèle global
├── backups/
│   ├── configs/YYYYMMDD_HHMMSS/                 ← Snapshots config
│   ├── results/YYYYMMDD_HHMMSS/                 ← Copies des résultats
│   └── models/YYYYMMDD_HHMMSS/                  ← Copies des modèles
└── logs/
    ├── exec_YYYYMMDD_HHMMSS.log                 ← Logs d'exécution
    └── sync_YYYYMMDD_HHMMSS.log                 ← Logs de synchronisation
```

---

## 7. Résultats Attendus

### 7.1 Visualisations Générées

Le pipeline produit sept visualisations principales dans `results/figures/` :

| Figure | Description | Interprétation |
|---|---|---|
| `fig1_convergence.png` | Loss et F1 de validation par optimiseur | Vitesse de convergence |
| `fig1b_all_trials.png` | Tous les trials par optimiseur | Variabilité des résultats |
| `fig2_heatmap.png` | F1 final par optimiseur × LR | Sensibilité au learning rate |
| `fig2b_lr_scatter.png` | Scatter LR vs F1 | Optimum par optimiseur |
| `fig3_loss_landscape.png` | Loss landscape 1D (3 optimiseurs) | Platitude des minima |
| `fig4_sharpness.png` | Sharpness vs Performance | Corrélation généralisation |
| `fig5_summary_table.png` | Tableau récapitulatif | Comparaison directe |
| `fig6_distributions.png` | Boxplots distributions | Robustesse par optimiseur |

### 7.2 Fichiers de Données

Les fichiers suivants sont générés dans `results/logs/` pour analyse approfondie :

| Fichier | Contenu | Usage |
|---|---|---|
| `random_search_results_<mode>.csv` | Tous les trials avec métriques | Analyse statistique |
| `training_histories.json` | Historiques complets d'entraînement | Courbes de convergence |
| `sharpness_scores.json` | Sharpness par optimiseur | Analyse loss landscape |

### 7.3 Modèles Sauvegardés

Les poids des meilleurs modèles sont sauvegardés dans `results/models/` :

| Fichier | Description |
|---|---|
| `best_adamw.pt` | Meilleur modèle avec AdamW |
| `best_sgd.pt` | Meilleur modèle avec SGD |
| `best_adafactor.pt` | Meilleur modèle avec Adafactor |
| `best_overall.pt` | Meilleur modèle tous optimiseurs confondus |

### 7.4 Hypothèses à Tester

Notre expérimentation permet de tester les hypothèses suivantes :

| Hypothèse | Attendu | Métrique de Validation |
|---|---|---|
| **H1** : AdamW domine en performance | F1 le plus élevé | `final_val_f1` |
| **H2** : SGD trouve des minima plus plats | Sharpness le plus faible | `sharpness_scores.json` |
| **H3** : Adafactor est le plus rapide | Temps d'entraînement le plus court | `train_time_min` |
| **H4** : LR optimal similaire AdamW/Adafactor | Plages LR proches | `fig2_heatmap.png` |

---

## 8. Dépannage & FAQ

### 8.1 Erreurs Courantes et Solutions

#### Erreur : `ModuleNotFoundError: No module named 'src'`

**Cause** : PYTHONPATH mal configuré lors de l'exécution via papermill.

**Solution** :
```bash
# Vérifier que vous êtes dans le bon dossier
pwd
# Doit afficher : .../G05_AGNews

# Exporter PYTHONPATH manuellement
export PYTHONPATH=$(pwd):$PYTHONPATH

# Relancer le script
./run_experiments.sh --mode cpu_mid
```

#### Erreur : `Permission denied: ./run_experiments.sh`

**Cause** : Le script n'a pas les permissions d'exécution.

**Solution** :
```bash
chmod +x run_experiments.sh
./run_experiments.sh --mode cpu_mid
```

#### Erreur : `CUDA out of memory`

**Cause** : Batch size trop grand pour la mémoire GPU disponible.

**Solution** :
```bash
# Utiliser le mode cpu_low avec batch size réduit
./run_experiments.sh --mode cpu_low --backup false
```

#### Erreur : `PapermillExecutionError`

**Cause** : Erreur dans l'exécution du notebook.

**Solution** :
```bash
# Vérifier les logs détaillés
cat logs/papermill_*.log | tail -50

# Exécuter le notebook manuellement pour debug
jupyter notebook notebooks/G05_full_analysis.ipynb
```

### 8.2 Questions Fréquentes

**Q : Comment changer le nombre de trials ?**

**R** : Modifiez `src/config.py` dans la fonction `get_config()` :
```python
cfg.search.n_trials = 12  # cpu_low
cfg.search.n_trials = 20  # cpu_mid
cfg.search.n_trials = 30  # gpu
```

**Q : L'exécution est trop lente sur CPU, que faire ?**

**R** : C'est un comportement normal pour DistilBERT sur CPU. Réduisez le nombre de trials dans `src/config.py` ou utilisez le mode `cpu_low`.

**Q : Comment reprendre une exécution interrompue ?**

**R** : Le script sauvegarde automatiquement les résultats. Relancez simplement :
```bash
./run_experiments.sh --mode cpu_mid --run random_search
```

---

## 9. Références Académiques

### 9.1 Articles Fondamentaux

| Référence | Titre | Venue | Année |
|---|---|---|---|
| Bergstra & Bengio | Random Search for Hyper-Parameter Optimization | JMLR 13 | 2012 |
| Devlin et al. | BERT: Pre-training of Deep Bidirectional Transformers | NAACL | 2019 |
| Dodge et al. | Fine-Tuning Pretrained Language Models | arXiv:2002.06305 | 2020 |
| Keskar et al. | Large-Batch Training & Sharp Minima | ICLR | 2017 |
| Li et al. | Visualizing the Loss Landscape of Neural Nets | NeurIPS | 2018 |
| Loshchilov & Hutter | Decoupled Weight Decay Regularization (AdamW) | ICLR | 2019 |
| Sanh et al. | DistilBERT, a Distilled Version of BERT | NeurIPS Workshop | 2019 |
| Shazeer & Stern | Adafactor: Adaptive Learning Rates | ICML | 2018 |
| Zhang et al. | Character-level Convolutional Networks (AG News) | NeurIPS | 2015 |

### 9.2 Ressources Complémentaires

| Ressource | Lien |
|---|---|
| **Python 3.11.5** | [python.org/downloads](https://www.python.org/downloads/release/python-3115/) |
| **PyTorch** | [pytorch.org](https://pytorch.org/) |
| **HuggingFace Transformers** | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) |
| **AG News Dataset** | [huggingface.co/datasets/ag_news](https://huggingface.co/datasets/ag_news) |
| **Git Bash (Windows)** | [gitforwindows.org](https://gitforwindows.org/) |

---

## 10. Informations du Projet

### 10.1 Contexte Académique

| Information | Détail |
|---|---|
| **Cours** | Fine-tuning de Transformers |
| **Enseignante** | MBIA NDI Marie Thérèse |
| **Groupe** | G05 |
| **Dataset** | D03 (AG News) |
| **Modèle** | M01 (DistilBERT) |
| **Problématique** | P01 (Benchmark Optimiseurs) |
| **Méthode** | Random Search + Loss Landscape |
| **Date Limite** | 13 mars 2026 |

### 10.2 Versionnement

| Élément | Version |
|---|---|
| **Script** | 2.0 (avec support complet run_experiments.sh) |
| **Python** | 3.11.5 (compatible 3.11.x) |
| **Dernière Mise à Jour** | 10 mars 2026 |

### 10.3 Licence

Ce projet est distribué sous licence MIT. Consultez le fichier [LICENSE](LICENSE) pour plus de détails.

---

## ✅ Checklist de Démarrage Rapide

Pour une installation et exécution sans erreur, suivez cette checklist :

```bash
# □ 1. Vérifier Python 3.11.5
python --version

# □ 2. Créer et activer l'environnement virtuel
python -m venv venv
# Windows : .\venv\Scripts\Activate.ps1
# Linux/macOS : source venv/bin/activate

# □ 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# □ 4. Vérifier l'installation
python -c "import torch, transformers; print('✅ OK')"

# □ 5. Rendre le script exécutable
chmod +x run_experiments.sh

# □ 6. Lancer un test rapide
./run_experiments.sh --mode cpu_low --run setup

# □ 7. Lancer l'exécution complète
./run_experiments.sh --mode cpu_mid --run full
```

---

> 💡 **Besoin d'aide ?** Vérifiez d'abord la section [Dépannage & FAQ](#8-dépannage--faq). Si le problème persiste, consultez les logs dans `logs/papermill_*.log` ou ouvrez une issue sur le dépôt du projet.


---

## 📞 Support & Contact

| Rôle | Nom |
|---|---|
| **Cours** | Fine-tuning de Transformers |
| **Enseignante** | MBIA NDI Marie Thérèse |
| **Groupe** | G05 |
| **Date limite** | 13 mars 2026 |

---

*Dernière mise à jour : 10 mars 2026*  
*Version du script : 2.0 (avec support complet run_experiments.sh)*