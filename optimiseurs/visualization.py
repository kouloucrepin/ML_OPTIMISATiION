"""
visualization.py — Génération de toutes les figures du projet
G05 | P01

Figures produites :
  Fig 1 — Courbes de convergence (loss train) par optimiseur
  Fig 2 — Accuracy finale : heatmap optimiseur × learning rate
  Fig 3 — Loss landscape 1D pour les 3 meilleurs modèles
  Fig 4 — Sharpness scores par optimiseur

Références :
- Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets"
  NeurIPS → Méthode de perturbation directionnelle normalisée
- Keskar et al. (2017) "On Large-Batch Training for Deep Learning:
  Generalization Gap and Sharp Minima"
  → Minima plats généralisent mieux → Sharpness metric
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import Config, get_config


# ─────────────────────────────────────────────
# STYLE GLOBAL
# ─────────────────────────────────────────────

OPTIMIZER_COLORS = {
    "adamw":    "#2563EB",   # Bleu
    "sgd":      "#DC2626",   # Rouge
    "adafactor": "#16A34A",  # Vert
}

OPTIMIZER_LABELS = {
    "adamw":    "AdamW",
    "sgd":      "SGD + Nesterov",
    "adafactor": "Adafactor",
}

def setup_style():
    plt.rcParams.update({
        "figure.dpi":      150,
        "font.family":     "DejaVu Sans",
        "font.size":       11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid":       True,
        "grid.alpha":      0.3,
        "grid.linestyle":  "--",
    })


# ─────────────────────────────────────────────
# FIGURE 1 — COURBES DE CONVERGENCE
# ─────────────────────────────────────────────

def plot_convergence_curves(histories: List[Dict], output_dir: str):
    """
    Courbes de loss d'entraînement pour les meilleurs runs de chaque optimiseur.
    Permet de visualiser la vitesse et la stabilité de convergence.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1 — Courbes de Convergence par Optimiseur\n"
                 "AG News | DistilBERT | G05", fontsize=13, fontweight="bold")

    # Sélection du meilleur run par optimiseur (sur val_f1)
    best_by_opt: Dict[str, Dict] = {}
    for h in histories:
        opt = h["optimizer"]
        val_hist = h.get("val_metrics_history", [])
        if not val_hist:
            continue
        best_f1 = max(v["f1"] for v in val_hist)
        if opt not in best_by_opt or best_f1 > best_by_opt[opt]["best_f1"]:
            best_by_opt[opt] = {**h, "best_f1": best_f1}

    # Axe gauche : train loss
    ax1 = axes[0]
    ax1.set_title("Loss d'Entraînement")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")

    for opt, data in best_by_opt.items():
        losses = data.get("train_loss_history", [])
        if losses:
            steps = list(range(1, len(losses) + 1))
            color = OPTIMIZER_COLORS.get(opt, "gray")
            label = OPTIMIZER_LABELS.get(opt, opt)
            ax1.plot(steps, losses, color=color, linewidth=2, label=label, alpha=0.9)
            # Lissage mobile pour la lisibilité
            if len(losses) > 5:
                window = 5
                smooth = pd.Series(losses).rolling(window, min_periods=1).mean().values
                ax1.plot(steps, smooth, color=color, linewidth=1, linestyle="--", alpha=0.5)

    ax1.legend(framealpha=0.9)

    # Axe droit : val F1 par step
    ax2 = axes[1]
    ax2.set_title("F1 de Validation")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("F1 Macro")

    for opt, data in best_by_opt.items():
        val_hist = data.get("val_metrics_history", [])
        if val_hist:
            steps = [v["step"] for v in val_hist]
            f1s   = [v["f1"]   for v in val_hist]
            color = OPTIMIZER_COLORS.get(opt, "gray")
            label = OPTIMIZER_LABELS.get(opt, opt)
            ax2.plot(steps, f1s, color=color, linewidth=2.5, marker="o",
                     markersize=6, label=label)

    ax2.legend(framealpha=0.9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig1_convergence_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Fig 1 sauvegardée → {out_path}")


# ─────────────────────────────────────────────
# FIGURE 2 — HEATMAP ACCURACY
# ─────────────────────────────────────────────

def plot_accuracy_heatmap(results_csv: str, output_dir: str):
    """
    Heatmap F1 final : optimiseur (axe Y) × learning rate rangé par ordre (axe X).
    Inspiré des figures de Bergstra & Bengio (2012).
    """
    setup_style()

    if not os.path.exists(results_csv):
        print(f"[VIZ] CSV non trouvé : {results_csv} — Fig 2 ignorée")
        return

    df = pd.read_csv(results_csv)
    if df.empty:
        return

    # Buckets de LR pour l'axe X
    df["lr_log"] = np.log10(df["lr"].astype(float))
    df["lr_bucket"] = pd.cut(df["lr_log"], bins=5, labels=False)

    pivot = df.pivot_table(
        values="final_val_f1",
        index="optimizer",
        columns="lr_bucket",
        aggfunc="max"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Figure 2 — F1 Validation : Optimiseur × Plage de Learning Rate\n"
                 "AG News | DistilBERT | G05", fontsize=13, fontweight="bold")

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True, fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "F1 Macro (max par bucket)"},
    )

    ax.set_xlabel("Bucket Learning Rate (LR croissant →)", labelpad=10)
    ax.set_ylabel("Optimiseur")
    ax.set_yticklabels([OPTIMIZER_LABELS.get(l.get_text(), l.get_text())
                        for l in ax.get_yticklabels()], rotation=0)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2_accuracy_heatmap.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Fig 2 sauvegardée → {out_path}")


# ─────────────────────────────────────────────
# LOSS LANDSCAPE
# ─────────────────────────────────────────────

def evaluate_on_subset(model: nn.Module, dataloader: DataLoader, device: torch.device, n_samples: int = 50) -> float:
    """
    Évalue la loss sur un petit sous-ensemble pour le loss landscape.
    Économie de calcul sur CPU.
    """
    model.eval()
    total_loss = 0.0
    n_seen = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_seen >= n_samples:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            batch_size = input_ids.size(0)
            remaining = n_samples - n_seen
            if batch_size > remaining:
                input_ids      = input_ids[:remaining]
                attention_mask = attention_mask[:remaining]
                labels         = labels[:remaining]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * min(batch_size, remaining)
            n_seen += min(batch_size, remaining)

    return total_loss / max(n_seen, 1)


def compute_loss_landscape_1d(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: Config,
    direction_seed: int = 0,
) -> tuple:
    """
    Calcule le loss landscape 1D par perturbation directionnelle normalisée.

    Méthode de Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets" :
    1. Choisir une direction aléatoire dans l'espace des paramètres
    2. Normaliser par filtre (filter normalization) → invariance à l'échelle
    3. Évaluer la loss en se déplaçant le long de cette direction

    La filter normalization (normalization par couche) est critique pour
    comparer des modèles de tailles différentes.

    Args:
        model       : modèle entraîné
        dataloader  : DataLoader de validation
        cfg         : configuration (landscape)
        direction_seed : graine pour la direction aléatoire

    Returns:
        alphas (positions), losses (valeurs)
    """
    device = torch.device(cfg.project.device)
    lc = cfg.landscape

    torch.manual_seed(direction_seed)
    model.eval()

    # Sauvegarde des paramètres originaux
    original_params = [p.clone().detach() for p in model.parameters()]

    # Direction aléatoire normalisée (filter normalization)
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        # Filter normalization : normalise par norme du paramètre correspondant
        # Li et al. (2018) eq. (4)
        if p.dim() > 1:
            param_norm = p.norm(dim=list(range(1, p.dim())), keepdim=True).clamp(min=1e-8)
            d_norm = d.norm(dim=list(range(1, d.dim())), keepdim=True).clamp(min=1e-8)
            d = d * (param_norm / d_norm)
        direction.append(d)

    # Grille de perturbation
    alphas = np.linspace(-lc.epsilon, lc.epsilon, lc.n_points)
    losses = []

    print(f"   [Landscape] Calcul sur {lc.n_points} points × {lc.n_samples_eval} samples...")

    for alpha in alphas:
        # Appliquer perturbation θ' = θ + α·d
        with torch.no_grad():
            for p, p0, d in zip(model.parameters(), original_params, direction):
                p.data = p0 + alpha * d

        loss = evaluate_on_subset(model, dataloader, device, n_samples=lc.n_samples_eval)
        losses.append(loss)

    # Restaurer les paramètres originaux
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), original_params):
            p.data = p0.clone()

    return alphas, losses


def compute_sharpness(alphas: np.ndarray, losses: list, loss_at_center: float) -> float:
    """
    Calcule la Sharpness selon la formule du document :
    Sharpness = (1/N) Σ |L(θ + ε·di) - L(θ)|

    Keskar et al. (2017) : un Sharpness élevé indique un minimum pointu
    qui généralise moins bien qu'un minimum plat.

    Args:
        alphas         : positions de perturbation
        losses         : losses correspondantes
        loss_at_center : loss au point central (θ original)

    Returns:
        sharpness : float (plus petit = meilleur)
    """
    deviations = [abs(l - loss_at_center) for l in losses]
    return float(np.mean(deviations))


# ─────────────────────────────────────────────
# FIGURE 3 — LOSS LANDSCAPE
# ─────────────────────────────────────────────

def plot_loss_landscape(cfg: Config, val_dataset, output_dir: str):
    """
    Loss landscape 1D pour les 3 meilleurs modèles (un par optimiseur).
    """
    from src.model_setup import load_fresh_model

    setup_style()
    device = torch.device(cfg.project.device)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Figure 3 — Loss Landscape 1D par Optimiseur\n"
                 "Méthode : Li et al. (2018) | AG News | DistilBERT | G05",
                 fontsize=13, fontweight="bold")

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    sharpness_results = {}

    for opt in cfg.search.optimizers:
        model_path = os.path.join(cfg.project.models_dir, f"best_{opt}.pt")
        if not os.path.exists(model_path):
            print(f"[VIZ] Modèle {opt} non trouvé ({model_path}) — ignoré")
            continue

        print(f"[VIZ] Landscape pour {opt}...")
        model = load_fresh_model(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))

        alphas, losses = compute_loss_landscape_1d(model, val_loader, cfg)

        # Loss au centre (alpha=0) → position médiane
        center_idx = len(losses) // 2
        loss_center = losses[center_idx]
        sharpness = compute_sharpness(alphas, losses, loss_center)
        sharpness_results[opt] = sharpness

        color = OPTIMIZER_COLORS.get(opt, "gray")
        label_str = f"{OPTIMIZER_LABELS.get(opt, opt)} (S={sharpness:.4f})"
        ax.plot(alphas, losses, color=color, linewidth=2.5, marker="o",
                markersize=5, label=label_str)

    ax.set_xlabel("Direction de perturbation (α)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.axvline(x=0, color="black", linestyle=":", linewidth=1.5, alpha=0.5, label="θ original")
    ax.legend(fontsize=10, framealpha=0.95)
    ax.annotate("Minimum plat = meilleure généralisation",
                xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3_loss_landscape.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Fig 3 sauvegardée → {out_path}")

    # Sauvegarde des sharpness
    sharpness_path = os.path.join(output_dir, "..", "logs", "sharpness_scores.json")
    with open(sharpness_path, "w") as f:
        json.dump(sharpness_results, f, indent=2)

    return sharpness_results


# ─────────────────────────────────────────────
# FIGURE 4 — SHARPNESS COMPARAISON
# ─────────────────────────────────────────────

def plot_sharpness_comparison(sharpness_results: Dict, results_csv: str, output_dir: str):
    """
    Figure 4 : Sharpness vs Performance finale.
    Un bon modèle doit avoir sharpness faible ET F1 élevé.
    """
    setup_style()

    if not sharpness_results:
        print("[VIZ] Aucun sharpness calculé — Fig 4 ignorée")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 4 — Analyse Sharpness vs Performance\n"
                 "Keskar et al. (2017) | AG News | DistilBERT | G05",
                 fontsize=13, fontweight="bold")

    # Axe gauche : barplot sharpness
    ax1 = axes[0]
    opts   = list(sharpness_results.keys())
    sharps = [sharpness_results[o] for o in opts]
    colors = [OPTIMIZER_COLORS.get(o, "gray") for o in opts]
    labels = [OPTIMIZER_LABELS.get(o, o) for o in opts]

    bars = ax1.bar(labels, sharps, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    ax1.set_title("Sharpness par Optimiseur\n(plus bas = minimum plus plat)", fontsize=11)
    ax1.set_ylabel("Sharpness")
    ax1.set_ylim(0, max(sharps) * 1.3 if sharps else 1)

    for bar, val in zip(bars, sharps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Axe droit : scatter sharpness vs F1
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        ax2 = axes[1]
        ax2.set_title("F1 Final vs Sharpness\n(idéal = haut gauche)", fontsize=11)
        ax2.set_xlabel("Sharpness")
        ax2.set_ylabel("F1 Macro (validation)")

        for opt in opts:
            opt_df = df[df["optimizer"] == opt]
            if len(opt_df) == 0:
                continue
            color = OPTIMIZER_COLORS.get(opt, "gray")
            label = OPTIMIZER_LABELS.get(opt, opt)

            # Tous les points de cet optimiseur
            ax2.scatter(
                [sharpness_results[opt]] * len(opt_df),
                opt_df["final_val_f1"],
                color=color, label=label, alpha=0.7, s=80, zorder=5
            )

        ax2.legend(fontsize=9)

        # Zone idéale
        ax2.annotate("Zone idéale →", xy=(0.02, 0.85), xycoords="axes fraction",
                     fontsize=9, color="green", style="italic")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig4_sharpness_analysis.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Fig 4 sauvegardée → {out_path}")


# ─────────────────────────────────────────────
# RÉSUMÉ TABLEAU
# ─────────────────────────────────────────────

def plot_summary_table(results_csv: str, output_dir: str):
    """Tableau récapitulatif des meilleurs résultats — Fig 5 bonus."""
    if not os.path.exists(results_csv):
        return

    setup_style()
    df = pd.read_csv(results_csv)
    if df.empty:
        return

    summary = df.groupby("optimizer").agg(
        Best_F1=("final_val_f1", "max"),
        Mean_F1=("final_val_f1", "mean"),
        Best_Acc=("final_val_accuracy", "max"),
        N_Trials=("run_id", "count"),
        Best_LR=("lr", lambda x: df.loc[x.idxmax(), "lr"] if len(x) > 0 else ""),
    ).round(4).reset_index()
    summary["optimizer"] = summary["optimizer"].map(lambda x: OPTIMIZER_LABELS.get(x, x))

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.suptitle("Figure 5 — Tableau Récapitulatif des Résultats\nG05 | AG News | DistilBERT",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Colorisation des headers
    for j in range(len(summary.columns)):
        table[0, j].set_facecolor("#2563EB")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig5_summary_table.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Fig 5 sauvegardée → {out_path}")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

def generate_all_figures(cfg: Config, results_csv: str, output_dir: str):
    """Lance la génération de toutes les figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Chargement des historiques
    histories_path = os.path.join(cfg.project.logs_dir, "training_histories.json")
    histories = []
    if os.path.exists(histories_path):
        with open(histories_path) as f:
            histories = json.load(f)

    # Fig 1 — Convergence
    if histories:
        plot_convergence_curves(histories, output_dir)

    # Fig 2 — Heatmap
    plot_accuracy_heatmap(results_csv, output_dir)

    # Fig 3 & 4 — Landscape + Sharpness
    from src.data_loader import prepare_data
    _, val_dataset, _ = prepare_data(cfg)
    sharpness = plot_loss_landscape(cfg, val_dataset, output_dir)
    plot_sharpness_comparison(sharpness, results_csv, output_dir)

    # Fig 5 — Tableau récap
    plot_summary_table(results_csv, output_dir)

    print(f"\n[VIZ] ✓ Toutes les figures générées dans {output_dir}/")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation — G05 P01")
    parser.add_argument("--mode",         type=str, default="cpu_mid",
                        choices=["cpu_low", "cpu_mid", "gpu"])
    parser.add_argument("--task",         type=str, default="all",
                        choices=["all", "convergence", "heatmap", "landscape", "sharpness"],
                        help="Figure(s) à générer")
    parser.add_argument("--results_csv",  type=str, default="results/logs/random_search_results.csv")
    parser.add_argument("--models_dir",   type=str, default="results/models")
    parser.add_argument("--output_dir",   type=str, default="results/figures")
    args = parser.parse_args()

    cfg = get_config(args.mode)

    if args.task == "all":
        generate_all_figures(cfg, args.results_csv, args.output_dir)
    elif args.task == "convergence":
        histories_path = os.path.join(cfg.project.logs_dir, "training_histories.json")
        with open(histories_path) as f:
            histories = json.load(f)
        plot_convergence_curves(histories, args.output_dir)
    elif args.task == "heatmap":
        plot_accuracy_heatmap(args.results_csv, args.output_dir)
    elif args.task in ("landscape", "sharpness"):
        from src.data_loader import prepare_data
        _, val_dataset, _ = prepare_data(cfg)
        sharpness = plot_loss_landscape(cfg, val_dataset, args.output_dir)
        if args.task == "sharpness":
            plot_sharpness_comparison(sharpness, args.results_csv, args.output_dir)
