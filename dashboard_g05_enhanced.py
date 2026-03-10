"""
Dashboard Dash — G05 Enhanced | AG News | DistilBERT | P01 Benchmark Optimiseurs
Version Corrigée — Bugs fixes :
  ✅ create_radar_chart correctement définie
  ✅ Conflit 'legend' dans PLOTLY_BASE résolu
  ✅ fillcolor hex+alpha remplacé par rgba valide
  ✅ Architecture modulaire propre

Lancement :
  pip install dash plotly pandas numpy scipy
  python dashboard_g05_enhanced.py
  → http://127.0.0.1:8050
"""

import os, json, warnings
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from scipy import stats
from functools import lru_cache

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 🎨 CONFIGURATION VISUELLE
# ══════════════════════════════════════════════════════════════════════════════

class Theme:
    bg           = "#F8F9FC"
    surface      = "#FFFFFF"
    surface2     = "#F1F4F9"
    border       = "#E2E8F0"
    border2      = "#CBD5E1"
    text         = "#1E293B"
    text_light   = "#475569"
    muted        = "#64748B"
    faint        = "#94A3B8"

    adamw        = "#2563EB"
    sgd          = "#DC2626"
    adafactor    = "#16A34A"

    # ✅ FIX 3 : fillcolor rgba valides (remplace "#hex30" invalide)
    adamw_fill     = "rgba(37, 99, 235, 0.15)"
    sgd_fill       = "rgba(220, 38, 38, 0.15)"
    adafactor_fill = "rgba(22, 163, 74, 0.15)"

    accent       = "#4F46E5"
    accent_bg    = "#EEF2FF"
    success      = "#16A34A"
    success_bg   = "#F0FDF4"
    warning      = "#F59E0B"
    warning_bg   = "#FEF3C7"
    info         = "#0EA5E9"
    info_bg      = "#E0F2FE"

C = Theme

LABELS = {
    "adamw":     "AdamW",
    "sgd":       "SGD + Nesterov",
    "adafactor": "Adafactor",
}

FONT = "'DM Sans', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "g05_graph",
        "height": 1080,
        "width": 1920,
        "scale": 2
    }
}

# ✅ PLOTLY_BASE : uniquement les clés qui ne conflictent JAMAIS
# xaxis / yaxis / legend sont exclus → appliqués séparément selon besoin
PLOTLY_BASE = dict(
    paper_bgcolor=C.surface,
    plot_bgcolor=C.surface,
    font=dict(family=FONT, color=C.text, size=12),
    margin=dict(l=60, r=35, t=65, b=50),
    hovermode="closest",
)

# Styles réutilisables appliqués via update_xaxes / update_yaxes
AXIS_STYLE = dict(
    gridcolor=C.border,
    zerolinecolor=C.border2,
    linecolor=C.border2,
    tickfont=dict(size=11, color=C.muted),
    title_font=dict(size=12, color=C.muted),
)

# Legend par défaut (horizontal, en haut)
LEGEND_DEFAULT = dict(
    bgcolor=C.surface,
    bordercolor=C.border,
    borderwidth=1,
    font=dict(size=11),
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1,
)

# Legend latérale (pour subplots)
LEGEND_SIDE = dict(
    bgcolor=C.surface,
    bordercolor=C.border,
    borderwidth=1,
    font=dict(size=11),
    x=1.05,
    y=0.5,
)

def apply_axes(fig):
    """Applique le style d'axes sur toutes les axes d'une figure."""
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig

def plotly_base_no_legend():
    """Rétrocompatibilité — PLOTLY_BASE est déjà sans legend."""
    return PLOTLY_BASE

# ══════════════════════════════════════════════════════════════════════════════
# 📊 CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_csv_cached():
    patterns = [
        "results/logs/random_search_results_*.csv",
        "results/logs/random_search_results.csv"
    ]
    for p in patterns:
        files = glob.glob(p)
        if files:
            df = pd.read_csv(sorted(files)[-1])
            df["opt_label"] = df["optimizer"].map(lambda x: LABELS.get(x, x))
            df["lr_log"] = np.log10(df["lr"].astype(float))
            df["lr_bucket"] = pd.cut(
                df["lr_log"], bins=5,
                labels=["1e-6→1e-5", "1e-5→5e-5", "5e-5→1e-4", "1e-4→2e-4", "2e-4→5e-4"]
            )
            return df
    return None

@lru_cache(maxsize=4)
def load_json_cached(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

df_raw            = load_csv_cached()
histories         = load_json_cached("results/logs/training_histories.json") or []
sharpness         = load_json_cached("results/logs/sharpness_scores.json") or {}
landscape         = load_json_cached("results/logs/landscape_results.json") or {}
undersampling_stats = load_json_cached("results/logs/undersampling_stats.json") or {}

HAS_CSV   = df_raw is not None
HAS_HIST  = len(histories) > 0
HAS_SHARP = len(sharpness) > 0
HAS_LAND  = len(landscape) > 0
HAS_UNDER = len(undersampling_stats) > 0

# ══════════════════════════════════════════════════════════════════════════════
# 🧩 COMPOSANTS UI
# ══════════════════════════════════════════════════════════════════════════════

def section_header(title, subtitle=None, icon=None):
    return html.Div([
        html.Div([
            html.Span(icon or "📊", style={"fontSize": "20px", "marginRight": "12px"}),
            html.H3(title, style={
                "color": C.text, "fontFamily": FONT,
                "fontSize": "16px", "fontWeight": "700",
                "margin": "0", "display": "inline-block",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
        html.P(subtitle or "", style={
            "color": C.muted, "fontFamily": FONT,
            "fontSize": "13px", "margin": "0 0 24px 32px",
        }) if subtitle else html.Div(style={"marginBottom": "24px"}),
    ])

def kpi_card(label, value, sub=None, color=None, icon=None, trend=None):
    return html.Div([
        html.Div([
            html.Span((icon or "📈") + "  ", style={"fontSize": "18px", "opacity": "0.8"}),
            html.Span(label, style={
                "color": C.muted, "fontSize": "11px", "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "0.08em",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Div([
            html.Div(value, style={
                "color": color or C.text, "fontSize": "28px",
                "fontWeight": "800", "fontFamily": FONT, "lineHeight": "1",
            }),
            html.Span(trend or "", style={
                "color": C.success if trend and "↑" in trend else C.muted,
                "fontSize": "14px", "marginLeft": "8px", "fontWeight": "600",
            }) if trend else None,
        ], style={"display": "flex", "alignItems": "baseline", "marginBottom": "6px"}),
        html.Div(sub or "", style={
            "color": C.faint, "fontSize": "11px",
            "fontFamily": FONT, "fontWeight": "500",
        }),
    ], style={
        "background": C.surface,
        "border": f"1px solid {C.border}",
        "borderRadius": "16px",
        "padding": "24px 28px",
        "flex": "1",
        "minWidth": "180px",
        "borderTop": f"4px solid {color or C.accent}",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
        "transition": "all 0.3s ease",
        "cursor": "default",
    }, className="kpi-card")

def status_badge(text, status="success"):
    colors = {
        "success": (C.success_bg, C.success, "#86efac"),
        "warning": (C.warning_bg, C.warning, "#fde047"),
        "info":    (C.info_bg,    C.info,    "#7dd3fc"),
    }
    bg, fg, border_c = colors.get(status, colors["info"])
    return html.Span(text, style={
        "background": bg, "color": fg,
        "border": f"1px solid {border_c}",
        "borderRadius": "20px", "padding": "4px 12px",
        "fontSize": "11px", "fontWeight": "600",
        "fontFamily": FONT, "marginRight": "8px",
        "display": "inline-block",
    })

def radio_group(id_, options, value, label=None):
    return html.Div([
        html.Label(label or "Options : ", style={
            "color": C.muted, "fontFamily": FONT, "fontSize": "12px",
            "fontWeight": "600", "marginRight": "12px",
            "textTransform": "uppercase", "letterSpacing": "0.05em",
        }),
        dcc.RadioItems(
            id=id_, options=options, value=value, inline=True,
            style={"fontSize": "13px", "fontFamily": FONT},
            inputStyle={"marginRight": "6px", "marginLeft": "16px", "accentColor": C.accent},
            labelStyle={"color": C.text, "cursor": "pointer", "fontWeight": "500"},
        )
    ], style={
        "display": "flex", "alignItems": "center",
        "marginBottom": "20px", "padding": "12px 16px",
        "background": C.surface2, "borderRadius": "10px",
        "border": f"1px solid {C.border}",
    })

def info_callout(text, type_="info", icon=None):
    configs = {
        "info":    (C.info_bg,    C.info,    icon or "ℹ️"),
        "success": (C.success_bg, C.success, icon or "✅"),
        "warning": (C.warning_bg, C.warning, icon or "⚠️"),
    }
    bg, border_c, emoji = configs.get(type_, configs["info"])
    return html.Div([
        html.Span(emoji + "  ", style={"fontSize": "16px"}),
        html.Span(text, style={
            "color": C.text_light, "fontFamily": FONT,
            "fontSize": "13px", "lineHeight": "1.6",
        }),
    ], style={
        "display": "flex", "alignItems": "flex-start",
        "background": bg, "borderRadius": "12px",
        "padding": "16px 20px", "marginTop": "20px",
        "border": f"1px solid {border_c}55",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
    })

def empty_state(title="Données non disponibles", message=None, icon="📭"):
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{icon} {title}</b><br>"
             f"<span style='font-size:13px; color:{C.muted}'>"
             f"{message or 'Lance le Random Search pour voir les résultats'}</span>",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=C.text, family=FONT),
        align="center",
    )
    fig.update_layout(**PLOTLY_BASE, height=360)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 📈 FONCTIONS DE VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def create_scatter_plot(metric="final_val_f1", show_trend=False):
    if not HAS_CSV:
        return empty_state("CSV manquant")
    df = df_raw
    ylabel = "F1 Macro" if "f1" in metric else "Accuracy"
    fig = go.Figure()

    for opt in ["adamw", "sgd", "adafactor"]:
        sub = df[df["optimizer"] == opt]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["lr"], y=sub[metric], mode="markers",
            name=LABELS[opt],
            marker=dict(color=getattr(C, opt), size=11, opacity=0.75,
                        line=dict(color="white", width=1.8)),
            customdata=sub[["lr", "batch_size", "warmup_steps", metric]].values,
            hovertemplate=(
                f"<b>{LABELS[opt]}</b><br>LR: %{{customdata[0]:.2e}}<br>"
                "Batch: %{customdata[1]:.0f}<br>Warmup: %{customdata[2]:.0f}<br>"
                f"{ylabel}: %{{customdata[3]:.4f}}<extra></extra>"
            ),
        ))
        best = sub.loc[sub[metric].idxmax()]
        fig.add_trace(go.Scatter(
            x=[best["lr"]], y=[best[metric]], mode="markers",
            marker=dict(color=getattr(C, opt), size=24, symbol="star",
                        line=dict(color="white", width=2.5)),
            showlegend=False,
            hovertemplate=f"<b>⭐ Meilleur {LABELS[opt]}</b><br>LR: {best['lr']:.2e}<br>{ylabel}: {best[metric]:.4f}<extra></extra>",
        ))
        if show_trend and len(sub) > 3:
            log_lr = np.log10(sub["lr"])
            z = np.polyfit(log_lr, sub[metric], 1)
            p = np.poly1d(z)
            lr_range = np.logspace(log_lr.min(), log_lr.max(), 100)
            fig.add_trace(go.Scatter(
                x=lr_range, y=p(np.log10(lr_range)), mode="lines",
                line=dict(color=getattr(C, opt), width=2, dash="dot"),
                showlegend=False, hoverinfo="skip", opacity=0.4
            ))

    fig.update_layout(
        **PLOTLY_BASE,
        xaxis_type="log",
        xaxis_title="Learning Rate (échelle log)",
        yaxis_title=ylabel,
        title=dict(
            text=f"<b>{ylabel} vs Learning Rate</b><br>"
                 "<sub>⭐ = meilleur run | Tous les trials</sub>",
            font=dict(size=15, color=C.text)
        ),
        height=480,
        legend=LEGEND_DEFAULT,
    )
    apply_axes(fig)
    return fig


def create_heatmap(metric="final_val_f1"):
    if not HAS_CSV:
        return empty_state()
    df = df_raw
    ylabel = "F1 Macro" if "f1" in metric else "Accuracy"
    pivot = df.pivot_table(values=metric, index="optimizer",
                           columns="lr_bucket", aggfunc="max")
    pivot.index = [LABELS.get(i, i) for i in pivot.index]
    colorscale = "Blues" if "accuracy" in metric else [[0, "#FFF7ED"], [0.5, "#FB923C"], [1, "#9A3412"]]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=list(pivot.index),
        colorscale=colorscale,
        text=[[f"{v:.4f}" if not np.isnan(v) else "—" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=13, family=FONT, color=C.text),
        hovertemplate="<b>%{y}</b><br>LR: %{x}<br>Max: %{z:.4f}<extra></extra>",
        zmin=0, zmax=1,
        colorbar=dict(tickfont=dict(size=11), outlinewidth=0, len=0.7,
                      title=dict(text=ylabel, font=dict(size=11))),
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        xaxis_title="Plage Learning Rate",
        yaxis_title="",
        title=dict(
            text=f"<b>Heatmap {ylabel}</b><br>"
                 "<sub>Optimiseur × Plage LR (max par cellule)</sub>",
            font=dict(size=15, color=C.text)
        ),
        height=340,
        legend=LEGEND_DEFAULT,
    )
    apply_axes(fig)
    return fig


def create_convergence_plot(show_all=False):
    if not HAS_HIST:
        return empty_state("Historiques manquants", "training_histories.json introuvable")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Train Loss (lissage mobile)", "Val F1 Macro"],
        horizontal_spacing=0.12
    )
    best_by_opt = {}
    for h in histories:
        opt   = h["optimizer"]
        vhist = h.get("val_metrics_history", [])
        if not vhist:
            continue
        bf1 = max(v["f1"] for v in vhist)
        if opt not in best_by_opt or bf1 > best_by_opt[opt]["best_f1"]:
            best_by_opt[opt] = {**h, "best_f1": bf1}

    if show_all:
        for h in histories:
            opt   = h["optimizer"]
            vhist = h.get("val_metrics_history", [])
            if not vhist:
                continue
            fig.add_trace(go.Scatter(
                x=[v["step"] for v in vhist], y=[v["f1"] for v in vhist],
                mode="lines", line=dict(color=getattr(C, opt, "gray"), width=0.8),
                opacity=0.12, showlegend=False, hoverinfo="skip",
            ), row=1, col=2)

    for opt, data in best_by_opt.items():
        color = getattr(C, opt, "gray")
        label = LABELS.get(opt, opt)
        losses = data.get("train_loss_history", [])
        if losses:
            smooth = pd.Series(losses).rolling(8, min_periods=1).mean().values
            fig.add_trace(go.Scatter(
                x=list(range(1, len(losses) + 1)), y=smooth,
                mode="lines", name=label,
                line=dict(color=color, width=3),
                hovertemplate=f"{label}<br>Step: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>",
            ), row=1, col=1)
        vhist = data.get("val_metrics_history", [])
        if vhist:
            fig.add_trace(go.Scatter(
                x=[v["step"] for v in vhist], y=[v["f1"] for v in vhist],
                mode="lines+markers", name=label,
                line=dict(color=color, width=3),
                marker=dict(size=8, color=color, line=dict(color="white", width=1.8)),
                showlegend=False,
                hovertemplate=f"{label}<br>Step: %{{x}}<br>F1: %{{y:.4f}}<extra></extra>",
            ), row=1, col=2)

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text="<b>Courbes de Convergence</b><br><sub>Meilleur run par optimiseur</sub>",
            font=dict(size=15, color=C.text)
        ),
        height=460,
    )
    fig.update_xaxes(gridcolor=C.border, linecolor=C.border2,
                     tickfont=dict(size=11, color=C.muted), title_text="Steps")
    fig.update_yaxes(gridcolor=C.border, linecolor=C.border2,
                     tickfont=dict(size=11, color=C.muted))
    for ann in fig.layout.annotations:
        ann.font.update(size=13, color=C.text)
    return fig


def create_landscape_plot():
    if not HAS_LAND:
        return empty_state("Landscape non calculé", "Section 10 du notebook requise")

    fig = go.Figure()
    for opt, data in landscape.items():
        sharp = data.get("sharpness", 0)
        fig.add_trace(go.Scatter(
            x=data["alphas"], y=data["losses"],
            mode="lines+markers",
            name=f"{LABELS.get(opt, opt)} (S={sharp:.4f})",
            line=dict(color=getattr(C, opt, "gray"), width=3),
            marker=dict(size=7, color=getattr(C, opt, "gray"),
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{LABELS.get(opt, opt)}</b><br>α: %{{x:.3f}}<br>Loss: %{{y:.4f}}<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color=C.faint, line_width=2,
                  annotation_text="θ* (α=0)",
                  annotation_font=dict(size=12, color=C.muted),
                  annotation_position="top")
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text="<b>Loss Landscape 1D</b><br><sub>Li et al. (2018) Filter Normalization</sub>",
            font=dict(size=15, color=C.text)
        ),
        xaxis_title="Direction de perturbation (α)", yaxis_title="Loss", height=460,
    )
    return fig


def create_sharpness_plot():
    if not HAS_SHARP:
        return empty_state("Sharpness non calculée", "Section 11 du notebook requise")

    opts        = list(sharpness.keys())
    vals        = [sharpness[o] for o in opts]
    labels_list = [LABELS.get(o, o) for o in opts]
    colors_list = [getattr(C, o, "gray") for o in opts]

    fig = go.Figure(go.Bar(
        x=labels_list, y=vals,
        marker_color=colors_list,
        marker_line_color="white", marker_line_width=2, marker_opacity=0.9,
        text=[f"{v:.5f}" for v in vals], textposition="outside",
        textfont=dict(family=FONT, size=13, color=C.text),
        hovertemplate="<b>%{x}</b><br>Sharpness: %{y:.5f}<extra></extra>",
    ))
    best_idx = int(np.argmin(vals))
    fig.add_annotation(
        x=labels_list[best_idx], y=vals[best_idx] * 0.5,
        text="<b>Minimum<br>le plus plat</b>",
        showarrow=False,
        font=dict(color=C.success, size=11, family=FONT),
        bgcolor=C.success_bg, bordercolor=C.success, borderwidth=1.5,
        borderpad=8,
    )
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text="<b>Sharpness par Optimiseur</b><br><sub>Keskar et al. (2017)</sub>",
            font=dict(size=15, color=C.text)
        ),
        yaxis_title="Sharpness", height=380,
    )
    return fig


# ✅ FIX 2 : plotly_base_no_legend() pour éviter le conflit 'legend'
def create_sharpness_vs_performance():
    if not HAS_SHARP or not HAS_CSV:
        return empty_state("Données incomplètes", "CSV et sharpness requis")

    df = df_raw
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Sharpness par Optimiseur<br><sub>(plus bas = minimum plus plat)</sub>",
            "F1 Final vs Sharpness<br><sub>(idéal = coin haut gauche)</sub>"
        ],
        horizontal_spacing=0.15
    )

    opts        = list(sharpness.keys())
    vals        = [sharpness[o] for o in opts]
    labels_list = [LABELS.get(o, o) for o in opts]
    colors_list = [getattr(C, o, "gray") for o in opts]

    # Subplot 1 : Barplot
    for i, (label, val, color) in enumerate(zip(labels_list, vals, colors_list)):
        fig.add_trace(go.Bar(
            x=[label], y=[val], marker_color=color,
            marker_line_color="white", marker_line_width=2, marker_opacity=0.9,
            text=[f"{val:.5f}"], textposition="outside",
            textfont=dict(size=12, family=FONT, color=C.text),
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>Sharpness: {val:.5f}<extra></extra>",
        ), row=1, col=1)

    best_idx = int(np.argmin(vals))
    fig.add_annotation(
        x=labels_list[best_idx], y=vals[best_idx] * 0.5,
        text="← Meilleur<br>(plus plat)", showarrow=True,
        arrowhead=2, arrowcolor=C.success,
        font=dict(color=C.success, size=10, family=FONT),
        bgcolor=C.success_bg, bordercolor=C.success, borderwidth=1, borderpad=6,
        xref="x", yref="y", row=1, col=1
    )

    # Subplot 2 : Scatter
    for opt in opts:
        sub   = df[df["optimizer"] == opt]
        if sub.empty:
            continue
        sharp = sharpness.get(opt, 0)
        color = getattr(C, opt, "gray")
        label = LABELS.get(opt, opt)
        fig.add_trace(go.Scatter(
            x=[sharp] * len(sub), y=sub["final_val_f1"],
            mode="markers", name=label,
            marker=dict(color=color, size=8, opacity=0.6,
                        line=dict(color="white", width=1)),
            showlegend=True,
            hovertemplate=f"<b>{label}</b><br>Sharpness: {sharp:.5f}<br>F1: %{{y:.4f}}<extra></extra>",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[sharp], y=[sub["final_val_f1"].mean()],
            mode="markers",
            marker=dict(color=color, size=16, symbol="diamond",
                        line=dict(color="white", width=2)),
            showlegend=False,
            hovertemplate=f"<b>{label} (moy)</b><br>F1 moy: {sub['final_val_f1'].mean():.4f}<extra></extra>",
        ), row=1, col=2)

    fig.update_xaxes(title_text="", row=1, col=1, gridcolor=C.border)
    fig.update_yaxes(title_text="Sharpness", row=1, col=1, gridcolor=C.border)
    fig.update_xaxes(title_text="Sharpness", row=1, col=2, gridcolor=C.border)
    fig.update_yaxes(title_text="F1 Macro (validation)", row=1, col=2, gridcolor=C.border)

    # ✅ FIX 2 appliqué ici : on n'inclut pas 'legend' de PLOTLY_BASE
    fig.update_layout(
        **plotly_base_no_legend(),
        title=dict(
            text="<b>Figure 4 — Analyse Sharpness vs Performance</b><br>"
                 "<sub>Keskar et al. (2017) | AG News | DistilBERT | G05</sub>",
            font=dict(size=15, color=C.text)
        ),
        height=480, showlegend=True,
        legend=dict(x=1.05, y=0.5, bgcolor=C.surface,
                    bordercolor=C.border, borderwidth=1),
    )
    for ann in fig.layout.annotations:
        ann.font.update(size=12, color=C.text)
    return fig


def create_boxplot(metric="final_val_f1"):
    if not HAS_CSV:
        return empty_state()

    df = df_raw
    labels_map = {
        "final_val_f1":       "F1 Macro",
        "final_val_accuracy": "Accuracy",
        "train_time_min":     "Temps (min)",
    }
    ylabel = labels_map.get(metric, metric)
    fig = go.Figure()

    for opt in ["adamw", "sgd", "adafactor"]:
        sub = df[df["optimizer"] == opt][metric]
        if sub.empty:
            continue

        color      = getattr(C, opt)
        fill_color = getattr(C, opt + "_fill", "rgba(100,100,100,0.15)")

        # boxpoints="all" + jitter intégré Plotly
        # Évite le bug list[str] + np.array numpy
        fig.add_trace(go.Box(
            y=sub,
            name=LABELS[opt],
            marker=dict(color=color, size=6, opacity=0.55,
                        line=dict(color="white", width=0.8)),
            line_color=color,
            fillcolor=fill_color,
            boxmean="sd",
            boxpoints="all",
            jitter=0.35,
            pointpos=0,
            hovertemplate="<b>%{fullData.name}</b><br>%{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text=f"<b>Distribution {ylabel}</b><br>"
                 "<sub>Tous les trials | Boxplot + points individuels</sub>",
            font=dict(size=15, color=C.text)
        ),
        yaxis_title=ylabel,
        height=440,
        legend=LEGEND_DEFAULT,
    )
    apply_axes(fig)
    return fig


def create_correlation_matrix():
    if not HAS_CSV:
        return empty_state()
    try:
        df = df_raw
        cols_numeric = ["lr", "batch_size", "warmup_steps", "num_epochs",
                        "final_val_f1", "final_val_accuracy", "train_time_min"]
        available_cols = [c for c in cols_numeric if c in df.columns]
        if len(available_cols) < 3:
            return empty_state("Colonnes manquantes")

        df_corr = df[available_cols].copy()
        if "lr" in df_corr.columns:
            df_corr["lr"] = np.log10(df_corr["lr"])

        corr_matrix = df_corr.corr()
        labels_map = {
            "lr": "log(LR)", "batch_size": "Batch",
            "warmup_steps": "Warmup", "num_epochs": "Epochs",
            "final_val_f1": "F1 Macro", "final_val_accuracy": "Accuracy",
            "train_time_min": "Temps (min)"
        }
        labels_readable = [labels_map.get(c, c) for c in corr_matrix.columns]

        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values, x=labels_readable, y=labels_readable,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
            texttemplate="%{text}",
            textfont=dict(size=11, family=FONT),
            hovertemplate="<b>%{y} vs %{x}</b><br>Corrélation: %{z:.3f}<extra></extra>",
            colorbar=dict(tickfont=dict(size=10, color=C.muted), outlinewidth=0,
                          title=dict(text="Corrélation", font=dict(size=11))),
        ))
        fig.update_layout(
            **PLOTLY_BASE,
            title=dict(
                text="<b>Matrice de Corrélation</b><br>"
                     "<sub>Hyperparamètres et métriques</sub>",
                font=dict(size=15, color=C.text)
            ),
            height=500,
            legend=LEGEND_DEFAULT,
        )
        fig.update_xaxes(side="bottom", **{k: v for k, v in AXIS_STYLE.items() if k != 'title_font'})
        apply_axes(fig)
        return fig
    except Exception as e:
        return empty_state("Erreur", str(e)[:60])


def create_undersampling_chart():
    if not HAS_UNDER:
        return empty_state("Stats undersampling manquantes",
                           "Lance save_undersampling_stats.py")
    try:
        u = undersampling_stats
        CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Dataset Complet (AG News original)",
                "Après Undersampling (équilibré)"
            ],
            horizontal_spacing=0.15
        )
        full_train  = u["full_dataset"]["train"]["distribution"]
        under_train = u["undersampled_dataset"]["train"]["distribution"]
        colors_cls  = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]

        for col_idx, (dist, label_sfx) in enumerate(
                [(full_train, ""), (under_train, "")], start=1):
            counts = [dist.get(cn, 0) for cn in CLASS_NAMES]
            fig.add_trace(go.Bar(
                x=CLASS_NAMES, y=counts,
                marker_color=colors_cls,
                marker_line_color="white", marker_line_width=1.5,
                text=[f"{c:,}" if col_idx == 1 else str(c) for c in counts],
                textposition="outside",
                textfont=dict(size=11),
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>Exemples: %{y:,}<extra></extra>",
            ), row=1, col=col_idx)

        fig.update_yaxes(title_text="Nombre d'exemples", row=1, col=1, gridcolor=C.border)
        fig.update_yaxes(title_text="Nombre d'exemples", row=1, col=2, gridcolor=C.border)
        fig.update_layout(
            **PLOTLY_BASE,
            title=dict(
                text="<b>Distribution Classes — Avant/Après Undersampling</b><br>"
                     "<sub>Stratégie d'échantillonnage équilibré</sub>",
                font=dict(size=15, color=C.text)
            ),
            height=400,
        )
        for ann in fig.layout.annotations[:2]:
            ann.font.update(size=12, color=C.text)
        return fig
    except Exception as e:
        return empty_state("Erreur", str(e)[:60])


# ✅ FIX 1 : create_radar_chart correctement définie (n'était plus attachée à aucun def)
def create_radar_chart():
    """Radar chart comparant les optimiseurs sur plusieurs métriques normalisées."""
    if not HAS_CSV:
        return empty_state("CSV manquant", "Données requises pour le radar chart")
    if not HAS_SHARP:
        return empty_state("Sharpness manquante",
                           "Lance compute_landscape.py pour générer les données")
    try:
        df = df_raw
        fig = go.Figure()
        categories = ["F1 Max", "F1 Moyen", "Accuracy Max", "Vitesse", "Platitude"]

        for opt in ["adamw", "sgd", "adafactor"]:
            sub = df[df["optimizer"] == opt]
            if sub.empty:
                continue

            f1_max    = sub["final_val_f1"].max()
            f1_mean   = sub["final_val_f1"].mean()
            acc_max   = sub["final_val_accuracy"].max()
            time_mean = sub["train_time_min"].mean()
            sharp     = sharpness.get(opt, 0)

            all_f1_max    = df["final_val_f1"].max()
            all_f1_m_max  = df.groupby("optimizer")["final_val_f1"].mean().max()
            all_acc_max   = df["final_val_accuracy"].max()
            t_min         = df.groupby("optimizer")["train_time_min"].mean().min()
            t_max         = df.groupby("optimizer")["train_time_min"].mean().max()
            sv            = list(sharpness.values())
            s_min, s_max  = (min(sv), max(sv)) if sv else (0, 1)

            values = [
                f1_max  / all_f1_max   if all_f1_max > 0  else 0,
                f1_mean / all_f1_m_max if all_f1_m_max > 0 else 0,
                acc_max / all_acc_max  if all_acc_max > 0  else 0,
                1 - (time_mean - t_min) / (t_max - t_min + 1e-8),
                1 - (sharp - s_min)    / (s_max - s_min   + 1e-8),
            ]

            fig.add_trace(go.Scatterpolar(
                r=values, theta=categories, fill="toself",
                name=LABELS.get(opt, opt),
                line_color=getattr(C, opt),
                fillcolor=getattr(C, opt + "_fill", "rgba(100,100,100,0.15)"),
                opacity=0.85,
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor=C.border,
                                tickfont=dict(size=10, color=C.muted)),
                angularaxis=dict(gridcolor=C.border, linecolor=C.border2),
                bgcolor=C.surface,
            ),
            paper_bgcolor=C.surface,
            font=dict(family=FONT, color=C.text, size=12),
            title=dict(
                text="<b>Radar Chart Comparatif</b><br>"
                     "<sub>Normalisation 0→1 | 1 = meilleur sur chaque axe</sub>",
                font=dict(size=15, color=C.text)
            ),
            height=480, showlegend=True,
            legend=dict(bgcolor=C.surface, bordercolor=C.border, borderwidth=1),
        )
        return fig
    except Exception as e:
        print(f"[ERROR] create_radar_chart: {e}")
        return empty_state("Erreur radar", str(e)[:60])


# ══════════════════════════════════════════════════════════════════════════════
# 📊 COMPOSANTS COMPLEXES
# ══════════════════════════════════════════════════════════════════════════════

def build_kpi_dashboard():
    if not HAS_CSV:
        return html.Div(info_callout(
            "Aucune donnée disponible. Lance le notebook pour générer les résultats.", "warning"
        ))
    df   = df_raw
    best = df.loc[df["final_val_f1"].idxmax()]
    n    = len(df)
    avg_f1   = df["final_val_f1"].mean()
    avg_time = df["train_time_min"].mean()

    return html.Div([
        html.Div([
            kpi_card("Total Trials", str(n), "Random Search complet", C.accent, "🔬"),
            kpi_card(
                "Best F1 Macro", f"{best['final_val_f1']:.4f}",
                f"LR = {best['lr']:.1e} | {LABELS.get(best['optimizer'])}",
                getattr(C, best['optimizer'], C.accent), "🏆",
                f"↑ {((best['final_val_f1'] - avg_f1) / avg_f1 * 100):.1f}%"
            ),
            kpi_card("Best Accuracy", f"{best['final_val_accuracy']:.4f}",
                     f"Batch = {int(best['batch_size'])}", C.sgd, "📊"),
            kpi_card("Meilleur Optimiseur", LABELS.get(best["optimizer"], best["optimizer"]),
                     "Par F1 Macro", getattr(C, best['optimizer'], C.accent), "⚙️"),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap", "marginBottom": "24px"}),
    ])


def build_summary_table():
    if not HAS_CSV:
        return html.Div(info_callout("Données non disponibles", "warning"))

    df = df_raw
    summary = df.groupby("optimizer").agg(
        Best_F1   = ("final_val_f1",       "max"),
        Mean_F1   = ("final_val_f1",       "mean"),
        Std_F1    = ("final_val_f1",       "std"),
        Best_Acc  = ("final_val_accuracy", "max"),
        Mean_Acc  = ("final_val_accuracy", "mean"),
        Temps_moy = ("train_time_min",     "mean"),
        N_Trials  = ("run_id",             "count"),
    ).round(4).reset_index()

    if HAS_SHARP:
        summary["Sharpness"] = summary["optimizer"].map(
            lambda x: round(sharpness.get(x, float("nan")), 5)
        )
    summary["Optimiseur"] = summary["optimizer"].map(lambda x: LABELS.get(x, x))

    cols = ["Optimiseur", "Best_F1", "Mean_F1", "Std_F1",
            "Best_Acc", "Mean_Acc", "Temps_moy", "N_Trials"]
    if "Sharpness" in summary.columns:
        cols.append("Sharpness")
    summary = summary[cols]

    col_names = ["Optimiseur", "Best F1", "Mean F1", "Std F1",
                 "Best Acc", "Mean Acc", "Temps moy (min)", "N Trials"]
    if len(cols) > 8:
        col_names.append("Sharpness")
    summary.columns = col_names
    best_f1 = summary["Best F1"].max()

    return dash_table.DataTable(
        data=summary.to_dict("records"),
        columns=[{"name": c, "id": c} for c in summary.columns],
        sort_action="native",
        style_table={
            "overflowX": "auto", "borderRadius": "12px",
            "overflow": "hidden", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"
        },
        style_header={
            "backgroundColor": C.surface2, "color": C.text,
            "fontFamily": FONT, "fontWeight": "700", "fontSize": "12px",
            "border": f"1px solid {C.border}", "padding": "14px 20px",
            "textAlign": "center", "textTransform": "uppercase",
        },
        style_cell={
            "backgroundColor": C.surface, "color": C.text,
            "fontFamily": FONT, "fontSize": "13px",
            "border": f"1px solid {C.border}", "padding": "14px 20px",
            "textAlign": "center",
        },
        style_data_conditional=[
            {"if": {"filter_query": f"{{Best F1}} = {best_f1}"},
             "backgroundColor": C.accent_bg, "color": C.accent, "fontWeight": "800"},
            {"if": {"row_index": "odd"}, "backgroundColor": C.surface2},
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 🎯 APPLICATION DASH
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    title="G05 — Benchmark Optimiseurs",
    suppress_callback_exceptions=True,
    update_title=None,
)

app.index_string = app.index_string.replace(
    "</head>",
    f"""
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%234F46E5'/><text y='.9em' font-size='70' x='12'>📊</text></svg>">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; background: {C.bg}; font-family: {FONT}; }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .fade-in {{ animation: fadeIn 0.4s ease-out; }}
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: {C.surface2}; }}
        ::-webkit-scrollbar-thumb {{ background: {C.border2}; border-radius: 5px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {C.muted}; }}

        /* ── Floating Action Button ── */
        .fab-btn {{
            animation: fabPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) 0.3s both;
        }}
        @keyframes fabPop {{
            from {{ transform: scale(0) rotate(-180deg); opacity: 0; }}
            to   {{ transform: scale(1) rotate(0deg);   opacity: 1; }}
        }}
        .fab-btn:hover {{
            transform: scale(1.12) translateY(-3px) !important;
            box-shadow: 0 8px 28px rgba(79, 70, 229, 0.55), 0 4px 12px rgba(0,0,0,0.2) !important;
        }}
        .fab-btn:hover #fab-tooltip {{
            opacity: 1 !important;
        }}
        .fab-btn:active {{
            transform: scale(0.95) !important;
        }}

        /* Pulse ring autour du FAB */
        .fab-btn::before {{
            content: '';
            position: absolute;
            width: 58px;
            height: 58px;
            border-radius: 50%;
            background: rgba(79, 70, 229, 0.3);
            animation: fabPulse 2.5s ease-out infinite;
            pointer-events: none;
        }}
        @keyframes fabPulse {{
            0%   {{ transform: scale(1);   opacity: 0.7; }}
            70%  {{ transform: scale(1.7); opacity: 0;   }}
            100% {{ transform: scale(1.7); opacity: 0;   }}
        }}
    </style>
    </head>
    """
)

TAB_STYLE = {
    "fontFamily": FONT, "fontSize": "13px", "fontWeight": "600",
    "color": C.muted, "background": "transparent", "border": "none",
    "borderBottom": "3px solid transparent", "padding": "12px 24px",
    "cursor": "pointer", "transition": "all 0.3s ease",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "color": C.accent, "borderBottom": f"3px solid {C.accent}",
    "background": C.accent_bg,
}

# ══════════════════════════════════════════════════════════════════════════════
# 🏗️ LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app.layout = html.Div([

    # ─── HEADER ───
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Span("G05", style={
                        "color": C.accent, "fontFamily": FONT,
                        "fontSize": "13px", "fontWeight": "800",
                        "background": C.accent_bg, "borderRadius": "8px",
                        "padding": "4px 12px", "marginRight": "16px",
                    }),
                    html.Span("Benchmark d'Optimiseurs", style={
                        "color": C.text, "fontFamily": FONT,
                        "fontSize": "20px", "fontWeight": "800",
                    }),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                html.Div([
                    html.Span("AG News"), html.Span(" · ", style={"color": C.faint}),
                    html.Span("DistilBERT"), html.Span(" · ", style={"color": C.faint}),
                    html.Span("Random Search"), html.Span(" · ", style={"color": C.faint}),
                    html.Span("P01"),
                ], style={"color": C.muted, "fontFamily": FONT, "fontSize": "13px"}),
            ]),
            html.Div([
                status_badge("CSV",          "success" if HAS_CSV   else "warning"),
                status_badge("Historiques",  "success" if HAS_HIST  else "warning"),
                status_badge("Sharpness",    "success" if HAS_SHARP else "warning"),
                status_badge("Landscape",    "success" if HAS_LAND  else "warning"),
                status_badge("Undersampling","success" if HAS_UNDER else "info"),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
    ], style={
        "background": C.surface, "borderBottom": f"1px solid {C.border}",
        "padding": "24px 48px", "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
    }),

    # ─── BODY ───
    html.Div([
        html.Div(build_kpi_dashboard(), style={"marginBottom": "32px"}, className="fade-in"),

        html.Div([
            dcc.Tabs(
                id="main-tabs", value="tab-overview",
                style={
                    "borderBottom": f"1px solid {C.border}",
                    "background": C.surface,
                    "borderRadius": "16px 16px 0 0",
                    "padding": "0 12px",
                },
                children=[
                    dcc.Tab(label="📊 Vue d'ensemble",    value="tab-overview",      style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="📈 Convergence",        value="tab-convergence",   style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="⚙️ Hyperparamètres",    value="tab-hyperparam",    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="🔬 Analyses Avancées",  value="tab-advanced",      style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="🏔️ Loss Landscape",     value="tab-landscape",     style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="📦 Distributions",      value="tab-distributions", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="📋 Tableau Récap",      value="tab-summary",       style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                ],
            ),
            html.Div(
                id="tab-content",
                style={
                    "background": C.surface,
                    "border": f"1px solid {C.border}", "borderTop": "none",
                    "borderRadius": "0 0 16px 16px",
                    "padding": "32px 36px",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.06)",
                    "minHeight": "600px",
                },
                className="fade-in"
            ),
        ]),
    ], style={"maxWidth": "1440px", "margin": "0 auto", "padding": "36px 48px 56px"}),

    html.Div([
        html.Div([
            html.Span("© 2026 G05 Research", style={"color": C.faint, "fontSize": "12px"}),
            html.Span("  —  ", style={"color": C.faint}),
            html.Span("AYONTA", style={"color": C.muted, "fontSize": "12px", "fontWeight": "600"}),
            html.Span(" · ", style={"color": C.faint}),
            html.Span("BAMOGO", style={"color": C.muted, "fontSize": "12px", "fontWeight": "600"}),
            html.Span(" · ", style={"color": C.faint}),
            html.Span("KOULOU", style={"color": C.muted, "fontSize": "12px", "fontWeight": "600"}),
        ], style={"textAlign": "center", "padding": "24px"})
    ], style={"background": C.surface2, "borderTop": f"1px solid {C.border}", "marginTop": "48px"}),


    # ─── BOUTON FLOTTANT TÉLÉCHARGEMENT RAPPORT ───
    html.A(
        href="/assets/rapport.pdf",
        download="rapport.pdf",
        target="_blank",
        children=[
            html.Div([
                # Icône PDF
                html.Div([
                    html.Span("📄", style={"fontSize": "26px", "lineHeight": "1"}),
                ], style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                }),
                # Tooltip label
                html.Span("Télécharger le rapport", id="fab-tooltip", style={
                    "position": "absolute",
                    "right": "72px",
                    "bottom": "16px",
                    "background": C.text,
                    "color": "white",
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "fontFamily": FONT,
                    "padding": "7px 14px",
                    "borderRadius": "8px",
                    "whiteSpace": "nowrap",
                    "pointerEvents": "none",
                    "opacity": "0",
                    "transition": "opacity 0.25s ease",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
                }),
            ], style={"position": "relative", "display": "flex",
                      "alignItems": "center", "justifyContent": "center",
                      "width": "100%", "height": "100%"}),
        ],
        style={
            "position": "fixed",
            "bottom": "32px",
            "right": "32px",
            "width": "58px",
            "height": "58px",
            "borderRadius": "50%",
            "background": f"linear-gradient(135deg, {C.accent} 0%, #7C3AED 100%)",
            "boxShadow": "0 4px 20px rgba(79, 70, 229, 0.45), 0 2px 8px rgba(0,0,0,0.15)",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "zIndex": "9999",
            "textDecoration": "none",
            "transition": "transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.25s ease",
            "cursor": "pointer",
        },
        className="fab-btn",
        id="fab-download",
    ),

], style={"background": C.bg, "minHeight": "100vh"})


# ══════════════════════════════════════════════════════════════════════════════
# 🔄 CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab_content(tab):

    if tab == "tab-overview":
        return html.Div([
            section_header("Vue d'Ensemble",
                           "Distribution des données et comparaison des optimiseurs", "📊"),
            html.Div([
                dcc.Graph(id="graph-undersampling",
                          figure=create_undersampling_chart(), config=PLOTLY_CONFIG),
            ], style={"marginBottom": "32px"}),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id="graph-overview-radar",
                        figure=create_radar_chart() if HAS_SHARP else create_boxplot("final_val_f1"),
                        config=PLOTLY_CONFIG
                    ),
                ], style={"width": "48%", "display": "inline-block"}),
                html.Div([
                    dcc.Graph(id="graph-overview-correlation",
                              figure=create_correlation_matrix(), config=PLOTLY_CONFIG),
                ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%"}),
            ]),
            info_callout(
                "L'undersampling équilibré garantit que chaque classe est représentée équitablement. " +
                ("Le radar chart normalise toutes les métriques entre 0 et 1." if HAS_SHARP
                 else "Le radar chart nécessite les données de sharpness (lance compute_landscape.py)."),
                "info" if HAS_SHARP else "warning"
            ),
        ])

    elif tab == "tab-convergence":
        return html.Div([
            section_header("Courbes de Convergence",
                           "Évolution de la loss et du F1 pendant l'entraînement", "📈"),
            html.Div([
                html.Label("Options d'affichage", style={
                    "color": C.text, "fontFamily": FONT, "fontSize": "13px",
                    "fontWeight": "700", "marginBottom": "12px", "display": "block",
                }),
                dcc.Checklist(
                    id="conv-show-all",
                    options=[{"label": " Afficher tous les trials (arrière-plan)", "value": "yes"}],
                    value=[], inline=True,
                    inputStyle={"marginRight": "8px", "accentColor": C.accent},
                    labelStyle={"color": C.text_light, "fontSize": "13px", "fontFamily": FONT},
                ),
            ], style={"marginBottom": "20px", "padding": "16px 20px",
                      "background": C.surface2, "borderRadius": "12px",
                      "border": f"1px solid {C.border}"}),
            dcc.Graph(id="graph-convergence",
                      figure=create_convergence_plot(False), config=PLOTLY_CONFIG),
            info_callout(
                "Les courbes épaisses = meilleur run de chaque optimiseur. "
                "La loss est lissée avec une fenêtre mobile de 8 étapes.", "info"
            ),
        ])

    elif tab == "tab-hyperparam":
        return html.Div([
            section_header("Exploration des Hyperparamètres",
                           "Impact du learning rate et des autres hyperparamètres", "⚙️"),
            radio_group("scatter-metric",
                        [{"label": "F1 Macro", "value": "final_val_f1"},
                         {"label": "Accuracy", "value": "final_val_accuracy"}],
                        "final_val_f1", "Métrique à afficher"),
            html.Div([
                html.Label("Ligne de tendance", style={
                    "color": C.text, "fontFamily": FONT, "fontSize": "13px",
                    "fontWeight": "600", "marginRight": "12px",
                }),
                dcc.Checklist(
                    id="scatter-trend",
                    options=[{"label": " Afficher", "value": "yes"}],
                    value=[], inline=True,
                    inputStyle={"marginRight": "6px", "accentColor": C.accent},
                    labelStyle={"color": C.text_light, "fontSize": "13px", "fontFamily": FONT},
                ),
            ], style={"marginBottom": "20px", "padding": "12px 16px",
                      "background": C.surface2, "borderRadius": "10px"}),
            dcc.Graph(id="graph-scatter",
                      figure=create_scatter_plot("final_val_f1", False), config=PLOTLY_CONFIG),
            html.Hr(style={"border": "none", "borderTop": f"1px solid {C.border}", "margin": "32px 0"}),
            section_header("Heatmap Optimiseur × Plage LR",
                           "Identification des zones favorables du learning rate"),
            html.Div([
                html.Div([
                    dcc.Graph(id="graph-heatmap-f1",
                              figure=create_heatmap("final_val_f1"), config=PLOTLY_CONFIG),
                ], style={"width": "48%", "display": "inline-block"}),
                html.Div([
                    dcc.Graph(id="graph-heatmap-acc",
                              figure=create_heatmap("final_val_accuracy"), config=PLOTLY_CONFIG),
                ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%"}),
            ]),
            info_callout("La heatmap affiche la valeur maximale par zone. "
                         "Les cellules vides = aucun trial dans cette zone.", "info"),
        ])

    elif tab == "tab-advanced":
        return html.Div([
            section_header("Analyses Avancées", "Sharpness vs Performance et corrélations", "🔬"),
            html.Div([
                dcc.Graph(id="graph-sharpness-vs-perf",
                          figure=create_sharpness_vs_performance(), config=PLOTLY_CONFIG),
            ]) if HAS_SHARP else info_callout(
                "Lance 'python compute_landscape.py' pour générer les données de sharpness.", "warning"
            ),
            html.Hr(style={"border": "none", "borderTop": f"1px solid {C.border}",
                           "margin": "32px 0"}) if HAS_SHARP else None,
            section_header("Matrice de Corrélation",
                           "Relations entre hyperparamètres et métriques"),
            dcc.Graph(id="graph-correlation-advanced",
                      figure=create_correlation_matrix(), config=PLOTLY_CONFIG),
            info_callout("Corrélation positives (rouge) = variables co-évoluent. "
                         "Corrélations négatives (bleu) = relation inverse. "
                         "Rappel : corrélation ≠ causalité.", "info", "💡"),
        ])

    elif tab == "tab-landscape":
        return html.Div([
            section_header("Loss Landscape & Sharpness",
                           "Analyse de la géométrie du paysage de loss", "🏔️"),
            dcc.Graph(id="graph-landscape",
                      figure=create_landscape_plot(), config=PLOTLY_CONFIG),
            info_callout("Un minimum plus plat (faible sharpness) est corrélé à une meilleure "
                         "généralisation.", "success", "✅"),
            html.Hr(style={"border": "none", "borderTop": f"1px solid {C.border}", "margin": "32px 0"}),
            section_header("Analyse de Sharpness",
                           "Keskar et al. (2017) — Métrique de platitude du minimum"),
            dcc.Graph(id="graph-sharpness",
                      figure=create_sharpness_plot(), config=PLOTLY_CONFIG),
            info_callout("S = (1/N) Σ |L(θ + ε·di) - L(θ)|. "
                         "Plus bas = minimum plus plat = meilleure généralisation attendue.", "info"),
        ])

    elif tab == "tab-distributions":
        return html.Div([
            section_header("Distribution des Performances",
                           "Analyse statistique détaillée par optimiseur", "📦"),
            radio_group("boxplot-metric",
                        [{"label": "F1 Macro",             "value": "final_val_f1"},
                         {"label": "Accuracy",             "value": "final_val_accuracy"},
                         {"label": "Temps d'entraînement", "value": "train_time_min"}],
                        "final_val_f1", "Métrique"),
            dcc.Graph(id="graph-boxplot",
                      figure=create_boxplot("final_val_f1"), config=PLOTLY_CONFIG),
            info_callout("Boxplot = médiane + quartiles + valeurs extrêmes. "
                         "Points individuels = tous les trials (swarm plot).", "info"),
        ])

    elif tab == "tab-summary":
        return html.Div([
            section_header("Tableau Récapitulatif",
                           "Synthèse complète des résultats par optimiseur", "📋"),
            build_summary_table(),
            info_callout("Critères P01 : Best F1 (performance pic), Mean F1 (robustesse), "
                         "Std F1 (variance), Sharpness (généralisation), Temps (efficacité). "
                         "Cliquer sur les en-têtes pour trier.", "info"),
        ])

    return html.Div()


@app.callback(Output("graph-convergence", "figure"),
              Input("conv-show-all", "value"), prevent_initial_call=True)
def update_convergence(show_all):
    return create_convergence_plot("yes" in (show_all or []))


@app.callback(Output("graph-scatter", "figure"),
              [Input("scatter-metric", "value"), Input("scatter-trend", "value")],
              prevent_initial_call=True)
def update_scatter(metric, show_trend):
    return create_scatter_plot(metric, "yes" in (show_trend or []))


@app.callback(Output("graph-boxplot", "figure"),
              Input("boxplot-metric", "value"), prevent_initial_call=True)
def update_boxplot(metric):
    return create_boxplot(metric)


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("  🚀 G05 Dashboard Enhanced v2.1 — Corrigé")
    print("═" * 70)
    print(f"  📊 CSV             : {'✅ Chargé'        if HAS_CSV   else '❌ Manquant'}")
    print(f"  📈 Historiques     : {'✅ Disponibles'   if HAS_HIST  else '❌ Manquants'}")
    print(f"  📐 Sharpness       : {'✅ Calculée'      if HAS_SHARP else '❌ Non calculée'}")
    print(f"  🏔️  Landscape       : {'✅ Disponible'   if HAS_LAND  else '❌ Non disponible'}")
    print(f"  📉 Undersampling   : {'✅ Stats dispo'   if HAS_UNDER else '💡 Lance save_undersampling_stats.py'}")
    print("═" * 70)
    print("  🌐 URL : http://127.0.0.1:8050")
    print("  💡 Bugs corrigés :")
    print("     ✅ create_radar_chart définie correctement")
    print("     ✅ Conflit 'legend' dans PLOTLY_BASE résolu")
    print("     ✅ fillcolor rgba valide (plus de #hex30)")
    print("═" * 70 + "\n")

    app.run(debug=True, host="0.0.0.0", port=8050)