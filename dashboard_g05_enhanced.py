import os, json, warnings
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, State, clientside_callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from functools import lru_cache

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 🎨 THÈMES — CSS vars + Python-side pour Plotly
# ══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "light": {
        "name": "light",
        "--bg":             "#F0F2F8",
        "--surface":        "#FFFFFF",
        "--surface2":       "#F5F7FB",
        "--surface3":       "#EEF1F8",
        "--border":         "#E1E6F0",
        "--border2":        "#C8D0E2",
        "--text":           "#111827",
        "--text-light":     "#374151",
        "--muted":          "#6B7280",
        "--faint":          "#9CA3AF",
        "--adamw":          "#2563EB",
        "--sgd":            "#DC2626",
        "--adafactor":      "#16A34A",
        "--accent":         "#4F46E5",
        "--accent-bg":      "#EEF2FF",
        "--accent2":        "#7C3AED",
        "--success":        "#059669",
        "--success-bg":     "#ECFDF5",
        "--warning":        "#D97706",
        "--warning-bg":     "#FFFBEB",
        "--info":           "#0284C7",
        "--info-bg":        "#E0F2FE",
        "--grid":           "#E8ECF5",
        "--zero-line":      "#D1D9EE",
        "--shadow":         "rgba(0,0,0,0.06)",
        "--shadow-deep":    "rgba(0,0,0,0.14)",
        "--plot-bg":        "#FAFBFF",
        "--paper-bg":       "#FFFFFF",
        "--scrollbar-bg":   "#E1E6F0",
        "--scrollbar-thumb":"#C8D0E2",
        # Python-side for Plotly (cannot read CSS vars)
        "adamw":          "#2563EB",
        "sgd":            "#DC2626",
        "adafactor":      "#16A34A",
        "adamw_fill":     "rgba(37,99,235,0.12)",
        "sgd_fill":       "rgba(220,38,38,0.12)",
        "adafactor_fill": "rgba(22,163,74,0.12)",
        "accent":         "#4F46E5",
        "accent2":        "#7C3AED",
        "success":        "#059669",
        "success_bg":     "#ECFDF5",
        "warning":        "#D97706",
        "warning_bg":     "#FFFBEB",
        "info":           "#0284C7",
        "info_bg":        "#E0F2FE",
        "text":           "#111827",
        "text_light":     "#374151",
        "muted":          "#6B7280",
        "faint":          "#9CA3AF",
        "border":         "#E1E6F0",
        "border2":        "#C8D0E2",
        "surface":        "#FFFFFF",
        "surface2":       "#F5F7FB",
        "surface3":       "#EEF1F8",
        "grid":           "#E8ECF5",
        "zero_line":      "#D1D9EE",
        "plot_bg":        "#FAFBFF",
        "paper_bg":       "#FFFFFF",
        "colorscale_heat": [[0,"#EEF2FF"],[0.5,"#818CF8"],[1,"#3730A3"]],
        "corr_scale":      [[0,"#EF4444"],[0.5,"#F8FAFC"],[1,"#2563EB"]],
    },
    "dark": {
        "name": "dark",
        "--bg":             "#0A0E1A",
        "--surface":        "#111827",
        "--surface2":       "#1C2537",
        "--surface3":       "#243044",
        "--border":         "#2D3748",
        "--border2":        "#374151",
        "--text":           "#F9FAFB",
        "--text-light":     "#E5E7EB",
        "--muted":          "#9CA3AF",
        "--faint":          "#6B7280",
        "--adamw":          "#60A5FA",
        "--sgd":            "#F87171",
        "--adafactor":      "#34D399",
        "--accent":         "#818CF8",
        "--accent-bg":      "#1E1B4B",
        "--accent2":        "#A78BFA",
        "--success":        "#34D399",
        "--success-bg":     "#064E3B",
        "--warning":        "#FBBF24",
        "--warning-bg":     "#451A03",
        "--info":           "#38BDF8",
        "--info-bg":        "#0C2D48",
        "--grid":           "#1E2D45",
        "--zero-line":      "#2D3F5C",
        "--shadow":         "rgba(0,0,0,0.45)",
        "--shadow-deep":    "rgba(0,0,0,0.65)",
        "--plot-bg":        "#111827",
        "--paper-bg":       "#111827",
        "--scrollbar-bg":   "#1C2537",
        "--scrollbar-thumb":"#2D3748",
        # Python-side for Plotly
        "adamw":          "#60A5FA",
        "sgd":            "#F87171",
        "adafactor":      "#34D399",
        "adamw_fill":     "rgba(96,165,250,0.18)",
        "sgd_fill":       "rgba(248,113,113,0.18)",
        "adafactor_fill": "rgba(52,211,153,0.18)",
        "accent":         "#818CF8",
        "accent2":        "#A78BFA",
        "success":        "#34D399",
        "success_bg":     "#064E3B",
        "warning":        "#FBBF24",
        "warning_bg":     "#451A03",
        "info":           "#38BDF8",
        "info_bg":        "#0C2D48",
        "text":           "#F9FAFB",
        "text_light":     "#E5E7EB",
        "muted":          "#9CA3AF",
        "faint":          "#6B7280",
        "border":         "#2D3748",
        "border2":        "#374151",
        "surface":        "#111827",
        "surface2":       "#1C2537",
        "surface3":       "#243044",
        "grid":           "#1E2D45",
        "zero_line":      "#2D3F5C",
        "plot_bg":        "#111827",
        "paper_bg":       "#111827",
        "colorscale_heat": [[0,"#1E1B4B"],[0.5,"#4F46E5"],[1,"#A5B4FC"]],
        "corr_scale":      [[0,"#F87171"],[0.5,"#1C2537"],[1,"#60A5FA"]],
    },
}

LABELS = {"adamw": "AdamW", "sgd": "SGD + Nesterov", "adafactor": "Adafactor"}
FONT   = "'DM Sans', 'Helvetica Neue', Arial, sans-serif"

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d","select2d"],
    "displaylogo": False,
    "toImageButtonOptions": {"format":"png","filename":"g05_graph","height":1080,"width":1920,"scale":2},
}

# ══════════════════════════════════════════════════════════════════════════════
# 📊 DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_csv_cached():
    for p in ["results/logs/random_search_results_*.csv","results/logs/random_search_results.csv"]:
        files = glob.glob(p)
        if files:
            df = pd.read_csv(sorted(files)[-1])
            df["opt_label"] = df["optimizer"].map(lambda x: LABELS.get(x,x))
            df["lr_log"]    = np.log10(df["lr"].astype(float))
            df["lr_bucket"] = pd.cut(df["lr_log"], bins=5,
                labels=["1e-6→1e-5","1e-5→5e-5","5e-5→1e-4","1e-4→2e-4","2e-4→5e-4"])
            return df
    return None

@lru_cache(maxsize=4)
def load_json_cached(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

df_raw              = load_csv_cached()
histories           = load_json_cached("results/logs/training_histories.json") or []
sharpness           = load_json_cached("results/logs/sharpness_scores.json") or {}
landscape           = load_json_cached("results/logs/landscape_results.json") or {}
undersampling_stats = load_json_cached("results/logs/undersampling_stats.json") or {}

HAS_CSV   = df_raw is not None
HAS_HIST  = len(histories) > 0
HAS_SHARP = len(sharpness) > 0
HAS_LAND  = len(landscape) > 0
HAS_UNDER = len(undersampling_stats) > 0

# ══════════════════════════════════════════════════════════════════════════════
# 🧩 PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def pb(T):
    return dict(paper_bgcolor=T["paper_bg"], plot_bgcolor=T["plot_bg"],
                font=dict(family=FONT, color=T["text"], size=12),
                margin=dict(l=60,r=35,t=75,b=55), hovermode="closest")

def ax(T):
    return dict(gridcolor=T["grid"], zerolinecolor=T["zero_line"],
                linecolor=T["border2"],
                tickfont=dict(size=11,color=T["muted"]),
                title_font=dict(size=12,color=T["muted"]))

def leg_h(T):
    return dict(bgcolor=T["surface"],bordercolor=T["border"],borderwidth=1,
                font=dict(size=11,color=T["text"]),
                orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1)

def leg_s(T):
    return dict(bgcolor=T["surface"],bordercolor=T["border"],borderwidth=1,
                font=dict(size=11,color=T["text"]),x=1.05,y=0.5)

def upd_ax(fig,T):
    fig.update_xaxes(**ax(T)); fig.update_yaxes(**ax(T)); return fig

def empty_fig(T, title="Données non disponibles", msg=None, icon="📭"):
    fig=go.Figure()
    fig.add_annotation(
        text=f"<b>{icon}  {title}</b><br><span style='font-size:13px'>{msg or 'Lance le Random Search'}</span>",
        xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False,
        font=dict(size=14,color=T["muted"],family=FONT),align="center")
    fig.update_layout(**pb(T),height=360)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 📈 VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_scatter_plot(T, metric="final_val_f1", show_trend=False):
    if not HAS_CSV: return empty_fig(T,"CSV manquant")
    df=df_raw; yl="F1 Macro" if "f1" in metric else "Accuracy"
    fig=go.Figure()
    for opt in ["adamw","sgd","adafactor"]:
        sub=df[df["optimizer"]==opt]
        if sub.empty: continue
        fig.add_trace(go.Scatter(x=sub["lr"],y=sub[metric],mode="markers",name=LABELS[opt],
            marker=dict(color=T[opt],size=10,opacity=0.75,line=dict(color=T["surface"],width=2)),
            customdata=sub[["lr","batch_size","warmup_steps",metric]].values,
            hovertemplate=f"<b>{LABELS[opt]}</b><br>LR:%{{customdata[0]:.2e}}<br>Batch:%{{customdata[1]:.0f}}<br>{yl}:%{{customdata[3]:.4f}}<extra></extra>"))
        best=sub.loc[sub[metric].idxmax()]
        fig.add_trace(go.Scatter(x=[best["lr"]],y=[best[metric]],mode="markers",showlegend=False,
            marker=dict(color=T[opt],size=22,symbol="star",line=dict(color=T["surface"],width=2.5)),
            hovertemplate=f"<b>⭐ Best {LABELS[opt]}</b><br>{yl}:{best[metric]:.4f}<extra></extra>"))
        if show_trend and len(sub)>3:
            ll=np.log10(sub["lr"]); z=np.polyfit(ll,sub[metric],1); p=np.poly1d(z)
            lr_r=np.logspace(ll.min(),ll.max(),100)
            fig.add_trace(go.Scatter(x=lr_r,y=p(np.log10(lr_r)),mode="lines",showlegend=False,
                line=dict(color=T[opt],width=2,dash="dot"),hoverinfo="skip",opacity=0.45))
    fig.update_layout(**pb(T),xaxis_type="log",xaxis_title="Learning Rate (log)",yaxis_title=yl,height=480,
        title=dict(text=f"<b>{yl} vs Learning Rate</b><br><sub>⭐ = meilleur run</sub>",font=dict(size=15,color=T["text"])),
        legend=leg_h(T))
    return upd_ax(fig,T)

def create_heatmap(T, metric="final_val_f1"):
    if not HAS_CSV: return empty_fig(T)
    df=df_raw; yl="F1 Macro" if "f1" in metric else "Accuracy"
    pivot=df.pivot_table(values=metric,index="optimizer",columns="lr_bucket",aggfunc="max")
    pivot.index=[LABELS.get(i,i) for i in pivot.index]
    fig=go.Figure(go.Heatmap(z=pivot.values,x=[str(c) for c in pivot.columns],y=list(pivot.index),
        colorscale=T["colorscale_heat"],
        text=[[f"{v:.4f}" if not np.isnan(v) else "—" for v in row] for row in pivot.values],
        texttemplate="%{text}",textfont=dict(size=13,family=FONT,color=T["text"]),
        hovertemplate="<b>%{y}</b><br>LR:%{x}<br>Max:%{z:.4f}<extra></extra>",zmin=0,zmax=1,
        colorbar=dict(tickfont=dict(size=11,color=T["muted"]),outlinewidth=0,len=0.7,
            bgcolor=T["surface"],title=dict(text=yl,font=dict(size=11,color=T["muted"])))))
    fig.update_layout(**pb(T),xaxis_title="Plage LR",yaxis_title="",height=340,
        title=dict(text=f"<b>Heatmap {yl}</b><br><sub>Optimiseur × Plage LR</sub>",font=dict(size=15,color=T["text"])))
    return upd_ax(fig,T)

def create_convergence_plot(T, show_all=False):
    if not HAS_HIST: return empty_fig(T,"Historiques manquants","training_histories.json introuvable")
    fig=make_subplots(rows=1,cols=2,subplot_titles=["Train Loss (lissage)","Val F1 Macro"],horizontal_spacing=0.12)
    best_by_opt={}
    for h in histories:
        opt=h["optimizer"]; vhist=h.get("val_metrics_history",[])
        if not vhist: continue
        bf1=max(v["f1"] for v in vhist)
        if opt not in best_by_opt or bf1>best_by_opt[opt]["best_f1"]: best_by_opt[opt]={**h,"best_f1":bf1}
    if show_all:
        for h in histories:
            opt=h["optimizer"]; vhist=h.get("val_metrics_history",[])
            if not vhist: continue
            fig.add_trace(go.Scatter(x=[v["step"] for v in vhist],y=[v["f1"] for v in vhist],
                mode="lines",line=dict(color=T.get(opt,T["muted"]),width=0.8),
                opacity=0.12,showlegend=False,hoverinfo="skip"),row=1,col=2)
    for opt,data in best_by_opt.items():
        color=T.get(opt,T["muted"]); label=LABELS.get(opt,opt)
        losses=data.get("train_loss_history",[])
        if losses:
            smooth=pd.Series(losses).rolling(8,min_periods=1).mean().values
            fig.add_trace(go.Scatter(x=list(range(1,len(losses)+1)),y=smooth,mode="lines",
                name=label,line=dict(color=color,width=3),
                hovertemplate=f"{label}<br>Step:%{{x}}<br>Loss:%{{y:.4f}}<extra></extra>"),row=1,col=1)
        vhist=data.get("val_metrics_history",[])
        if vhist:
            fig.add_trace(go.Scatter(x=[v["step"] for v in vhist],y=[v["f1"] for v in vhist],
                mode="lines+markers",name=label,showlegend=False,line=dict(color=color,width=3),
                marker=dict(size=8,color=color,line=dict(color=T["surface"],width=2)),
                hovertemplate=f"{label}<br>F1:%{{y:.4f}}<extra></extra>"),row=1,col=2)
    fig.update_layout(**pb(T),height=460,legend=leg_h(T),
        title=dict(text="<b>Courbes de Convergence</b><br><sub>Meilleur run par optimiseur</sub>",font=dict(size=15,color=T["text"])))
    a=ax(T); fig.update_xaxes(**a,title_text="Steps"); fig.update_yaxes(**a)
    for ann in fig.layout.annotations: ann.font.update(size=13,color=T["text"])
    return fig

def create_landscape_plot(T):
    if not HAS_LAND: return empty_fig(T,"Landscape non calculé","Section 10 du notebook requise")
    fig=go.Figure()
    for opt,data in landscape.items():
        color=T.get(opt,T["muted"])
        fig.add_trace(go.Scatter(x=data["alphas"],y=data["losses"],mode="lines+markers",
            name=f"{LABELS.get(opt,opt)} (S={data.get('sharpness',0):.4f})",
            line=dict(color=color,width=3),marker=dict(size=7,color=color,line=dict(color=T["surface"],width=1.5)),
            hovertemplate=f"<b>{LABELS.get(opt,opt)}</b><br>α:%{{x:.3f}}<br>Loss:%{{y:.4f}}<extra></extra>"))
    fig.add_vline(x=0,line_dash="dash",line_color=T["faint"],line_width=2,
        annotation_text="θ* (α=0)",annotation_font=dict(size=12,color=T["muted"]),annotation_position="top")
    fig.update_layout(**pb(T),xaxis_title="Direction (α)",yaxis_title="Loss",height=460,legend=leg_h(T),
        title=dict(text="<b>Loss Landscape 1D</b><br><sub>Li et al. (2018) Filter Normalization</sub>",font=dict(size=15,color=T["text"])))
    return upd_ax(fig,T)

def create_sharpness_plot(T):
    if not HAS_SHARP: return empty_fig(T,"Sharpness non calculée")
    opts=list(sharpness.keys()); vals=[sharpness[o] for o in opts]
    ll=[LABELS.get(o,o) for o in opts]; cc=[T.get(o,T["muted"]) for o in opts]
    fig=go.Figure(go.Bar(x=ll,y=vals,marker_color=cc,marker_line_color=T["surface"],
        marker_line_width=2,marker_opacity=0.9,text=[f"{v:.5f}" for v in vals],
        textposition="outside",textfont=dict(family=FONT,size=13,color=T["text"]),
        hovertemplate="<b>%{x}</b><br>Sharpness:%{y:.5f}<extra></extra>"))
    bi=int(np.argmin(vals))
    fig.add_annotation(x=ll[bi],y=vals[bi]*0.5,text="<b>Minimum<br>le plus plat</b>",showarrow=False,
        font=dict(color=T["success"],size=11,family=FONT),
        bgcolor=T["success_bg"],bordercolor=T["success"],borderwidth=1.5,borderpad=8)
    fig.update_layout(**pb(T),yaxis_title="Sharpness",height=380,
        title=dict(text="<b>Sharpness par Optimiseur</b><br><sub>Keskar et al. (2017)</sub>",font=dict(size=15,color=T["text"])))
    return upd_ax(fig,T)

def create_sharpness_vs_performance(T):
    if not HAS_SHARP or not HAS_CSV: return empty_fig(T,"Données incomplètes")
    df=df_raw
    fig=make_subplots(rows=1,cols=2,
        subplot_titles=["Sharpness par Optimiseur","F1 vs Sharpness<br><sub>(idéal=haut-gauche)</sub>"],
        horizontal_spacing=0.15)
    opts=list(sharpness.keys()); vals=[sharpness[o] for o in opts]
    ll=[LABELS.get(o,o) for o in opts]; cc=[T.get(o,T["muted"]) for o in opts]
    for label,val,color in zip(ll,vals,cc):
        fig.add_trace(go.Bar(x=[label],y=[val],marker_color=color,marker_line_color=T["surface"],
            marker_line_width=2,marker_opacity=0.9,text=[f"{val:.5f}"],textposition="outside",
            textfont=dict(size=12,family=FONT,color=T["text"]),showlegend=False,
            hovertemplate=f"<b>{label}</b><br>Sharpness:{val:.5f}<extra></extra>"),row=1,col=1)
    bi=int(np.argmin(vals))
    fig.add_annotation(x=ll[bi],y=vals[bi]*0.5,text="← Meilleur",showarrow=True,
        arrowhead=2,arrowcolor=T["success"],font=dict(color=T["success"],size=10,family=FONT),
        bgcolor=T["success_bg"],bordercolor=T["success"],borderwidth=1,borderpad=6,xref="x",yref="y",row=1,col=1)
    for opt in opts:
        sub=df[df["optimizer"]==opt]; sharp=sharpness.get(opt,0)
        if sub.empty: continue
        color=T.get(opt,T["muted"]); label=LABELS.get(opt,opt)
        fig.add_trace(go.Scatter(x=[sharp]*len(sub),y=sub["final_val_f1"],mode="markers",name=label,
            marker=dict(color=color,size=8,opacity=0.6,line=dict(color=T["surface"],width=1)),
            hovertemplate=f"<b>{label}</b><br>Sharpness:{sharp:.5f}<br>F1:%{{y:.4f}}<extra></extra>"),row=1,col=2)
        fig.add_trace(go.Scatter(x=[sharp],y=[sub["final_val_f1"].mean()],mode="markers",showlegend=False,
            marker=dict(color=color,size=16,symbol="diamond",line=dict(color=T["surface"],width=2))),row=1,col=2)
    a=ax(T); fig.update_xaxes(**a); fig.update_yaxes(**a)
    fig.update_layout(**pb(T),height=480,showlegend=True,legend=leg_s(T),
        title=dict(text="<b>Sharpness vs Performance</b><br><sub>Keskar et al. (2017) | G05</sub>",font=dict(size=15,color=T["text"])))
    for ann in fig.layout.annotations: ann.font.update(size=12,color=T["text"])
    return fig

def create_boxplot(T, metric="final_val_f1"):
    if not HAS_CSV: return empty_fig(T)
    df=df_raw; yl={"final_val_f1":"F1 Macro","final_val_accuracy":"Accuracy","train_time_min":"Temps (min)"}.get(metric,metric)
    fig=go.Figure()
    for opt in ["adamw","sgd","adafactor"]:
        sub=df[df["optimizer"]==opt][metric]
        if sub.empty: continue
        fig.add_trace(go.Box(y=sub,name=LABELS[opt],
            marker=dict(color=T[opt],size=6,opacity=0.6,line=dict(color=T["surface"],width=0.8)),
            line_color=T[opt],fillcolor=T[opt+"_fill"],boxmean="sd",boxpoints="all",jitter=0.35,pointpos=0,
            hovertemplate="<b>%{fullData.name}</b><br>%{y:.4f}<extra></extra>"))
    fig.update_layout(**pb(T),yaxis_title=yl,height=440,legend=leg_h(T),
        title=dict(text=f"<b>Distribution {yl}</b><br><sub>Tous les trials</sub>",font=dict(size=15,color=T["text"])))
    return upd_ax(fig,T)

def create_correlation_matrix(T):
    if not HAS_CSV: return empty_fig(T)
    try:
        df=df_raw; cols=["lr","batch_size","warmup_steps","num_epochs","final_val_f1","final_val_accuracy","train_time_min"]
        avail=[c for c in cols if c in df.columns]
        if len(avail)<3: return empty_fig(T,"Colonnes manquantes")
        dc=df[avail].copy()
        if "lr" in dc.columns: dc["lr"]=np.log10(dc["lr"])
        corr=dc.corr()
        lm={"lr":"log(LR)","batch_size":"Batch","warmup_steps":"Warmup","num_epochs":"Epochs",
            "final_val_f1":"F1 Macro","final_val_accuracy":"Accuracy","train_time_min":"Temps (min)"}
        lr=[lm.get(c,c) for c in corr.columns]
        fig=go.Figure(go.Heatmap(z=corr.values,x=lr,y=lr,colorscale=T["corr_scale"],zmid=0,zmin=-1,zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],texttemplate="%{text}",
            textfont=dict(size=11,family=FONT,color=T["text"]),
            hovertemplate="<b>%{y} vs %{x}</b><br>Corrélation:%{z:.3f}<extra></extra>",
            colorbar=dict(tickfont=dict(size=10,color=T["muted"]),outlinewidth=0,bgcolor=T["surface"],
                title=dict(text="Corrélation",font=dict(size=11,color=T["muted"])))))
        fig.update_layout(**pb(T),height=500,
            title=dict(text="<b>Matrice de Corrélation</b><br><sub>Hyperparamètres et métriques</sub>",font=dict(size=15,color=T["text"])))
        fig.update_xaxes(side="bottom"); return upd_ax(fig,T)
    except Exception as e: return empty_fig(T,"Erreur",str(e)[:60])

def create_undersampling_chart(T):
    if not HAS_UNDER: return empty_fig(T,"Stats undersampling manquantes")
    try:
        u=undersampling_stats; CN=["World","Sports","Business","Sci/Tech"]
        fig=make_subplots(rows=1,cols=2,
            subplot_titles=["Dataset Complet (AG News)","Après Undersampling (équilibré)"],
            horizontal_spacing=0.15)
        ft=u["full_dataset"]["train"]["distribution"]; ut=u["undersampled_dataset"]["train"]["distribution"]
        cc=[T["adamw"],T["sgd"],T["adafactor"],T["accent"]]
        for ci,dist in enumerate([ft,ut],start=1):
            cts=[dist.get(c,0) for c in CN]
            fig.add_trace(go.Bar(x=CN,y=cts,marker_color=cc,marker_line_color=T["surface"],marker_line_width=1.5,
                text=[f"{c:,}" if ci==1 else str(c) for c in cts],textposition="outside",
                textfont=dict(size=11,color=T["text"]),showlegend=False,
                hovertemplate="<b>%{x}</b><br>Exemples:%{y:,}<extra></extra>"),row=1,col=ci)
        a=ax(T); fig.update_xaxes(**a); fig.update_yaxes(**a,title_text="Nombre d'exemples")
        fig.update_layout(**pb(T),height=400,
            title=dict(text="<b>Distribution Classes — Avant/Après Undersampling</b>",font=dict(size=15,color=T["text"])))
        for ann in fig.layout.annotations[:2]: ann.font.update(size=12,color=T["text"])
        return fig
    except Exception as e: return empty_fig(T,"Erreur",str(e)[:60])

def create_radar_chart(T):
    if not HAS_CSV: return empty_fig(T,"CSV manquant")
    if not HAS_SHARP: return empty_fig(T,"Sharpness manquante")
    try:
        df=df_raw; cats=["F1 Max","F1 Moyen","Accuracy Max","Vitesse","Platitude"]; fig=go.Figure()
        for opt in ["adamw","sgd","adafactor"]:
            sub=df[df["optimizer"]==opt]
            if sub.empty: continue
            aF1=df["final_val_f1"].max(); aMF=df.groupby("optimizer")["final_val_f1"].mean().max()
            aA=df["final_val_accuracy"].max(); tmin=df.groupby("optimizer")["train_time_min"].mean().min()
            tmax=df.groupby("optimizer")["train_time_min"].mean().max()
            sv=list(sharpness.values()); sm,sx=(min(sv),max(sv)) if sv else (0,1)
            vals=[
                sub["final_val_f1"].max()/aF1 if aF1>0 else 0,
                sub["final_val_f1"].mean()/aMF if aMF>0 else 0,
                sub["final_val_accuracy"].max()/aA if aA>0 else 0,
                1-(sub["train_time_min"].mean()-tmin)/(tmax-tmin+1e-8),
                1-(sharpness.get(opt,0)-sm)/(sx-sm+1e-8),
            ]
            fig.add_trace(go.Scatterpolar(r=vals,theta=cats,fill="toself",
                name=LABELS.get(opt,opt),line_color=T[opt],fillcolor=T[opt+"_fill"],opacity=0.85))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True,range=[0,1],gridcolor=T["border"],
                    tickfont=dict(size=10,color=T["muted"]),linecolor=T["border2"]),
                angularaxis=dict(gridcolor=T["border"],linecolor=T["border2"],tickfont=dict(color=T["text_light"])),
                bgcolor=T["plot_bg"]),
            paper_bgcolor=T["paper_bg"],font=dict(family=FONT,color=T["text"],size=12),height=480,showlegend=True,
            legend=dict(bgcolor=T["surface"],bordercolor=T["border"],borderwidth=1,font=dict(color=T["text"])),
            title=dict(text="<b>Radar Chart Comparatif</b><br><sub>Normalisation 0→1</sub>",font=dict(size=15,color=T["text"])))
        return fig
    except Exception as e: return empty_fig(T,"Erreur radar",str(e)[:60])

# ══════════════════════════════════════════════════════════════════════════════
# 🧩 COMPOSANTS UI — CSS-var based
# ══════════════════════════════════════════════════════════════════════════════

def sh(title, subtitle=None, icon=None):
    return html.Div([
        html.Div([html.Span(icon or "📊",style={"fontSize":"20px","marginRight":"12px"}),
                  html.H3(title,className="section-title")],
                 style={"display":"flex","alignItems":"center","marginBottom":"6px"}),
        html.P(subtitle,className="section-subtitle") if subtitle else html.Div(style={"marginBottom":"24px"}),
    ])

def callout(text, type_="info", icon=None):
    em=(icon or {"info":"ℹ️","success":"✅","warning":"⚠️"}.get(type_,"ℹ️"))
    return html.Div([
        html.Span(em+"  ",style={"fontSize":"16px"}),
        html.Span(text,className="callout-text"),
    ],className=f"callout callout-{type_}")

def radio_grp(id_, options, value, label=None):
    return html.Div([
        html.Label(label or "Options",className="radio-label"),
        dcc.RadioItems(id=id_,options=options,value=value,inline=True,
            inputStyle={"marginRight":"6px","marginLeft":"14px","accentColor":"var(--accent)"},
            labelStyle={"color":"var(--text)","cursor":"pointer","fontWeight":"500","fontSize":"13px","fontFamily":FONT}),
    ],className="radio-group")

def build_kpis():
    if not HAS_CSV: return callout("Aucune donnée. Lance le notebook.","warning")
    df=df_raw; best=df.loc[df["final_val_f1"].idxmax()]; n=len(df); avg=df["final_val_f1"].mean()
    opt=best["optimizer"]
    av={"adamw":"--adamw","sgd":"--sgd","adafactor":"--adafactor"}.get(opt,"--accent")
    def card(label,value,sub,av,icon,trend=None):
        return html.Div([
            html.Div([html.Span(icon+"  ",style={"fontSize":"18px","opacity":"0.8"}),
                      html.Span(label,className="kpi-label")],
                     style={"display":"flex","alignItems":"center","marginBottom":"12px"}),
            html.Div([html.Div(value,className="kpi-value",style={"color":f"var({av})"}),
                      html.Span(trend,className="kpi-trend") if trend else None],
                     style={"display":"flex","alignItems":"baseline","marginBottom":"6px"}),
            html.Div(sub,className="kpi-sub"),
        ],className="kpi-card",style={"borderTopColor":f"var({av})"})
    return html.Div([
        card("Total Trials",str(n),"Random Search","--accent","🔬"),
        card("Best F1 Macro",f"{best['final_val_f1']:.4f}",f"LR={best['lr']:.1e} · {LABELS.get(opt)}",av,"🏆",
             f"↑ {((best['final_val_f1']-avg)/avg*100):.1f}%"),
        card("Best Accuracy",f"{best['final_val_accuracy']:.4f}",f"Batch={int(best['batch_size'])}","--sgd","📊"),
        card("Meilleur Optimiseur",LABELS.get(opt,opt),"Par F1 Macro",av,"⚙️"),
    ],style={"display":"flex","gap":"18px","flexWrap":"nowrap","width":"100%"})

def badge(text, status="success"):
    return html.Span(text,className=f"badge badge-{status}")

def build_summary_table():
    if not HAS_CSV: return callout("Données non disponibles","warning")
    df=df_raw
    summary=df.groupby("optimizer").agg(
        Best_F1=("final_val_f1","max"),Mean_F1=("final_val_f1","mean"),
        Std_F1=("final_val_f1","std"),Best_Acc=("final_val_accuracy","max"),
        Mean_Acc=("final_val_accuracy","mean"),Temps_moy=("train_time_min","mean"),
        N_Trials=("run_id","count")).round(4).reset_index()
    if HAS_SHARP:
        summary["Sharpness"]=summary["optimizer"].map(lambda x: round(sharpness.get(x,float("nan")),5))
    summary["Optimiseur"]=summary["optimizer"].map(lambda x: LABELS.get(x,x))
    cols=["Optimiseur","Best_F1","Mean_F1","Std_F1","Best_Acc","Mean_Acc","Temps_moy","N_Trials"]
    if "Sharpness" in summary.columns: cols.append("Sharpness")
    summary=summary[cols]
    names=["Optimiseur","Best F1","Mean F1","Std F1","Best Acc","Mean Acc","Temps (min)","N Trials"]
    if len(cols)>8: names.append("Sharpness")
    summary.columns=names; bf=summary["Best F1"].max()
    return dash_table.DataTable(
        data=summary.to_dict("records"),
        columns=[{"name":c,"id":c} for c in summary.columns],
        sort_action="native",
        style_table={"overflowX":"auto","borderRadius":"12px","overflow":"hidden"},
        style_header={"backgroundColor":"var(--surface3)","color":"var(--text)","fontFamily":FONT,
            "fontWeight":"700","fontSize":"11px","border":"1px solid var(--border)",
            "padding":"14px 18px","textAlign":"center","textTransform":"uppercase","letterSpacing":"0.06em"},
        style_cell={"backgroundColor":"var(--surface)","color":"var(--text)","fontFamily":FONT,
            "fontSize":"13px","border":"1px solid var(--border)","padding":"13px 18px","textAlign":"center"},
        style_data_conditional=[
            {"if":{"filter_query":f"{{Best F1}} = {bf}"},"backgroundColor":"var(--accent-bg)","color":"var(--accent)","fontWeight":"800"},
            {"if":{"row_index":"odd"},"backgroundColor":"var(--surface2)"},
        ],
    )

# ══════════════════════════════════════════════════════════════════════════════
# 🎯 APP
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, title="G05 — Benchmark Optimiseurs", suppress_callback_exceptions=True)

CSS_LIGHT = "\n".join(f"  {k}: {v};" for k,v in THEMES["light"].items() if k.startswith("--"))
CSS_DARK  = "\n".join(f"  {k}: {v};" for k,v in THEMES["dark"].items()  if k.startswith("--"))

app.index_string = app.index_string.replace("</head>", f"""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&display=swap" rel="stylesheet">
<style>
/* ── CSS VARIABLES ────────────────────────────────────────────────────────── */
:root {{
{CSS_LIGHT}
}}
[data-theme="dark"] {{
{CSS_DARK}
}}

/* ── BASE ─────────────────────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  font-family: {FONT};
  background: var(--bg);
  color: var(--text);
  transition: background 0.3s ease, color 0.3s ease;
}}

/* ── SCROLLBAR ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: var(--scrollbar-bg); }}
::-webkit-scrollbar-thumb {{ background: var(--scrollbar-thumb); border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--border2); }}

/* ── HEADER ───────────────────────────────────────────────────────────────── */
.g05-header {{
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 18px 48px;
  position: sticky; top: 0; z-index: 200;
  box-shadow: 0 2px 16px var(--shadow);
  transition: background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}}
.g05-header-inner {{
  display: flex; justify-content: space-between; align-items: center;
  max-width: 1440px; margin: 0 auto;
}}
.g05-logo {{
  display: inline-flex;
  background: var(--accent-bg);
  border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
  border-radius: 10px; padding: 6px 14px; margin-right: 18px;
  font-size: 15px; letter-spacing: 0.05em;
  transition: background 0.3s ease, border-color 0.3s ease;
}}
.g05-logo .lg {{ color: var(--accent); font-weight: 900; }}
.g05-logo .ln {{ color: var(--accent2); font-weight: 900; }}
.g05-title {{
  color: var(--text); font-size: 19px; font-weight: 800;
  transition: color 0.3s ease;
}}
.g05-subtitle {{ font-size: 12px; margin-top: 2px; color: var(--muted); }}
.g05-subtitle .dot {{ color: var(--faint); }}

/* ── TOGGLE ───────────────────────────────────────────────────────────────── */
.toggle-wrap {{
  display: flex; align-items: center; gap: 8px;
  padding: 6px 14px; background: var(--surface2); border-radius: 24px;
  border: 1px solid var(--border); cursor: pointer;
  transition: background 0.3s ease, border-color 0.3s ease;
  user-select: none;
}}
.toggle-track {{
  position: relative; width: 46px; height: 24px; border-radius: 12px;
  background: var(--border2); cursor: pointer;
  transition: background 0.35s ease;
}}
[data-theme="dark"] .toggle-track {{ background: var(--accent); }}
.toggle-thumb {{
  position: absolute; top: 3px; left: 3px;
  width: 18px; height: 18px; border-radius: 50%;
  background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.25);
  transition: transform 0.35s cubic-bezier(0.34,1.56,0.64,1);
}}
[data-theme="dark"] .toggle-thumb {{ transform: translateX(22px); }}
.toggle-icon {{ font-size: 15px; line-height: 1; }}
.toggle-icon.sun {{ opacity: 1; transition: opacity 0.3s ease; }}
.toggle-icon.moon {{ opacity: 0.35; transition: opacity 0.3s ease; }}
[data-theme="dark"] .toggle-icon.sun {{ opacity: 0.35; }}
[data-theme="dark"] .toggle-icon.moon {{ opacity: 1; }}

/* ── BADGES ───────────────────────────────────────────────────────────────── */
.badge {{
  border-radius: 20px; padding: 4px 11px; font-size: 11px;
  font-weight: 600; display: inline-block; margin-right: 4px;
  transition: background 0.3s ease, color 0.3s ease;
}}
.badge-success {{ background: var(--success-bg); color: var(--success); }}
.badge-warning  {{ background: var(--warning-bg); color: var(--warning); }}
.badge-info     {{ background: var(--info-bg);    color: var(--info); }}

/* ── MAIN LAYOUT ──────────────────────────────────────────────────────────── */
.g05-main {{ max-width: 1440px; margin: 0 auto; padding: 36px 48px 56px; }}

/* ── KPI CARDS ────────────────────────────────────────────────────────────── */
.kpi-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px; padding: 22px 26px;
  flex: 1 1 0; min-width: 0;
  border-top: 4px solid var(--accent);
  box-shadow: 0 2px 12px var(--shadow);
  transition: transform 0.2s ease, box-shadow 0.2s ease,
              background 0.3s ease, border-color 0.3s ease;
  cursor: default;
}}
.kpi-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px var(--shadow-deep); }}
.kpi-label {{
  color: var(--muted); font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em;
  transition: color 0.3s ease;
}}
.kpi-value {{ font-size: 26px; font-weight: 800; line-height: 1; transition: color 0.3s ease; }}
.kpi-trend {{ font-size: 13px; margin-left: 8px; font-weight: 600; color: var(--success); }}
.kpi-sub {{ color: var(--faint); font-size: 11px; font-weight: 500; transition: color 0.3s ease; }}

/* ── TAB CONTAINER ────────────────────────────────────────────────────────── */
.tab-container {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: 0 4px 16px var(--shadow);
  overflow: hidden;
  transition: background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}}
.tab-content-area {{
  padding: 32px 36px; min-height: 600px;
  background: var(--surface);
  color: var(--text);
  transition: background 0.3s ease, color 0.3s ease;
}}

/* ── SECTION TITLES ───────────────────────────────────────────────────────── */
.section-title {{
  color: var(--text); font-size: 16px; font-weight: 700; margin: 0;
  transition: color 0.3s ease;
}}
.section-subtitle {{
  color: var(--muted); font-size: 13px; margin: 0 0 24px 32px;
  transition: color 0.3s ease;
}}

/* ── CALLOUTS ─────────────────────────────────────────────────────────────── */
.callout {{
  display: flex; align-items: flex-start; border-radius: 12px;
  padding: 14px 18px; margin-top: 18px; font-size: 13px;
  transition: background 0.3s ease, border-color 0.3s ease;
}}
.callout-text {{ color: var(--text-light); line-height: 1.6; font-size: 13px; transition: color 0.3s ease; }}
.callout-info    {{ background: var(--info-bg);    border: 1px solid color-mix(in srgb, var(--info)    30%, transparent); }}
.callout-success {{ background: var(--success-bg); border: 1px solid color-mix(in srgb, var(--success) 30%, transparent); }}
.callout-warning {{ background: var(--warning-bg); border: 1px solid color-mix(in srgb, var(--warning) 30%, transparent); }}

/* ── RADIO / OPTIONS ──────────────────────────────────────────────────────── */
.radio-group {{
  display: flex; align-items: center; margin-bottom: 20px;
  padding: 12px 16px; background: var(--surface2); border-radius: 10px;
  border: 1px solid var(--border);
  transition: background 0.3s ease, border-color 0.3s ease;
}}
.radio-label {{
  color: var(--muted); font-size: 11px; font-weight: 600; margin-right: 12px;
  text-transform: uppercase; letter-spacing: 0.07em; transition: color 0.3s ease;
}}
.option-panel {{
  margin-bottom: 20px; padding: 14px 18px;
  background: var(--surface2); border-radius: 12px; border: 1px solid var(--border);
  transition: background 0.3s ease, border-color 0.3s ease;
}}
.option-label {{
  color: var(--text); font-size: 13px; font-weight: 700;
  margin-bottom: 10px; display: block; transition: color 0.3s ease;
}}

/* ── HR ───────────────────────────────────────────────────────────────────── */
.g05-hr {{ border: none; border-top: 1px solid var(--border); margin: 32px 0; transition: border-color 0.3s ease; }}

/* ── FOOTER ───────────────────────────────────────────────────────────────── */
.g05-footer {{
  background: var(--surface2); border-top: 1px solid var(--border);
  margin-top: 48px; padding: 24px; text-align: center;
  transition: background 0.3s ease, border-color 0.3s ease;
}}
.footer-copy {{ color: var(--faint); font-size: 12px; transition: color 0.3s ease; }}
.footer-name {{ color: var(--muted); font-size: 12px; font-weight: 600; transition: color 0.3s ease; }}

/* ── FAB ──────────────────────────────────────────────────────────────────── */
.fab-btn {{
  position: fixed; top: 75vh; right: 32px;
  width: 60px; height: 60px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 26px; text-decoration: none; z-index: 9999; color: white;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  box-shadow: 0 4px 20px color-mix(in srgb, var(--accent) 55%, transparent);
  transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.3s ease,
              background 0.3s ease;
}}
.fab-btn:hover {{ transform: translateY(-4px) scale(1.08); }}

/* ── FAB ABOUT ────────────────────────────────────────────────────────────── */
.fab-about {{
  position: fixed; top: calc(75vh - 76px); right: 32px;
  width: 60px; height: 60px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 26px; z-index: 9999; color: white; cursor: pointer;
  border: none; outline: none;
  background: linear-gradient(135deg, var(--info), var(--accent));
  box-shadow: 0 4px 20px color-mix(in srgb, var(--info) 55%, transparent);
  transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.3s ease,
              background 0.3s ease;
}}
.fab-about:hover {{ transform: translateY(-4px) scale(1.08); }}

/* ── MODAL OVERLAY ────────────────────────────────────────────────────────── */
.modal-overlay {{
  position: fixed; inset: 0; z-index: 99999;
  background: rgba(0,0,0,0.65);
  backdrop-filter: blur(6px);
  display: flex; align-items: center; justify-content: center;
  padding: 24px;
  animation: fadeIn 0.25s ease-out;
}}
.modal-overlay.hidden {{ display: none; }}

.modal-box {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px;
  width: 100%; max-width: 780px;
  max-height: 88vh;
  overflow-y: auto;
  box-shadow: 0 24px 80px rgba(0,0,0,0.5);
  animation: modalPop 0.3s cubic-bezier(0.34,1.56,0.64,1);
}}
@keyframes modalPop {{
  from {{ opacity: 0; transform: scale(0.93) translateY(20px); }}
  to   {{ opacity: 1; transform: scale(1) translateY(0); }}
}}

.modal-header {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 28px 32px 0;
  border-bottom: 1px solid var(--border);
  padding-bottom: 20px;
}}
.modal-title {{
  color: var(--text); font-size: 20px; font-weight: 800;
  font-family: {FONT}; margin: 0;
}}
.modal-close {{
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 50%; width: 36px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; font-size: 16px; color: var(--muted);
  transition: background 0.2s, color 0.2s, transform 0.2s;
  flex-shrink: 0;
}}
.modal-close:hover {{ background: var(--surface3); color: var(--text); transform: scale(1.1); }}

.modal-body {{ padding: 28px 32px 32px; }}

.modal-section {{
  margin-bottom: 28px;
}}
.modal-section-title {{
  color: var(--accent); font-size: 13px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.1em;
  margin: 0 0 12px; display: flex; align-items: center; gap: 8px;
}}
.modal-section-title::after {{
  content: ''; flex: 1; height: 1px; background: var(--border);
}}
.modal-text {{
  color: var(--text-light); font-size: 14px; line-height: 1.75;
  font-family: {FONT}; margin: 0 0 10px;
}}
.modal-tag {{
  display: inline-block; background: var(--accent-bg); color: var(--accent);
  border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
  border-radius: 8px; padding: 4px 12px; font-size: 12px; font-weight: 600;
  margin: 3px 4px 3px 0; font-family: {FONT};
}}
.modal-tag-green {{
  background: var(--success-bg); color: var(--success);
  border-color: color-mix(in srgb, var(--success) 30%, transparent);
}}
.modal-tag-orange {{
  background: var(--warning-bg); color: var(--warning);
  border-color: color-mix(in srgb, var(--warning) 30%, transparent);
}}
.modal-step {{
  display: flex; gap: 14px; margin-bottom: 14px; align-items: flex-start;
}}
.modal-step-num {{
  background: var(--accent); color: white; border-radius: 50%;
  width: 26px; height: 26px; display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 800; flex-shrink: 0; margin-top: 2px;
}}
.modal-step-text {{ color: var(--text-light); font-size: 14px; line-height: 1.65; }}
.modal-step-text strong {{ color: var(--text); font-weight: 700; }}

.modal-back-btn {{
  display: inline-flex; align-items: center; gap: 8px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: white; border: none; border-radius: 12px;
  padding: 12px 24px; font-size: 14px; font-weight: 700;
  font-family: {FONT}; cursor: pointer; margin-top: 8px;
  box-shadow: 0 4px 16px color-mix(in srgb, var(--accent) 40%, transparent);
  transition: transform 0.2s, box-shadow 0.2s;
}}
.modal-back-btn:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px color-mix(in srgb, var(--accent) 50%, transparent); }}


/* ── ANIMATIONS ───────────────────────────────────────────────────────────── */
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.fade-in {{ animation: fadeIn 0.4s ease-out; }}

/* ── DASH TABLE — dark fix ────────────────────────────────────────────────── */
.dash-spreadsheet-container .dash-spreadsheet-inner td {{ transition: background 0.3s ease, color 0.3s ease; }}
.dash-spreadsheet-container .dash-spreadsheet-inner th {{ transition: background 0.3s ease, color 0.3s ease; }}
.Select-control, .Select-menu-outer {{ background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important; }}
</style>
</head>""")

TAB_STYLE = {
    "fontFamily": FONT, "fontSize": "13px", "fontWeight": "600",
    "color": "var(--muted)", "background": "transparent", "border": "none",
    "borderBottom": "3px solid transparent", "padding": "12px 22px",
    "cursor": "pointer", "transition": "color 0.25s, border-color 0.25s, background 0.25s",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "color": "var(--accent)", "borderBottom": "3px solid var(--accent)",
    "background": "var(--accent-bg)",
}

# ══════════════════════════════════════════════════════════════════════════════
# 🏗️ LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    dcc.Store(id="theme-store", data="dark"),

    # Header
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Span("G", className="lg"), html.Span("05", className="ln"),
                ], className="g05-logo"),
                html.Div([
                    html.Div("Benchmark d'Optimiseurs", className="g05-title"),
                    html.Div([
                        html.Span("AG News",       style={"color":"var(--adamw)"}),
                        html.Span("  ·  ",         className="dot"),
                        html.Span("DistilBERT",    style={"color":"var(--adafactor)"}),
                        html.Span("  ·  ",         className="dot"),
                        html.Span("Random Search", style={"color":"var(--muted)"}),
                        html.Span("  ·  ",         className="dot"),
                        html.Span("P01",           style={"color":"var(--muted)"}),
                    ], className="g05-subtitle"),
                ]),
            ], style={"display":"flex","alignItems":"center"}),

            html.Div([
                html.Div([
                    badge("CSV",           "success" if HAS_CSV   else "warning"),
                    badge("Historiques",   "success" if HAS_HIST  else "warning"),
                    badge("Sharpness",     "success" if HAS_SHARP else "warning"),
                    badge("Landscape",     "success" if HAS_LAND  else "warning"),
                    badge("Undersampling", "success" if HAS_UNDER else "info"),
                ], style={"display":"flex","flexWrap":"wrap","gap":"4px","marginRight":"18px"}),

                html.Div([
                    html.Span("☀️", className="toggle-icon sun"),
                    html.Div([html.Div(className="toggle-thumb")], className="toggle-track"),
                    html.Span("🌙", className="toggle-icon moon"),
                ], className="toggle-wrap", id="theme-toggle-btn", n_clicks=0),
            ], style={"display":"flex","alignItems":"center"}),
        ], className="g05-header-inner"),
    ], className="g05-header"),

    # Main
    html.Div([
        html.Div(build_kpis(), style={"marginBottom":"32px"}, className="fade-in"),
        html.Div([
            dcc.Tabs(id="main-tabs", value="tab-overview",
                style={"borderBottom":"1px solid var(--border)","background":"var(--surface)",
                       "borderRadius":"16px 16px 0 0","padding":"0 12px"},
                children=[
                    dcc.Tab(label="📊 Vue d'ensemble",   value="tab-overview",      style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="📈 Convergence",       value="tab-convergence",   style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="⚙️ Hyperparamètres",   value="tab-hyperparam",    style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="🔬 Analyses Avancées", value="tab-advanced",      style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="🏔️ Loss Landscape",    value="tab-landscape",     style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="📦 Distributions",     value="tab-distributions", style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="📋 Tableau Récap",     value="tab-summary",       style=TAB_STYLE, selected_style=TAB_SELECTED),
                ],
            ),
            html.Div(id="tab-content", className="tab-content-area"),
        ], className="tab-container fade-in"),
    ], className="g05-main"),

    # Footer
    html.Div([
        html.Span("© 2026 G05 Research  —  ", className="footer-copy"),
        html.Span("AYONTA", className="footer-name"),
        html.Span("  ·  ", className="footer-copy"),
        html.Span("BAMOGO", className="footer-name"),
        html.Span("  ·  ", className="footer-copy"),
        html.Span("KOULOU", className="footer-name"),
    ], className="g05-footer"),

    html.A("📄", href="assets/rapport.pdf", download="G05_Rapport.pdf",
           target="_blank", className="fab-btn", title="Télécharger le rapport"),

    # ── Bouton flottant "À propos" ─────────────────────────────────────────
    html.Button("ℹ️", id="fab-about-btn", className="fab-about", title="À propos du projet", n_clicks=0),

    # ── Modale "À propos" ─────────────────────────────────────────────────
    html.Div([
        html.Div([
            # Header
            html.Div([
                html.H2("À propos du projet G05", className="modal-title"),
                html.Button("✕", id="modal-close-btn", className="modal-close", n_clicks=0),
            ], className="modal-header"),

            # Body
            html.Div([

                # Objectif
                html.Div([
                    html.P("🎯  OBJECTIF", className="modal-section-title"),
                    html.P("Ce projet compare trois optimiseurs de descente de gradient — "
                           "AdamW, SGD avec momentum de Nesterov et Adafactor — "
                           "appliqués au fine-tuning du modèle DistilBERT sur le dataset de classification "
                           "de texte AG News (4 classes : World, Sports, Business, Sci/Tech).",
                           className="modal-text"),
                    html.P("L'objectif principal est d'identifier lequel de ces optimiseurs offre "
                           "le meilleur compromis entre performance (F1 Macro, Accuracy), "
                           "stabilité d'entraînement et généralisation, dans un cadre de recherche "
                           "d'hyperparamètres aléatoire (Random Search).",
                           className="modal-text"),
                    html.Div([
                        html.Span("DistilBERT", className="modal-tag"),
                        html.Span("AG News", className="modal-tag"),
                        html.Span("Classification 4 classes", className="modal-tag"),
                        html.Span("AdamW", className="modal-tag modal-tag-green"),
                        html.Span("SGD + Nesterov", className="modal-tag modal-tag-green"),
                        html.Span("Adafactor", className="modal-tag modal-tag-green"),
                    ], style={"marginTop":"14px"}),
                ], className="modal-section"),

                # Méthodologie
                html.Div([
                    html.P("🔬  MÉTHODOLOGIE", className="modal-section-title"),
                    html.Div([
                        html.Div([
                            html.Div("1", className="modal-step-num"),
                            html.Div([
                                html.Strong("Préparation des données — "),
                                "Undersampling équilibré du dataset AG News pour garantir "
                                "une représentation égale des 4 classes et éviter tout biais "
                                "de classe dominant."
                            ], className="modal-step-text"),
                        ], className="modal-step"),
                        html.Div([
                            html.Div("2", className="modal-step-num"),
                            html.Div([
                                html.Strong("Random Search — "),
                                "Exploration aléatoire de l'espace des hyperparamètres : "
                                "learning rate (1e-6 → 5e-4), batch size, warmup steps et "
                                "nombre d'époques. Chaque trial est évalué sur F1 Macro et Accuracy."
                            ], className="modal-step-text"),
                        ], className="modal-step"),
                        html.Div([
                            html.Div("3", className="modal-step-num"),
                            html.Div([
                                html.Strong("Analyse de convergence — "),
                                "Suivi des courbes de loss et de F1 à chaque step pour "
                                "identifier la vitesse de convergence et la stabilité "
                                "de chaque optimiseur."
                            ], className="modal-step-text"),
                        ], className="modal-step"),
                        html.Div([
                            html.Div("4", className="modal-step-num"),
                            html.Div([
                                html.Strong("Analyse de la sharpness — "),
                                "Mesure de la netteté du minimum convergé selon Keskar et al. (2017). "
                                "Un minimum plat (faible sharpness) est associé à une meilleure "
                                "généralisation hors distribution."
                            ], className="modal-step-text"),
                        ], className="modal-step"),
                        html.Div([
                            html.Div("5", className="modal-step-num"),
                            html.Div([
                                html.Strong("Loss landscape 1D — "),
                                "Visualisation de la géométrie de la surface de perte autour du "
                                "minimum trouvé, selon la méthode Filter Normalization de Li et al. (2018)."
                            ], className="modal-step-text"),
                        ], className="modal-step"),
                    ]),
                ], className="modal-section"),

                # Stack technique
                html.Div([
                    html.P("🛠️  STACK TECHNIQUE", className="modal-section-title"),
                    html.Div([
                        html.Span("PyTorch", className="modal-tag modal-tag-orange"),
                        html.Span("HuggingFace Transformers", className="modal-tag modal-tag-orange"),
                        html.Span("DistilBERT-base-uncased", className="modal-tag modal-tag-orange"),
                        html.Span("Plotly Dash", className="modal-tag"),
                        html.Span("Pandas / NumPy", className="modal-tag"),
                        html.Span("SciPy", className="modal-tag"),
                    ]),
                ], className="modal-section"),

                # Équipe + retour
                html.Div([
                    html.P("👥  ÉQUIPE G05", className="modal-section-title"),
                    html.P("AYONTA  ·  BAMOGO  ·  KOULOU", className="modal-text",
                           style={"fontWeight":"600","fontSize":"15px","letterSpacing":"0.05em"}),
                    html.P("Projet P01 — 2026", className="modal-text",
                           style={"color":"var(--faint)","fontSize":"12px","marginTop":"4px"}),
                ], className="modal-section", style={"marginBottom":"8px"}),

                # Bouton retour
                html.Button(
                    ["← ", "Revenir au dashboard"],
                    id="modal-back-btn", className="modal-back-btn", n_clicks=0,
                ),

            ], className="modal-body"),
        ], className="modal-box"),
    ], id="modal-about", className="modal-overlay hidden"),

], id="app-root", style={"minHeight":"100vh"})

# ══════════════════════════════════════════════════════════════════════════════
# 🔄 CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("theme-store","data"),
    Input("theme-toggle-btn","n_clicks"),
    State("theme-store","data"),
    prevent_initial_call=True,
)
def toggle_theme(n, current):
    return "dark" if current == "light" else "light"

# ── Modal open/close ─────────────────────────────────────────────────────────
@app.callback(
    Output("modal-about","className"),
    [Input("fab-about-btn","n_clicks"),
     Input("modal-close-btn","n_clicks"),
     Input("modal-back-btn","n_clicks")],
    State("modal-about","className"),
    prevent_initial_call=True,
)
def toggle_modal(open_clicks, close_clicks, back_clicks, current_class):
    from dash import ctx
    trigger = ctx.triggered_id
    if trigger == "fab-about-btn":
        return "modal-overlay"          # ouvrir
    return "modal-overlay hidden"       # fermer (close ou back)

# Applique data-theme sur <html> — change TOUTES les CSS variables d'un coup
clientside_callback(
    """
    function(theme) {
        document.documentElement.setAttribute('data-theme', theme || 'light');
        return window.dash_clientside.no_update;
    }
    """,
    Output("app-root","id"),   # output fictif (id ne change pas)
    Input("theme-store","data"),
    prevent_initial_call=False,
)

@app.callback(
    Output("tab-content","children"),
    [Input("main-tabs","value"), Input("theme-store","data")],
)
def render_tab(tab, theme_name):
    T = THEMES.get(theme_name or "light", THEMES["light"])
    def G(id_, fig): return dcc.Graph(id=id_, figure=fig, config=PLOTLY_CONFIG)
    def HR(): return html.Hr(className="g05-hr")

    if tab == "tab-overview":
        return html.Div([
            sh("Vue d'Ensemble","Distribution des données et comparaison multi-dimensionnelle","📊"),
            html.Div([G("g-under", create_undersampling_chart(T))], style={"marginBottom":"32px"}),
            html.Div([
                html.Div([G("g-radar", create_radar_chart(T) if HAS_SHARP else create_boxplot(T))],
                         style={"width":"48%","display":"inline-block"}),
                html.Div([G("g-corr-ov", create_correlation_matrix(T))],
                         style={"width":"48%","display":"inline-block","marginLeft":"4%"}),
            ]),
            callout("L'undersampling garantit une représentation équitable. "+
                    ("Radar : normalisation 0→1." if HAS_SHARP else "Radar nécessite compute_landscape.py."),
                    "info" if HAS_SHARP else "warning"),
        ])
    elif tab == "tab-convergence":
        return html.Div([
            sh("Courbes de Convergence","Évolution loss et F1 par optimiseur","📈"),
            html.Div([
                html.Span("Options d'affichage",className="option-label"),
                dcc.Checklist(id="conv-show-all",
                    options=[{"label":"  Afficher tous les trials en arrière-plan","value":"yes"}],
                    value=[],inline=True,
                    inputStyle={"marginRight":"8px","accentColor":"var(--accent)"},
                    labelStyle={"color":"var(--text-light)","fontSize":"13px","fontFamily":FONT}),
            ],className="option-panel"),
            G("g-conv", create_convergence_plot(T,False)),
            callout("Trait épais = meilleur run. Loss lissée (fenêtre=8).","info"),
        ])
    elif tab == "tab-hyperparam":
        return html.Div([
            sh("Hyperparamètres","Impact du learning rate sur les performances","⚙️"),
            radio_grp("scatter-metric",
                [{"label":"F1 Macro","value":"final_val_f1"},
                 {"label":"Accuracy","value":"final_val_accuracy"}],"final_val_f1","Métrique"),
            html.Div([
                html.Span("Ligne de tendance",className="radio-label"),
                dcc.Checklist(id="scatter-trend",
                    options=[{"label":"  Afficher","value":"yes"}],value=[],inline=True,
                    inputStyle={"marginRight":"6px","accentColor":"var(--accent)"},
                    labelStyle={"color":"var(--text-light)","fontSize":"13px","fontFamily":FONT}),
            ],className="radio-group"),
            G("g-scatter", create_scatter_plot(T,"final_val_f1",False)),
            HR(),
            sh("Heatmaps Optimiseur × LR","Zones favorables par plage de learning rate"),
            html.Div([
                html.Div([G("g-hm-f1",  create_heatmap(T,"final_val_f1"))],      style={"width":"48%","display":"inline-block"}),
                html.Div([G("g-hm-acc", create_heatmap(T,"final_val_accuracy"))], style={"width":"48%","display":"inline-block","marginLeft":"4%"}),
            ]),
            callout("Valeur max par cellule. Vide = aucun trial.","info"),
        ])
    elif tab == "tab-advanced":
        return html.Div([
            sh("Analyses Avancées","Sharpness vs Performance","🔬"),
            G("g-sharp-perf",create_sharpness_vs_performance(T)) if HAS_SHARP
            else callout("Lance compute_landscape.py pour activer cette section.","warning"),
            HR() if HAS_SHARP else None,
            sh("Matrice de Corrélation","Relations entre hyperparamètres et métriques"),
            G("g-corr-adv",create_correlation_matrix(T)),
            callout("Rouge=corrélation positive. Bleu=inverse. Corrélation ≠ causalité.","info","💡"),
        ])
    elif tab == "tab-landscape":
        return html.Div([
            sh("Loss Landscape","Géométrie de la surface de perte","🏔️"),
            G("g-land",create_landscape_plot(T)),
            callout("Minimum plat (faible S) → meilleure généralisation.","success","✅"),
            HR(),
            sh("Sharpness par Optimiseur","Mesure Keskar et al. (2017)"),
            G("g-sharp",create_sharpness_plot(T)),
            callout("S=(1/N)Σ|L(θ+ε·dᵢ)−L(θ)|. Valeur faible=plat=robustesse.","info"),
        ])
    elif tab == "tab-distributions":
        return html.Div([
            sh("Distributions","Statistiques descriptives par optimiseur","📦"),
            radio_grp("boxplot-metric",
                [{"label":"F1 Macro","value":"final_val_f1"},
                 {"label":"Accuracy","value":"final_val_accuracy"},
                 {"label":"Temps (min)","value":"train_time_min"}],"final_val_f1","Métrique"),
            G("g-box",create_boxplot(T,"final_val_f1")),
            callout("Boxplot=médiane+quartiles. Points=chaque trial individuel.","info"),
        ])
    elif tab == "tab-summary":
        return html.Div([
            sh("Tableau Récapitulatif","Synthèse complète des résultats P01","📋"),
            build_summary_table(),
            callout("Cliquer sur un en-tête pour trier. Ligne surlignée=meilleur Best F1.","info"),
        ])
    return html.Div()


@app.callback(Output("g-conv","figure"),
    [Input("conv-show-all","value"),Input("theme-store","data")],prevent_initial_call=True)
def upd_conv(v,th): return create_convergence_plot(THEMES.get(th,THEMES["light"]),"yes" in (v or []))

@app.callback(Output("g-scatter","figure"),
    [Input("scatter-metric","value"),Input("scatter-trend","value"),Input("theme-store","data")],prevent_initial_call=True)
def upd_scatter(m,t,th): return create_scatter_plot(THEMES.get(th,THEMES["light"]),m,"yes" in (t or []))

@app.callback(Output("g-box","figure"),
    [Input("boxplot-metric","value"),Input("theme-store","data")],prevent_initial_call=True)
def upd_box(m,th): return create_boxplot(THEMES.get(th,THEMES["light"]),m)


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n"+"═"*70)
    print("  🚀 G05 Dashboard — Dark/Light Mode COMPLET (CSS Variables)")
    print("═"*70)
    print(f"  CSV           : {'✅' if HAS_CSV   else '❌'}")
    print(f"  Historiques   : {'✅' if HAS_HIST  else '❌'}")
    print(f"  Sharpness     : {'✅' if HAS_SHARP else '❌'}")
    print(f"  Landscape     : {'✅' if HAS_LAND  else '❌'}")
    print(f"  Undersampling : {'✅' if HAS_UNDER else '❌'}")
    print("═"*70)
    print("  🌐 http://127.0.0.1:8050")
    print("  🎨 Toggle ☀️/🌙 → TOUT bascule : fond, header, KPIs, tabs, tableaux, graphiques")
    print("═"*70+"\n")
    app.run(debug=False, host="0.0.0.0", port=8050)