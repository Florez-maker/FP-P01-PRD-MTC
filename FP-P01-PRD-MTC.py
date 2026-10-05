import warnings
warnings.filterwarnings("ignore")

import io
import os
import re
import unicodedata

from scipy.stats import gaussian_kde
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dashboard Predictivo Producción — FarmPrecision",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_FILE = "1a54dac1-ec50-4498-ad65-871e80328f60 (1).xlsx"

COLORS = {
    "primary":  "#1b60a7",
    "success":  "#2ca02c",
    "danger":   "#d62728",
    "warning":  "#F1C40F",
    "info":     "#17becf",
    "bg":       "#F0F4F8",
    "pred":     "#7a3db8",
    "Q1": "brown",
    "Q2": "orange",
    "Q3": "green",
    "Q4": "dodgerblue",
}

MONTH_MAP = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dic": 12,
}

NUM_COLS = [
    "ano_siembra", "ano", "ton_ppto", "ppr", "racimos", "ton_real",
    "dias_ciclo_cos", "dias_ciclo_enf", "dias_ciclo_poli", "area_neta",
]

TEXT_COLS = ["gran_zona", "finca", "bloque", "lote", "material", "estado", "mes"]

# Features estructurales del modelo (sin racimos/ppr/ton_ppto: fuga de información)
FEATURES_BASE = [
    "edad", "area_neta", "dias_ciclo_cos", "dias_ciclo_enf",
    "dias_ciclo_poli", "mes_sin", "mes_cos",
]

EXCLUSIONES_TARGET = {
    "ton_real": ["racimos", "ppr", "ton_ppto", "ton_ha"],
    "ppr":      ["racimos", "ton_real", "ton_ha", "ton_ppto"],
}


# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A3D62 0%, #1A6B3C 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }
    .kpi-card {
        background: white; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #1b60a7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .kpi-label { font-size: 0.72rem; font-weight: 600; color: #7A8899;
                 text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1C2B3A; line-height: 1.1; }
    .kpi-sub   { font-size: 0.72rem; color: #7A8899; margin-top: 2px; }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #0A3D62;
        border-bottom: 2px solid #1A6B3C;
        padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem;
    }
    [data-testid="stSidebar"] { background: #F0F4F8; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.85rem;
    }
    .upload-zone {
        border: 2px dashed #1b60a7; border-radius: 10px;
        padding: 2rem; text-align: center; background: #f0f7ff;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 1. CARGA Y NORMALIZACIÓN DE DATOS
# ──────────────────────────────────────────────────────────────

def normalize_colname(col: str) -> str:
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col)
    col = "".join(ch for ch in col if not unicodedata.combining(ch))
    col = col.replace("%", "pct")
    col = col.replace("/", "_").replace("-", "_").replace("+", "_").replace(".", "_")
    col = re.sub(r"[()\[\]{}]", "", col)
    col = re.sub(r"[^a-z0a-z0-9_]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def sanitize_key(s):
    return (str(s).replace(" ", "_").replace("/", "_").replace("-", "_")
                  .replace(".", "_").replace("(", "").replace(")", ""))


def cargar_dataset_produccion(file_bytes: bytes, file_name: str = "") -> pd.DataFrame:
    if str(file_name).lower().endswith(".csv"):
        try:
            df_raw = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", dtype=object)
        except Exception:
            df_raw = pd.read_csv(io.BytesIO(file_bytes), sep=",", dtype=object)
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    df = df_raw.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    requeridas = set(TEXT_COLS) | set(NUM_COLS)
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")

    for c in TEXT_COLS:
        df[c] = df[c].astype(str).str.strip().str.upper()

    for c in NUM_COLS:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    df["mes_num"] = df["mes"].str.lower().map(MONTH_MAP)
    df = df.dropna(subset=["mes_num", "ano"]).copy()

    df["fecha"] = pd.to_datetime(
        dict(year=df["ano"].astype(int), month=df["mes_num"].astype(int), day=1)
    )

    df["edad"] = df["ano"] - df["ano_siembra"]
    df["ton_ha"] = np.where(df["area_neta"] > 0, df["ton_real"] / df["area_neta"], np.nan)
    df["ton_ppto_ha"] = np.where(df["area_neta"] > 0, df["ton_ppto"] / df["area_neta"], np.nan)
    df["brecha_ton"] = df["ton_real"] - df["ton_ppto"]
    df["cumplimiento_pct"] = np.where(df["ton_ppto"] > 0, df["ton_real"] / df["ton_ppto"] * 100, np.nan)

    df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)

    # Corte histórico: los meses futuros traen Ton Real = 0 (solo presupuesto)
    ultimo_con_cosecha = df.loc[df["ton_real"] > 0, "fecha"].max()
    if pd.isna(ultimo_con_cosecha):
        raise ValueError("El dataset no contiene producción real (Ton Real > 0).")

    df["es_historico"] = df["fecha"] <= ultimo_con_cosecha

    print(f"  ✓ Dataset producción estandarizado: {df.shape}")
    print(f"  ✓ Último mes con cosecha registrada: {ultimo_con_cosecha.date()}")

    return df


def construir_serie_mensual(df: pd.DataFrame) -> tuple:
    """
    Serie mensual agregada + detección del mes en curso.

    Un mes 'en curso' es el último con cosecha cuyo total queda por
    debajo del 50 % de la mediana de los 3 meses anteriores (cosecha
    parcial). El modelo se entrena solo con meses cerrados.
    """
    d = df.copy()
    d["ton_real_hist"] = np.where(d["es_historico"], d["ton_real"], np.nan)

    mensual = (
        d.groupby("fecha", as_index=False)
        .agg(ton_real=("ton_real_hist", "sum"), ton_ppto=("ton_ppto", "sum"))
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    mensual["ton_ppto"] = np.where(mensual["ton_ppto"] > 0, mensual["ton_ppto"], np.nan)

    con_cosecha = mensual[mensual["ton_real"] > 0].copy()
    mes_curso = None
    if len(con_cosecha) >= 4:
        last = con_cosecha["fecha"].max()
        val_last = con_cosecha.loc[con_cosecha["fecha"] == last, "ton_real"].iloc[0]
        med_prev = con_cosecha[con_cosecha["fecha"] < last].tail(3)["ton_real"].median()
        if val_last < 0.5 * med_prev:
            mes_curso = last

    corte = (mes_curso - pd.DateOffset(months=1)) if mes_curso is not None else con_cosecha["fecha"].max()

    return mensual, corte, mes_curso


# ════════════════════════════════════════════════════
# 2. MODELO DE PROYECCIÓN TEMPORAL
# ════════════════════════════════════════════════════

def construir_proyeccion(serie: pd.DataFrame, horizon: int = 12,
                         recent_periods: int = 12, phi: float = 0.9) -> pd.DataFrame:
    """
    Línea base temporal con tendencia amortiguada + estacionalidad mensual.

    1. Tendencia lineal sobre los últimos N meses.
    2. Factor estacional por mes calendario.
    3. Amortiguación de la tendencia (phi) para no extrapolar en falso.
    4. Suavizado con el último valor observado.
    """
    if serie is None or len(serie) < 3:
        return pd.DataFrame()

    hist = serie.sort_values("fecha").copy()
    hist = hist[hist["ton_real"].notna()]

    recent = hist.tail(min(recent_periods, len(hist))).copy()
    recent["idx"] = np.arange(len(recent))

    if recent["ton_real"].sum() > 0:
        slope, intercept = np.polyfit(recent["idx"], recent["ton_real"], 1)
    else:
        slope, intercept = 0.0, float(recent["ton_real"].mean())

    media = recent["ton_real"].mean()
    if media > 0:
        est = recent.assign(m=recent["fecha"].dt.month).groupby("m")["ton_real"].mean()
        factores = (est / media).to_dict()
    else:
        factores = {}

    last_date = hist["fecha"].max()
    last_val = float(recent["ton_real"].iloc[-1])

    rows = []
    for step in range(1, horizon + 1):
        fdate = last_date + pd.DateOffset(months=step)
        damp = sum(phi ** k for k in range(1, step + 1))
        base = max(0.0, intercept + slope * (len(recent) - 1) + slope * damp)
        pred = max(0.0, base * factores.get(fdate.month, 1.0))
        pred = 0.65 * pred + 0.35 * last_val
        rows.append({"fecha": fdate, "ton_pred": pred, "tipo": "Proyeccion"})
        last_val = pred

    fc = pd.DataFrame(rows)

    hout = hist[["fecha", "ton_real"]].copy()
    hout["ton_pred"] = np.nan
    hout["tipo"] = "Historico"

    return pd.concat([hout, fc], ignore_index=True)


def metricas_backtest(serie: pd.DataFrame, n_test: int = 6) -> dict:
    """
    Validación temporal: entrena con el pasado, prueba con los últimos
    n_test meses cerrados. Devuelve MAE, RMSE y MAPE.
    """
    vacio = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "n_test": n_test}

    if serie is None or len(serie) <= n_test + 3:
        return vacio

    train = serie.iloc[:-n_test]
    test = serie.iloc[-n_test:]

    fc = construir_proyeccion(train, horizon=n_test)
    if fc.empty:
        return vacio

    pred = fc[fc["tipo"] == "Proyeccion"]["ton_pred"].to_numpy()
    act = test["ton_real"].to_numpy()

    if len(pred) != len(act):
        return vacio

    err = act - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = act != 0
    mape = float(np.mean(np.abs(err[nz] / act[nz])) * 100) if nz.any() else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n_test": n_test}


# ════════════════════════════════════════════════════
# 3. FUNCIONES DE GRÁFICOS
# ════════════════════════════════════════════════════

def fig_boxplot(df, x_col, y_col, global_mean=None) -> go.Figure:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return go.Figure()
    d = df[[x_col, y_col]].dropna()
    if d.empty:
        return go.Figure()
    order = d.groupby(x_col)[y_col].median().sort_values().index.tolist()
    fig = go.Figure()
    for cat in order:
        vals = d[d[x_col] == cat][y_col].tolist()
        fig.add_trace(go.Box(y=vals, name=str(cat), marker_color=COLORS["primary"], showlegend=False))
    if global_mean is not None and np.isfinite(global_mean):
        fig.add_hline(y=global_mean, line_dash="dash", line_color="red",
                      annotation_text=f"Media global: {global_mean:.2f}",
                      annotation_position="top right")
    fig.update_layout(height=300, margin=dict(t=10, l=0, r=0, b=0), xaxis_tickangle=-45,
                      showlegend=False, template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def fig_distplot(vals: pd.Series, label: str) -> go.Figure:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    fig = go.Figure()
    if len(vals) < 2:
        return fig
    fig.add_trace(go.Histogram(x=vals, histnorm="probability density",
                               marker_color=COLORS["primary"], opacity=0.55,
                               nbinsx=25, name="Distribución"))
    if vals.nunique() >= 3:
        try:
            kde_x = np.linspace(vals.min(), vals.max(), 300)
            kde_y = gaussian_kde(vals)(kde_x)
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode="lines",
                                     line=dict(color=COLORS["primary"], width=2), name="KDE"))
        except Exception:
            pass
    fig.add_vline(x=float(vals.mean()), line_dash="dash", line_color=COLORS["danger"],
                  annotation_text=f"μ={vals.mean():.2f}", annotation_position="top right")
    fig.update_layout(height=300, margin=dict(t=0, l=0, r=0, b=0), xaxis_title=label,
                      yaxis_title="Densidad", showlegend=False, template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def calcular_cuadrantes(df: pd.DataFrame, prod_col: str = "ton_ha",
                        group_cols=("lote", "finca")) -> pd.DataFrame:
    gc = [c for c in group_cols if c in df.columns]
    if not gc or prod_col not in df.columns:
        return pd.DataFrame()

    agg = df.groupby(gc)[prod_col].agg(media="mean", de="std").reset_index()
    agg = agg.dropna(subset=["media", "de"])
    if agg.empty:
        return agg

    med_g = agg["media"].mean()
    de_g = agg["de"].mean()

    def asignar_q(media, de, med_ref, de_ref):
        if de < de_ref and media < med_ref:
            return "Q2"
        elif de < de_ref and media >= med_ref:
            return "Q4"
        elif de >= de_ref and media < med_ref:
            return "Q1"
        else:
            return "Q3"

    agg["q_global"] = agg.apply(lambda r: asignar_q(r["media"], r["de"], med_g, de_g), axis=1)

    finca_col = "finca" if "finca" in gc else gc[0]
    agg["q_local"] = ""
    for finca, sub in agg.groupby(finca_col):
        med_f = sub["media"].mean()
        de_f = sub["de"].mean()
        for idx in sub.index:
            agg.at[idx, "q_local"] = asignar_q(agg.at[idx, "media"], agg.at[idx, "de"], med_f, de_f)

    return agg


def fig_scatter_cuadrantes(df_q, q_col, x_label, y_label, title) -> go.Figure:
    if df_q.empty:
        return go.Figure()
    fig = go.Figure()
    mapping = {"Q1": "brown", "Q2": "orange", "Q3": "green", "Q4": "dodgerblue"}
    for q, color in mapping.items():
        sub = df_q[df_q[q_col] == q]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["media"], y=sub["de"], mode="markers",
            name=q, marker=dict(color=color, size=9, opacity=0.85,
                                line=dict(color="white", width=0.8)),
            text=sub["lote"].astype(str).tolist(),
            hovertemplate=(f"<b>%{{text}}</b><br>{x_label}: %{{x:.2f}}<br>"
                           f"{y_label}: %{{y:.2f}}<br>Cuadrante: {q}<extra></extra>"),
        ))
    mx = df_q["media"].mean()
    my = df_q["de"].mean()
    fig.add_vline(x=mx, line_dash="dash", line_color="black", line_width=0.8,
                  annotation_text=f"μ={mx:.2f}", annotation_position="top right",
                  annotation_font_size=10)
    fig.add_hline(y=my, line_dash="dash", line_color="black", line_width=0.8,
                  annotation_text=f"σ={my:.2f}", annotation_position="bottom right",
                  annotation_font_size=10)
    xmn, xmx = df_q["media"].min(), df_q["media"].max()
    ymn, ymx = df_q["de"].min(), df_q["de"].max()
    annot = [
        ("Q1\nBaja prod.\nAlta var.", (xmn + mx) / 2, (my + ymx) / 2, "brown"),
        ("Q2\nBaja prod.\nBaja var.", (xmn + mx) / 2, (ymn + my) / 2, "orange"),
        ("Q3\nAlta prod.\nAlta var.", (mx + xmx) / 2, (my + ymx) / 2, "green"),
        ("Q4\nAlta prod.\nBaja var.", (mx + xmx) / 2, (ymn + my) / 2, "dodgerblue"),
    ]
    for txt, ax, ay, col in annot:
        fig.add_annotation(x=ax, y=ay, text=txt, showarrow=False,
                           font=dict(size=9, color=col), opacity=0.5)
    fig.update_layout(title=dict(text=title, font_size=13),
                      xaxis_title=x_label, yaxis_title=y_label, height=480,
                      margin=dict(t=50, l=0, r=0, b=0), template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.12))
    return fig


# ════════════════════════════════════════════════════
# 4. SECCIONES DEL DASHBOARD
# ════════════════════════════════════════════════════

def seccion_kpis(data: pd.DataFrame):
    hist = data[data["es_historico"]]
    area = data.drop_duplicates("lote")["area_neta"].sum()
    ton_real = hist["ton_real"].sum()

    con_ppto = data[data["ton_ppto"] > 0]
    if not con_ppto.empty:
        ppto_total = con_ppto["ton_ppto"].sum()
        cumpl = ton_real / ppto_total * 100 if ppto_total > 0 else np.nan
        anios_ppto = ", ".join(str(a) for a in sorted(con_ppto["ano"].unique().astype(int)))
    else:
        ppto_total, cumpl, anios_ppto = 0.0, np.nan, "—"

    cols = st.columns(6)
    kpis = [
        ("Registros",      f"{len(data):,}",                                "📋"),
        ("Lotes",          str(data["lote"].nunique()),                     "🌿"),
        ("Área neta",      f"{area:,.1f} ha",                               "📐"),
        ("Ton real",       f"{ton_real:,.1f} t",                            "🏋️"),
        ("Ppto " + anios_ppto, f"{ppto_total:,.1f} t",                      "🎯"),
        ("Cumplimiento",   f"{cumpl:.1f} %" if pd.notna(cumpl) else "—",    "✅"),
    ]
    for col, (label, value, icon) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def tab_proyeccion(df, mensual, corte, mes_curso, horizon, recent_periods):
    st.markdown('<div class="section-title">Proyección de producción — tendencia amortiguada + estacionalidad</div>',
                unsafe_allow_html=True)

    serie_cerrada = mensual[mensual["fecha"] <= corte].copy()
    serie_modelo = serie_cerrada[serie_cerrada["ton_real"].notna()]

    if len(serie_modelo) < 3:
        st.warning("No hay suficiente historia cerrada para proyectar.")
        return

    fc = construir_proyeccion(serie_modelo, horizon=horizon, recent_periods=recent_periods)

    fig = go.Figure()

    hist = fc[fc["tipo"] == "Historico"]
    proj = fc[fc["tipo"] == "Proyeccion"]

    ppto = mensual[mensual["ton_ppto"].notna()]

    fig.add_trace(go.Scatter(x=hist["fecha"], y=hist["ton_real"], mode="lines+markers",
                             name="Producción real", line=dict(color=COLORS["success"], width=3),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=ppto["fecha"], y=ppto["ton_ppto"], mode="lines",
                             name="Presupuesto", line=dict(color=COLORS["warning"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=proj["fecha"], y=proj["ton_pred"], mode="lines+markers",
                             name="Proyección", line=dict(color=COLORS["pred"], width=3, dash="dot"),
                             marker=dict(size=5)))

    ref = mes_curso if mes_curso is not None else (
            corte + pd.DateOffset(months=1)
    )

    ref_plot = pd.Timestamp(ref).to_pydatetime()

    texto_ref = (
        "Mes en curso"
        if mes_curso is not None
        else "Inicio de proyección"
    )

    fig.add_shape(
        type="line",
        x0=ref_plot,
        x1=ref_plot,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color="#6b7280",
            width=1,
            dash="dash",
        ),
    )

    fig.add_annotation(
        x=ref_plot,
        y=1,
        xref="x",
        yref="paper",
        text=texto_ref,
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=4,
        font=dict(
            size=10,
            color="#6b7280",
        ),
    )

    fig.update_layout(height=480, margin=dict(t=30, l=0, r=0, b=0), hovermode="x unified",
                      template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      xaxis=dict(title="Fecha", gridcolor="rgba(0,0,0,0.08)"),
                      yaxis=dict(title="Toneladas", gridcolor="rgba(0,0,0,0.08)"))
    st.plotly_chart(fig, use_container_width=True, key="proj_serie")

    if mes_curso is not None:
        st.caption("⚠️ El último mes con cosecha queda por debajo del 50 % de la mediana previa: "
                   "se interpreta como mes en curso y el modelo entrena solo con meses cerrados.")

    st.markdown('<div class="section-title">Validación temporal del modelo (backtest)</div>',
                unsafe_allow_html=True)
    m = metricas_backtest(serie_modelo, n_test=6)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (t/mes)", f"{m['MAE']:,.0f}" if pd.notna(m["MAE"]) else "—")
    c2.metric("RMSE (t/mes)", f"{m['RMSE']:,.0f}" if pd.notna(m["RMSE"]) else "—")
    c3.metric("MAPE", f"{m['MAPE']:.1f} %" if pd.notna(m["MAPE"]) else "—")
    c4.metric("Ventana de prueba", f"{m['n_test']} meses")
    st.caption("Backtest sobre meses cerrados: el modelo se entrena sin la ventana de prueba. "
               "MAPE indica el error porcentual medio mensual de la línea base.")

    # ── Tabla de proyección vs presupuesto ──
    st.markdown('<div class="section-title">Detalle de la proyección vs presupuesto</div>',
                unsafe_allow_html=True)
    tabla = proj[["fecha", "ton_pred"]].copy()
    tabla["Período"] = tabla["fecha"].dt.strftime("%Y-%m")
    tabla = tabla.rename(columns={"ton_pred": "Proyección (t)"})

    ppto_fut = mensual[mensual["ton_ppto"].notna()][["fecha", "ton_ppto"]].copy()
    ppto_fut["Período"] = ppto_fut["fecha"].dt.strftime("%Y-%m")
    tabla = tabla.merge(ppto_fut[["Período", "ton_ppto"]], on="Período", how="left")
    tabla = tabla.rename(columns={"ton_ppto": "Ppto (t)"})
    tabla["Δ vs ppto (t)"] = tabla["Proyección (t)"] - tabla["Ppto (t)"]

    st.dataframe(
        tabla[["Período", "Proyección (t)", "Ppto (t)", "Δ vs ppto (t)"]].round(1),
        use_container_width=True, hide_index=True,
    )


def tab_produccion(df, mensual):
    st.markdown('<div class="section-title">Distribución y variabilidad de la productividad</div>',
                unsafe_allow_html=True)

    c_sel1, _, _ = st.columns(3)
    agrupar = c_sel1.selectbox("Agrupar por:", ["bloque", "material", "estado", "gran_zona"],
                               key="prod_group")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            fig_boxplot(df, agrupar, "ton_ha", global_mean=df["ton_ha"].mean()),
            use_container_width=True, key=sanitize_key(f"prod_box_{agrupar}"),
        )
    with c2:
        st.plotly_chart(fig_distplot(df["ton_ha"], "ton/ha mensual"),
                        use_container_width=True, key="prod_dist")

    st.markdown('<div class="section-title">Estructura jerárquica de la producción</div>',
                unsafe_allow_html=True)
    df_tm = df.dropna(subset=["ton_ha"])
    if not df_tm.empty:
        fig_tree = px.treemap(
            df_tm,
            path=[px.Constant("Total"), "gran_zona", "bloque", "lote"],
            values=df_tm["ton_real"].abs() + 0.01,
            color=df_tm["ton_ha"],
            color_continuous_scale="Spectral",
            custom_data=["material", "area_neta", "edad"],
        )
        fig_tree.update_traces(
            texttemplate="MATERIAL: %{customdata[0]}<br>AREA: %{customdata[1]:.1f} ha<br>EDAD: %{customdata[2]:.0f} años",
            hovertemplate="%{label}<br>%{value:.2f} t",
        )
        fig_tree.update_layout(height=380, template="plotly_white",
                               margin=dict(t=10, l=0, r=0, b=0),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_tree, use_container_width=True, key="prod_treemap")

    # ── Real vs presupuesto mensual (meses con presupuesto) ──
    con_ppto = mensual[mensual["ton_ppto"].notna()].copy()
    if not con_ppto.empty:
        st.markdown('<div class="section-title">Producción real vs presupuesto — meses presupuestados</div>',
                    unsafe_allow_html=True)
        fig_pp = go.Figure()
        fig_pp.add_trace(go.Bar(x=con_ppto["fecha"], y=con_ppto["ton_ppto"], name="Presupuesto",
                                marker_color=COLORS["warning"], opacity=0.7))
        fig_pp.add_trace(go.Bar(x=con_ppto["fecha"], y=con_ppto["ton_real"], name="Real",
                                marker_color=COLORS["success"]))
        cumpl = con_ppto["ton_real"] / con_ppto["ton_ppto"] * 100
        fig_pp.add_trace(go.Scatter(x=con_ppto["fecha"], y=cumpl, name="Cumplimiento %",
                                    yaxis="y2", line=dict(color=COLORS["primary"], width=2),
                                    mode="lines+markers"))
        fig_pp.update_layout(barmode="group", height=380, margin=dict(t=10, l=0, r=0, b=0),
                             template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)",
                             yaxis=dict(title="Toneladas"),
                             yaxis2=dict(title="Cumplimiento %", overlaying="y", side="right",
                                         range=[0, max(150, float(cumpl.max()) * 1.2)]),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_pp, use_container_width=True, key="prod_ppto")

    # ── Resumen anual ──
    st.markdown('<div class="section-title">Resumen anual</div>', unsafe_allow_html=True)
    hist = df[df["es_historico"]]
    anual = hist.groupby("ano", as_index=False).agg(
        ton_real=("ton_real", "sum"), ton_ppto=("ton_ppto", "sum"),
        racimos=("racimos", "sum"), area=("area_neta", "mean"),
    )
    anual["ton_ha"] = np.where(anual["area"] > 0, anual["ton_real"] / anual["area"], np.nan)
    anual = anual.rename(columns={"ano": "Año", "ton_real": "Ton real", "ton_ppto": "Ton ppto",
                                  "racimos": "Racimos", "area": "Área ha", "ton_ha": "Ton/ha"})
    st.dataframe(anual.round(2), use_container_width=True, hide_index=True)

    # ── Ranking de lotes ──
    st.markdown('<div class="section-title">Ranking productivo por lote</div>', unsafe_allow_html=True)
    lote = (
        hist.groupby(["lote", "bloque", "material", "estado"], as_index=False)
        .agg(area=("area_neta", "mean"), edad=("edad", "mean"),
             ton_real=("ton_real", "sum"), ton_ppto=("ton_ppto", "sum"), racimos=("racimos", "sum"))
    )
    lote["brecha"] = lote["ton_real"] - lote["ton_ppto"]
    lote["cumplimiento"] = np.where(lote["ton_ppto"] > 0, lote["ton_real"] / lote["ton_ppto"] * 100, np.nan)
    lote["ton_ha"] = np.where(lote["area"] > 0, lote["ton_real"] / lote["area"], np.nan)

    orden = st.radio("Ordenar lotes por:",
                     ["Mayor brecha negativa", "Mayor producción real",
                      "Mayor cumplimiento", "Mayor ton/ha"],
                     horizontal=True, key="rank_orden")
    if orden == "Mayor brecha negativa":
        lote = lote.sort_values("brecha", ascending=True)
    elif orden == "Mayor producción real":
        lote = lote.sort_values("ton_real", ascending=False)
    elif orden == "Mayor cumplimiento":
        lote = lote.sort_values("cumplimiento", ascending=False, na_option="last")
    else:
        lote = lote.sort_values("ton_ha", ascending=False)

    st.dataframe(lote.head(30).round(2), use_container_width=True, hide_index=True)
    st.caption("El ranking de mayor brecha negativa prioriza lotes para validación de campo. "
               "Cumplimiento solo se calcula sobre meses con presupuesto.")


def tab_cuadrantes(df):
    st.markdown('<div class="section-title">Análisis de Cuadrantes — Productividad vs Variabilidad</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    modo_q = c1.radio("Escala de referencia:", ["Global", "Por Finca"], horizontal=True, key="q_modo")
    group_q = c2.selectbox("Nivel de agrupación:", ["Lote + Finca", "Lote", "Finca"], key="q_group")

    group_map = {
        "Lote + Finca": ("lote", "finca"),
        "Lote": ("lote",),
        "Finca": ("finca",),
    }

    df_q = calcular_cuadrantes(df, prod_col="ton_ha", group_cols=group_map[group_q])
    if df_q.empty:
        st.warning("No hay suficientes datos para calcular cuadrantes.")
        return

    q_col = "q_global" if modo_q == "Global" else "q_local"
    label_col = group_map[group_q][0]

    st.plotly_chart(
        fig_scatter_cuadrantes(df_q, q_col,
                               "Productividad media (ton/ha)", "Variabilidad (DE)",
                               f"Cuadrantes — Nivel: {group_q} · Referencia: {modo_q}"),
        use_container_width=True, key=sanitize_key(f"q_scatter_{modo_q}_{group_q}"),
    )

    resumen = df_q.groupby(q_col).agg(
        N=("media", "count"), Prod_media=("media", "mean"), DE_media=("de", "mean"),
    ).round(2).reset_index()

    st.dataframe(resumen.rename(columns={q_col: "Cuadrante"}),
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Detalle por cuadrante</div>', unsafe_allow_html=True)
    filtro_q = st.multiselect("Filtrar cuadrante(s):", ["Q1", "Q2", "Q3", "Q4"],
                              default=["Q1", "Q2", "Q3", "Q4"], key="q_filter")

    df_qd = df_q[df_q[q_col].isin(filtro_q)].copy()
    extra = df.drop_duplicates("lote").set_index("lote")

    df_qd["material"] = df_qd["lote"].map(extra["material"])
    df_qd["estado"] = df_qd["lote"].map(extra["estado"])
    df_qd["area_ha"] = df_qd["lote"].map(extra["area_neta"])
    df_qd["edad"] = df_qd["lote"].map(extra["edad"])

    st.dataframe(
        df_qd[[label_col, "media", "de", q_col, "material", "estado", "area_ha", "edad"]]
        .sort_values("media").round(3),
        use_container_width=True, hide_index=True,
    )
    st.caption("Q1: baja producción / alta variabilidad → revisar manejo. "
               "Q2: baja y estable → deficiencia estructural. "
               "Q3: alta producción / alta variabilidad → potencial con variación. "
               "Q4: alta y estable → lotes de referencia.")


def tab_modelo(df):
    st.markdown('<div class="section-title">Modelo Predictivo — Random Forest (datos de producción)</div>',
                unsafe_allow_html=True)

    target_sel = st.radio(
        "Variable objetivo:",
        ["ton_real (Toneladas)", "ppr (Peso por racimo, kg)"],
        horizontal=True, key="rf_target",
    )
    target = "ton_real" if target_sel.startswith("ton_real") else "ppr"

    exclusiones = EXCLUSIONES_TARGET[target]

    df_model = df[df["es_historico"]].copy()
    if target == "ppr":
        df_model = df_model[df_model["racimos"] > 0]

    feature_pool = [c for c in FEATURES_BASE if c in df_model.columns]

    dummies = pd.get_dummies(df_model["material"], prefix="mat")
    X = pd.concat([df_model[feature_pool], dummies], axis=1)
    y = df_model[target].astype(float)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    keep = [c for c in X.columns if X[c].notna().any() and X[c].dropna().nunique() > 1]
    X = X[keep]

    st.info(
        "Variables excluidas por fuga de información o falta de cobertura: "
        + ", ".join(exclusiones)
        + ". Racimos y PPR componen directamente las toneladas cosechadas "
        "(Ton Real = Racimos × PPR / 1000) y el presupuesto solo existe para 2026, "
        "por lo que no se usan como predictors."
    )

    df_model = df_model.loc[X.index]
    y = df_model[target].astype(float)

    n = len(df_model)
    if n < 30:
        st.warning(f"Registros insuficientes tras limpiar: {n}. Mínimo 30.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    rf = RandomForestRegressor(
        n_estimators=500, max_depth=12, min_samples_leaf=3,
        max_features=0.8, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_imp, y_train)
    y_pred = rf.predict(X_test_imp)

    r2 = r2_score(y_test, y_pred)
    err = y_test.to_numpy() - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (test)", f"{r2:.3f}")
    m2.metric("MAE", f"{mae:,.2f}")
    m3.metric("RMSE", f"{rmse:,.2f}")
    m4.metric("Registros", f"{n:,}")

    imp_desc = pd.DataFrame({
        "variable": X.columns, "importancia": rf.feature_importances_,
    }).sort_values("importancia", ascending=False)

    top5 = imp_desc["variable"].tolist()[:5]
    if top5:
        cols_top = st.columns(len(top5))
        for i, feat in enumerate(top5):
            vals = df_model[feat].dropna()
            if len(vals) == 0:
                continue
            p25, p75 = vals.quantile([0.25, 0.75])
            mu_low = df_model.loc[df_model[feat] <= p25, target].mean()
            mu_high = df_model.loc[df_model[feat] >= p75, target].mean()
            delta = mu_high - mu_low
            color = COLORS["success"] if delta > 0 else COLORS["danger"]
            cols_top[i].markdown(f"""
                <div class="kpi-card" style="border-left:4px solid {color}; padding:8px; border-radius:6px;">
                  <div style="font-weight:700;">{feat}</div>
                  <div style="font-size:.85rem; margin-top:.35rem;">
                    P25 ≤ {p25:.2f}: media <b>{mu_low:.2f}</b><br>
                    P75 ≥ {p75:.2f}: media <b>{mu_high:.2f}</b><br>
                    <span style="color:{color};font-weight:700;">Δ = {delta:+.2f}</span>
                  </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        imp_plot = imp_desc.head(20).sort_values("importancia", ascending=True)
        fig_imp = px.bar(imp_plot, x="importancia", y="variable", orientation="h",
                         color="importancia", color_continuous_scale="Greens")
        fig_imp.update_layout(height=480, margin=dict(t=10, l=0, r=0, b=0),
                              template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_imp, use_container_width=True, key="rf_importance")

    with c2:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                      marker=dict(color=COLORS["primary"], opacity=0.6, size=6),
                                      name="test"))
        lim_min = float(min(y_test.min(), np.min(y_pred)))
        lim_max = float(max(y_test.max(), np.max(y_pred)))
        fig_pred.add_trace(go.Scatter(x=[lim_min, lim_max], y=[lim_min, lim_max],
                                      mode="lines", line=dict(color="red", dash="dash"),
                                      name="ideal"))
        fig_pred.update_layout(
            title=f"Predicho vs Real — R² = {r2:.3f}",
            xaxis_title="Real", yaxis_title="Predicho", height=520,
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pred, use_container_width=True, key="rf_pred_vs_real")

    st.caption(
        "Modelo estructural de producción: predice el desempeño mensual del lote "
        "a partir de edad, área, días de ciclo, estacionalidad y material genético. "
        "Es un modelo de referencia operativa; el diagnóstico causal agronómico "
        "(suelo, clima, nutrición, sanidad) requiere integrar esas fuentes en una "
        "siguiente versión del pipeline."
    )


# ════════════════════════════════════════════════════
# 5. SIDEBAR
# ════════════════════════════════════════════════════


def render_sidebar():
    with st.sidebar:
        try:
            img = Image.open("logo_sidebar.png")
            st.image(img, width=260)
        except Exception:
            st.markdown("## 🌴 FarmPrecision")

        st.markdown("---")
        st.markdown("### 📂 Cargar datos")
        uploaded = st.file_uploader(
            "Sube tu archivo Excel o CSV",
            type=["xlsx", "xls", "csv"],
            help="Dataset de producción: finca, lote, año, mes, producción, presupuesto, racimos, días de ciclo y área.",
        )

        st.markdown("---")
        st.markdown("### ℹ️ Columnas requeridas")
        st.markdown("""
        | **Variable** | **Nombre interno** | **Variantes aceptadas** |
        |:-------------|:-------------------|:-------------------------|
        | **Gran Zona** | `gran_zona` | gran zona, gran_zona, zona |
        | **Finca** | `finca` | finca, farm, hacienda |
        | **Bloque** | `bloque` | bloque, block |
        | **Lote** | `lote` | lote, lot, parcela |
        | **Material** | `material` | material, genotipo, variedad |
        | **Estado** | `estado` | estado, status |
        | **Año siembra** | `ano_siembra` | año siembra, ano siembra, año de siembra |
        | **Año** | `ano` | año, ano, year, campaña |
        | **Mes** | `mes` | mes, month |
        | **Ton. presupuesto** | `ton_ppto` | ton ppto, ton presupuesto, presupuesto |
        | **PPR** | `ppr` | ppr |
        | **Racimos** | `racimos` | racimos, bunches |
        | **Ton. real** | `ton_real` | ton real, producción real |
        | **Días de ciclo** | `dias_ciclo_cos` | días de ciclo, dias ciclo |
        | **Área neta** | `area_neta` | área neta, area neta, hectáreas |
        """)

        st.caption(
            "El sistema normaliza automáticamente mayúsculas, minúsculas, "
            "acentos, espacios y caracteres especiales."
        )

        st.markdown("---")

    return uploaded


# ════════════════════════════════════════════════════
# 8. MAIN
# ════════════════════════════════════════════════════


def main():
    uploaded = render_sidebar()

    st.markdown("""
    <div class="main-header">
        <h1>Dashboard Predictivo Producción — FarmPrecision</h1>
        <p>Producción · Presupuesto · Proyección · Cuadrantes · Modelo RF</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div class="upload-zone">
            <h3>📂 Sube tu archivo Excel o CSV para comenzar</h3>
            <p>Usa el panel lateral para cargar el archivo.</p>
            <p><strong>Formatos soportados:</strong> .xlsx · .xls · .csv</p>
            <p><strong>Columnas principales:</strong> Finca · Lote · Año · Mes ·
            Ton Real · Ton Ppto · Racimos · PPR · Área neta · Días de ciclo · Material · Estado</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 ¿Qué puedes analizar?")

        c1, c2, c3, c4 = st.columns(4)
        c1.info(
            "**Proyección de producción**\n\n"
            "Proyecta la producción futura a partir de la tendencia histórica "
            "y la estacionalidad mensual."
        )
        c2.info(
            "**Productividad y presupuesto**\n\n"
            "Analiza la producción en ton/ha, compara resultados reales frente "
            "al presupuesto y consulta el cumplimiento."
        )
        c3.info(
            "**Análisis de lotes**\n\n"
            "Identifica diferencias de productividad y variabilidad mediante "
            "cuadrantes y análisis por lote, finca, material y zona."
        )
        c4.info(
            "**Modelo predictivo**\n\n"
            "Estima Ton Real o PPR mediante un modelo Random Forest utilizando "
            "edad, área, días de ciclo, estacionalidad y material genético."
        )

        return

    else:
        with st.spinner("⏳ Procesando datos..."):
            file_bytes = uploaded.read()
            file_name = uploaded.name

    try:
        data = cargar_dataset_produccion(file_bytes, file_name=file_name)
        mensual, corte, mes_curso = construir_serie_mensual(data)
    except ValueError as e:
        st.error(f"❌ Error al cargar el archivo:\n\n{e}")
        return
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
        return

    if data.empty:
        st.warning("El archivo no contiene datos válidos.")
        return

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔎 Filtros")

        def multiselect_col(df, col, key):
            opciones = sorted(df[col].dropna().unique().tolist())
            return st.multiselect(f"{col.replace('_', ' ').capitalize()}:", opciones,
                                  default=opciones, key=key)

        gz_sel = multiselect_col(data, "gran_zona", "f_gz")
        bloque_sel = multiselect_col(data, "bloque", "f_bloque")
        mat_sel = multiselect_col(data, "material", "f_mat")
        est_sel = multiselect_col(data, "estado", "f_est")
        anios_all = sorted(pd.to_numeric(data["ano"], errors="coerce").dropna().unique().astype(int).tolist())
        anio_sel = st.multiselect("Año:", anios_all, default=anios_all, key="f_ano")
        lote_q = st.text_input("Buscar Lote:", "", key="f_lote")

        st.markdown("---")
        st.markdown("### ⚙️ Parámetros del modelo")
        horizon = st.slider("Meses a proyectar:", 3, 48, 12, step=3, key="p_horizon")
        recent = st.slider("Meses de historial por el modelo:", 3, 24, 12, key="p_recent")

    df = data.copy()
    df = df[df["gran_zona"].isin(gz_sel)]
    df = df[df["bloque"].isin(bloque_sel)]
    df = df[df["material"].isin(mat_sel)]
    df = df[df["estado"].isin(est_sel)]
    df = df[df["ano"].astype(int).isin(anio_sel)]
    if lote_q.strip():
        df = df[df["lote"].astype(str).str.contains(lote_q.strip(), case=False, na=False)]

    st.sidebar.write(f"Registros filtrados: **{len(df):,}**")

    if df.empty:
        st.warning("Sin datos con los filtros actuales.")
        return

    mensual_f, corte_f, curso_f = construir_serie_mensual(df)

    seccion_kpis(df)
    st.markdown("---")

    tab_names = ["📌 Proyección", "📊 Producción", "🔲 Cuadrantes", "📋 Modelo RF"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_proyeccion(df, mensual_f, corte_f, curso_f, horizon, recent)
    with tabs[1]:
        tab_produccion(df, mensual_f)
    with tabs[2]:
        tab_cuadrantes(df)
    with tabs[3]:
        tab_modelo(df)


if __name__ == "__main__":
    main()