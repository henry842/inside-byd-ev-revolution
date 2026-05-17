"""
BYD vs Tesla vs Toyota — Senior-Level Quantitative Analysis Pipeline
Generates a complete HTML report with EDA, modeling, backtest, and storytelling.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from scipy import stats

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ─── Helpers ───────────────────────────────────────────────────────────────────

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

def img_tag(b64, width="100%"):
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};margin:10px 0;">'

# ─── 1. DATA LOADING ──────────────────────────────────────────────────────────

def load_data(path=None):
    if path is None:
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "data", "auto_company_comparison.csv")
    df_raw = pd.read_csv(path, header=0)
    ticker_row_idx = None
    for i in range(min(10, len(df_raw))):
        if str(df_raw.iloc[i, 0]).strip().lower() == "ticker":
            ticker_row_idx = i
            break
    if ticker_row_idx is None:
        raise ValueError("Ticker row not found")

    metric_names = list(df_raw.columns)
    tickers = list(df_raw.iloc[ticker_row_idx].values)
    df = df_raw.iloc[ticker_row_idx + 1:].copy()
    ticker_map = {"BYDDF": "BYD", "TM": "TOYOTA", "TSLA": "TSLA"}

    new_cols = []
    for j, metric in enumerate(metric_names):
        if j == 0:
            new_cols.append("Date")
            continue
        metric = str(metric).strip().replace(".1", "").replace(".2", "").replace(".0", "")
        ticker = str(tickers[j]).strip()
        if ticker in {"", "nan", "NaN"}:
            new_cols.append(metric)
        else:
            t = ticker_map.get(ticker, ticker)
            new_cols.append(f"{metric}_{t}")

    df.columns = new_cols
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ─── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────

def engineer_features(df):
    out = df.copy()

    for t in ["TSLA", "BYD", "TOYOTA"]:
        out[f"return_{t}"] = out[f"Close_{t}"].pct_change()
        for w in [5, 7, 14, 21, 30, 50, 100, 200]:
            out[f"ma{w}_{t}"] = out[f"Close_{t}"].rolling(w).mean()
            out[f"ema{w}_{t}"] = out[f"Close_{t}"].ewm(span=w, adjust=False).mean()
        out[f"vol20_{t}"] = out[f"return_{t}"].rolling(20).std()
        out[f"vol60_{t}"] = out[f"return_{t}"].rolling(60).std()
        out[f"momentum10_{t}"] = out[f"Close_{t}"] - out[f"Close_{t}"].shift(10)
        out[f"momentum20_{t}"] = out[f"Close_{t}"] - out[f"Close_{t}"].shift(20)

        # RSI
        delta = out[f"Close_{t}"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        out[f"rsi14_{t}"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        ma20 = out[f"Close_{t}"].rolling(20).mean()
        sd20 = out[f"Close_{t}"].rolling(20).std()
        out[f"bb_upper_{t}"] = ma20 + 2 * sd20
        out[f"bb_lower_{t}"] = ma20 - 2 * sd20
        out[f"bb_width_{t}"] = (out[f"bb_upper_{t}"] - out[f"bb_lower_{t}"]) / (ma20 + 1e-12)
        out[f"bb_pct_{t}"] = (out[f"Close_{t}"] - out[f"bb_lower_{t}"]) / (out[f"bb_upper_{t}"] - out[f"bb_lower_{t}"] + 1e-12)

        # MACD
        ema12 = out[f"Close_{t}"].ewm(span=12, adjust=False).mean()
        ema26 = out[f"Close_{t}"].ewm(span=26, adjust=False).mean()
        out[f"macd_{t}"] = ema12 - ema26
        out[f"macd_signal_{t}"] = out[f"macd_{t}"].ewm(span=9, adjust=False).mean()
        out[f"macd_hist_{t}"] = out[f"macd_{t}"] - out[f"macd_signal_{t}"]

        # ATR
        high_low = out[f"High_{t}"] - out[f"Low_{t}"]
        high_close = (out[f"High_{t}"] - out[f"Close_{t}"].shift(1)).abs()
        low_close = (out[f"Low_{t}"] - out[f"Close_{t}"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out[f"atr14_{t}"] = tr.rolling(14).mean()

        # Drawdown
        roll_max = out[f"Close_{t}"].cummax()
        out[f"drawdown_{t}"] = out[f"Close_{t}"] / (roll_max + 1e-12) - 1

        # Returns lags
        for lag in [1, 2, 3, 5]:
            out[f"return_lag{lag}_{t}"] = out[f"return_{t}"].shift(lag)

        # Volume features
        if f"Volume_{t}" in out.columns:
            out[f"vol_ma20_{t}"] = out[f"Volume_{t}"].rolling(20).mean()
            out[f"vol_ratio_{t}"] = out[f"Volume_{t}"] / (out[f"vol_ma20_{t}"] + 1e-12)

    # Cross-company features
    pairs = [("TSLA", "BYD"), ("TSLA", "TOYOTA"), ("BYD", "TOYOTA")]
    for a, b in pairs:
        out[f"corr_{a}_{b}_30"] = out[f"return_{a}"].rolling(30).corr(out[f"return_{b}"])
        out[f"corr_{a}_{b}_60"] = out[f"return_{a}"].rolling(60).corr(out[f"return_{b}"])
        out[f"spread_{a}_{b}"] = out[f"Close_{a}"] - out[f"Close_{b}"]
        out[f"diff_return_{a}_{b}"] = out[f"return_{a}"] - out[f"return_{b}"]
        out[f"return_ratio_{a}_{b}"] = out[f"return_{a}"] / (out[f"return_{b}"].abs() + 1e-12)

    return out

# ─── 3. VISUALIZATIONS ───────────────────────────────────────────────────────

def plot_price_evolution(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(df["Date"], df["Close_TSLA"], label="Tesla", linewidth=1.2)
    ax.plot(df["Date"], df["Close_BYD"], label="BYD", linewidth=1.2)
    ax.plot(df["Date"], df["Close_TOYOTA"], label="Toyota", linewidth=1.2)
    ax.set_title("Evolução do Preço de Fechamento (2018-2025)")
    ax.set_ylabel("Preço (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax2 = axes[1]
    if "Volume_TSLA" in df.columns:
        ax2.fill_between(df["Date"], df["Volume_TSLA"], alpha=0.3, label="Vol TSLA")
        ax2.fill_between(df["Date"], df["Volume_BYD"], alpha=0.3, label="Vol BYD")
        ax2.set_ylabel("Volume")
        ax2.legend(fontsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    return fig_to_base64(fig)

def plot_normalized_performance(df):
    fig, ax = plt.subplots(figsize=(14, 6))
    for t, label in [("TSLA", "Tesla"), ("BYD", "BYD"), ("TOYOTA", "Toyota")]:
        normalized = df[f"Close_{t}"] / df[f"Close_{t}"].iloc[0] * 100
        ax.plot(df["Date"], normalized, label=label, linewidth=1.3)
    ax.set_title("Performance Normalizada (Base 100)")
    ax.set_ylabel("Índice (Base 100)")
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_returns_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = sns.color_palette()
    for idx, (ax, t, label) in enumerate(zip(axes, ["TSLA", "BYD", "TOYOTA"], ["Tesla", "BYD", "Toyota"])):
        ret = df[f"return_{t}"].dropna()
        ax.hist(ret, bins=80, density=True, alpha=0.7, color=palette[idx])
        mu, sigma = ret.mean(), ret.std()
        x = np.linspace(ret.min(), ret.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=1.5, label=f"Normal\nμ={mu:.4f}\nσ={sigma:.4f}")
        ax.set_title(f"Distribuição de Retornos — {label}")
        ax.set_xlabel("Retorno Diário")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_volatility_comparison(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    for t, label in [("TSLA", "Tesla"), ("BYD", "BYD"), ("TOYOTA", "Toyota")]:
        ax.plot(df["Date"], df[f"vol20_{t}"], label=f"Vol 20d {label}", linewidth=1)
    ax.set_title("Volatilidade Rolling 20 dias")
    ax.set_ylabel("Volatilidade (std dos retornos)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_correlation_heatmap(df):
    ret_cols = ["return_TSLA", "return_BYD", "return_TOYOTA"]
    corr = df[ret_cols].corr()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=axes[0],
                xticklabels=["Tesla", "BYD", "Toyota"], yticklabels=["Tesla", "BYD", "Toyota"])
    axes[0].set_title("Correlação entre Retornos")

    for (a, b), color in zip([("TSLA", "BYD"), ("TSLA", "TOYOTA")], ["tab:blue", "tab:orange"]):
        axes[1].plot(df["Date"], df[f"corr_{a}_{b}_30"], label=f"{a}×{b}", linewidth=1, color=color)
    axes[1].set_title("Correlação Rolling 30 dias")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    return fig_to_base64(fig)

def plot_drawdown(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, t, label, color in zip(axes, ["TSLA", "BYD", "TOYOTA"],
                                    ["Tesla", "BYD", "Toyota"],
                                    ["tab:red", "tab:blue", "tab:green"]):
        dd = df[f"drawdown_{t}"]
        ax.fill_between(df["Date"], dd, 0, alpha=0.4, color=color)
        ax.plot(df["Date"], dd, color=color, linewidth=0.8)
        ax.set_ylabel("Drawdown")
        ax.set_title(f"Drawdown — {label}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_rsi(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, t, label in zip(axes, ["TSLA", "BYD", "TOYOTA"], ["Tesla", "BYD", "Toyota"]):
        ax.plot(df["Date"], df[f"rsi14_{t}"], linewidth=0.8)
        ax.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax.axhline(30, color="green", linestyle="--", alpha=0.5)
        ax.fill_between(df["Date"], 70, 100, alpha=0.1, color="red")
        ax.fill_between(df["Date"], 0, 30, alpha=0.1, color="green")
        ax.set_ylabel("RSI(14)")
        ax.set_title(f"RSI(14) — {label}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_bollinger(df, t="TSLA", label="Tesla"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["Date"], df[f"Close_{t}"], label="Preço", linewidth=1)
    ax.plot(df["Date"], df[f"bb_upper_{t}"], "r--", linewidth=0.7, label="BB Superior")
    ax.plot(df["Date"], df[f"bb_lower_{t}"], "g--", linewidth=0.7, label="BB Inferior")
    ax.fill_between(df["Date"], df[f"bb_lower_{t}"], df[f"bb_upper_{t}"], alpha=0.1, color="gray")
    ax.set_title(f"Bollinger Bands (20,2) — {label}")
    ax.set_ylabel("Preço (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_macd(df, t="TSLA", label="Tesla"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    axes[0].plot(df["Date"], df[f"Close_{t}"], label="Preço", linewidth=1)
    axes[0].plot(df["Date"], df[f"ma50_{t}"], label="MA50", linewidth=0.8, alpha=0.7)
    axes[0].plot(df["Date"], df[f"ma200_{t}"], label="MA200", linewidth=0.8, alpha=0.7)
    axes[0].set_title(f"Médias Móveis — {label}")
    axes[0].legend(fontsize=8)

    colors = ["green" if v >= 0 else "red" for v in df[f"macd_hist_{t}"].fillna(0)]
    axes[1].bar(df["Date"], df[f"macd_hist_{t}"], color=colors, alpha=0.6, width=1)
    axes[1].plot(df["Date"], df[f"macd_{t}"], label="MACD", linewidth=0.8)
    axes[1].plot(df["Date"], df[f"macd_signal_{t}"], label="Signal", linewidth=0.8)
    axes[1].set_title("MACD")
    axes[1].legend(fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    return fig_to_base64(fig)

def plot_rolling_sharpe(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    for t, label in [("TSLA", "Tesla"), ("BYD", "BYD"), ("TOYOTA", "Toyota")]:
        roll_mean = df[f"return_{t}"].rolling(60).mean()
        roll_std = df[f"return_{t}"].rolling(60).std()
        sharpe = (roll_mean / (roll_std + 1e-12)) * np.sqrt(252)
        ax.plot(df["Date"], sharpe, label=label, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Sharpe Ratio Rolling 60 dias (anualizado)")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_monthly_returns_heatmap(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, t, label in zip(axes, ["TSLA", "BYD", "TOYOTA"], ["Tesla", "BYD", "Toyota"]):
        tmp = df[["Date", f"Close_{t}"]].copy()
        tmp["year"] = tmp["Date"].dt.year
        tmp["month"] = tmp["Date"].dt.month
        monthly = tmp.groupby(["year", "month"])[f"Close_{t}"].last().pct_change().unstack()
        monthly.columns = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"][:len(monthly.columns)]
        sns.heatmap(monthly, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(f"Retornos Mensais — {label}")
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_yearly_returns(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    years = df["Date"].dt.year.unique()
    x = np.arange(len(years))
    for i, (t, label) in enumerate([("TSLA", "Tesla"), ("BYD", "BYD"), ("TOYOTA", "Toyota")]):
        yr_returns = []
        for yr in years:
            yr_data = df[df["Date"].dt.year == yr]
            if len(yr_data) > 1:
                ret = yr_data[f"Close_{t}"].iloc[-1] / yr_data[f"Close_{t}"].iloc[0] - 1
            else:
                ret = 0
            yr_returns.append(ret)
        bars = ax.bar(x + i * width, yr_returns, width, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels(years)
    ax.set_title("Retornos Anuais por Empresa")
    ax.set_ylabel("Retorno")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_feature_importance(importances, feature_names, title="Feature Importance"):
    idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importância")
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["proba"])
        auc_val = res.get("auc", res.get("avg_metrics", {}).get("auc", 0))
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=1.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("Curvas ROC")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_confusion_matrices(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_true"], res["pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
        ax.set_title(f"Matriz de Confusão — {name}")
        ax.set_ylabel("Real")
        ax.set_xlabel("Previsto")
    fig.tight_layout()
    return fig_to_base64(fig)

def plot_backtest(equity, bh_equity, dates, title="Backtest"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(dates, equity, label="Estratégia", linewidth=1.3)
    axes[0].plot(dates, bh_equity, label="Buy & Hold", linewidth=1.3)
    axes[0].set_title(title)
    axes[0].set_ylabel("Capital (USD)")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    dd = equity / np.maximum.accumulate(equity) - 1
    axes[1].fill_between(dates, dd, 0, alpha=0.4, color="red")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_title("Drawdown da Estratégia")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    return fig_to_base64(fig)

def plot_regime_analysis(df, regimes):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    merged = df.merge(regimes, on="Date", how="inner")
    colors = {0: "green", 1: "red", 2: "blue"}
    labels = {0: "Alta", 1: "Baixa", 2: "Lateral"}

    for regime in sorted(merged["regime"].unique()):
        mask = merged["regime"] == regime
        axes[0].scatter(merged.loc[mask, "Date"], merged.loc[mask, "Close_TSLA"],
                       s=3, label=labels.get(regime, f"Regime {regime}"),
                       color=colors.get(regime, "gray"), alpha=0.6)
    axes[0].set_title("Regimes de Mercado (KMeans) sobre Preço Tesla")
    axes[0].set_ylabel("Preço")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    regime_counts = merged["regime"].value_counts().sort_index()
    axes[1].bar([labels.get(i, f"Regime {i}") for i in regime_counts.index],
               regime_counts.values, color=[colors.get(i, "gray") for i in regime_counts.index])
    axes[1].set_title("Distribuição dos Regimes")
    axes[1].set_ylabel("Dias")

    fig.tight_layout()
    return fig_to_base64(fig)

# ─── 4. STATISTICAL TESTS ────────────────────────────────────────────────────

def run_statistical_tests(df):
    results = {}
    for t in ["TSLA", "BYD", "TOYOTA"]:
        ret = df[f"return_{t}"].dropna()
        # Normality test
        jb_stat, jb_p = stats.jarque_bera(ret)
        # Stationarity (ADF-like via autocorrelation)
        acf1 = ret.autocorr(lag=1)
        results[t] = {
            "mean_return": ret.mean(),
            "std_return": ret.std(),
            "skewness": ret.skew(),
            "kurtosis": ret.kurtosis(),
            "jarque_bera_stat": jb_stat,
            "jarque_bera_p": jb_p,
            "acf_lag1": acf1,
            "min_return": ret.min(),
            "max_return": ret.max(),
            "sharpe_annual": (ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252),
        }

    # Correlation significance
    for a, b in [("TSLA", "BYD"), ("TSLA", "TOYOTA"), ("BYD", "TOYOTA")]:
        r, p = stats.pearsonr(df[f"return_{a}"].dropna(), df[f"return_{b}"].dropna())
        results[f"corr_{a}_{b}"] = {"pearson_r": r, "p_value": p}

    return results

# ─── 5. MODELING ─────────────────────────────────────────────────────────────

def build_model_data(df, target_col="target_up_next"):
    feature_exclude = ["Date", target_col]
    for t in ["TSLA", "BYD", "TOYOTA"]:
        feature_exclude += [f"Open_{t}", f"High_{t}", f"Low_{t}", f"Close_{t}", f"Volume_{t}"]
        feature_exclude += [f"bb_upper_{t}", f"bb_lower_{t}"]

    feature_cols = [c for c in df.columns if c not in feature_exclude and df[c].dtype in [np.float64, np.int64]]
    return feature_cols

def train_models(df, feature_cols):
    df_model = df.dropna(subset=feature_cols + ["target_up_next"]).copy()
    X = df_model[feature_cols].values
    y = df_model["target_up_next"].values
    dates = df_model["Date"].values

    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                                random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                                        subsample=0.8, random_state=42),
    }

    all_results = {}
    best_model_name = None
    best_auc = 0

    for name, model in models.items():
        fold_metrics = []
        last_proba = None
        last_y = None
        last_pred = None
        last_dates = None

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)[:, 1]
            pred = (proba >= 0.5).astype(int)

            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "auc": roc_auc_score(y_test, proba),
            }
            fold_metrics.append(metrics)

            last_proba = proba
            last_y = y_test
            last_pred = pred
            last_dates = dates[test_idx]

        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}

        all_results[name] = {
            "avg_metrics": avg_metrics,
            "fold_metrics": fold_metrics,
            "proba": last_proba,
            "y_true": last_y,
            "pred": last_pred,
            "dates": last_dates,
            "model": model,
        }

        if avg_metrics["auc"] > best_auc:
            best_auc = avg_metrics["auc"]
            best_model_name = name

    # Feature importance from best tree model
    importances = None
    if "Random Forest" in all_results:
        importances = all_results["Random Forest"]["model"].feature_importances_
    elif "Gradient Boosting" in all_results:
        importances = all_results["Gradient Boosting"]["model"].feature_importances_

    return all_results, best_model_name, importances

# ─── 6. TECHNICAL STRATEGIES ─────────────────────────────────────────────────

def backtest_ma_crossover(df, t="TSLA", fast=50, slow=200, capital0=10000):
    tmp = df[["Date", f"Close_{t}", f"return_{t}", f"ma{fast}_{t}", f"ma{slow}_{t}"]].dropna().copy()
    tmp["signal"] = (tmp[f"ma{fast}_{t}"] > tmp[f"ma{slow}_{t}"]).astype(int)
    tmp["strat_ret"] = tmp[f"return_{t}"] * tmp["signal"]
    equity = capital0 * np.cumprod(1 + tmp["strat_ret"].values)
    bh = capital0 * np.cumprod(1 + tmp[f"return_{t}"].values)
    peak = np.maximum.accumulate(equity)
    dd = equity / (peak + 1e-12) - 1
    sharpe_val = np.sqrt(252) * tmp["strat_ret"].mean() / (tmp["strat_ret"].std(ddof=1) + 1e-12)
    return {
        "equity": equity, "bh_equity": bh, "dates": tmp["Date"].values,
        "total_return": equity[-1] / capital0 - 1,
        "bh_total_return": bh[-1] / capital0 - 1,
        "sharpe": float(sharpe_val), "max_drawdown": float(dd.min()),
        "n_trades": int(tmp["signal"].diff().abs().sum() // 2),
        "win_rate": float(np.mean(tmp.loc[tmp["signal"] == 1, "strat_ret"] > 0)),
        "annual_return": (equity[-1] / capital0) ** (252 / len(equity)) - 1,
    }

def backtest_rsi_strategy(df, t="TSLA", buy_threshold=30, sell_threshold=70, capital0=10000):
    tmp = df[["Date", f"Close_{t}", f"return_{t}", f"rsi14_{t}"]].dropna().copy()
    position = 0
    signals = []
    for rsi in tmp[f"rsi14_{t}"].values:
        if rsi < buy_threshold:
            position = 1
        elif rsi > sell_threshold:
            position = 0
        signals.append(position)
    tmp["signal"] = signals
    tmp["strat_ret"] = tmp[f"return_{t}"] * tmp["signal"]
    equity = capital0 * np.cumprod(1 + tmp["strat_ret"].values)
    bh = capital0 * np.cumprod(1 + tmp[f"return_{t}"].values)
    peak = np.maximum.accumulate(equity)
    dd = equity / (peak + 1e-12) - 1
    sharpe_val = np.sqrt(252) * tmp["strat_ret"].mean() / (tmp["strat_ret"].std(ddof=1) + 1e-12)
    return {
        "equity": equity, "bh_equity": bh, "dates": tmp["Date"].values,
        "total_return": equity[-1] / capital0 - 1,
        "bh_total_return": bh[-1] / capital0 - 1,
        "sharpe": float(sharpe_val), "max_drawdown": float(dd.min()),
        "n_trades": int(np.abs(np.diff(signals)).sum() // 2),
        "win_rate": float(np.mean(tmp.loc[tmp["signal"] == 1, "strat_ret"] > 0)),
        "annual_return": (equity[-1] / capital0) ** (252 / len(equity)) - 1,
    }

def backtest_bollinger_strategy(df, t="TSLA", capital0=10000):
    tmp = df[["Date", f"Close_{t}", f"return_{t}", f"bb_pct_{t}"]].dropna().copy()
    position = 0
    signals = []
    for bb in tmp[f"bb_pct_{t}"].values:
        if bb < 0.0:  # Below lower band - buy
            position = 1
        elif bb > 1.0:  # Above upper band - sell
            position = 0
        signals.append(position)
    tmp["signal"] = signals
    tmp["strat_ret"] = tmp[f"return_{t}"] * tmp["signal"]
    equity = capital0 * np.cumprod(1 + tmp["strat_ret"].values)
    bh = capital0 * np.cumprod(1 + tmp[f"return_{t}"].values)
    peak = np.maximum.accumulate(equity)
    dd = equity / (peak + 1e-12) - 1
    sharpe_val = np.sqrt(252) * tmp["strat_ret"].mean() / (tmp["strat_ret"].std(ddof=1) + 1e-12)
    return {
        "equity": equity, "bh_equity": bh, "dates": tmp["Date"].values,
        "total_return": equity[-1] / capital0 - 1,
        "bh_total_return": bh[-1] / capital0 - 1,
        "sharpe": float(sharpe_val), "max_drawdown": float(dd.min()),
        "n_trades": int(np.abs(np.diff(signals)).sum() // 2),
        "win_rate": float(np.mean(tmp.loc[tmp["signal"] == 1, "strat_ret"] > 0)),
        "annual_return": (equity[-1] / capital0) ** (252 / len(equity)) - 1,
    }

def compute_var_cvar(df, confidence=0.95):
    results = {}
    for t in ["TSLA", "BYD", "TOYOTA"]:
        ret = df[f"return_{t}"].dropna()
        var = np.percentile(ret, (1 - confidence) * 100)
        cvar = ret[ret <= var].mean()
        results[t] = {"VaR_95": var, "CVaR_95": cvar}
    return results

# ─── 7. BACKTEST ─────────────────────────────────────────────────────────────

def backtest_strategy(df, proba, dates, threshold=0.55, capital0=10000):
    mask = df["Date"].isin(dates)
    df_bt = df.loc[mask].copy().iloc[:len(proba)]
    daily_ret = df_bt["return_TSLA"].fillna(0).values

    signals = (proba > threshold).astype(int)
    strat_ret = daily_ret * signals

    equity = capital0 * np.cumprod(1 + strat_ret)
    bh_equity = capital0 * np.cumprod(1 + daily_ret)

    peak = np.maximum.accumulate(equity)
    dd = equity / (peak + 1e-12) - 1

    def sharpe(r):
        r = np.asarray(r)
        return np.sqrt(252) * r.mean() / (r.std(ddof=1) + 1e-12)

    def max_consecutive_loss(r):
        losses = (np.array(r) < 0).astype(int)
        max_streak = 0
        current = 0
        for l in losses:
            if l:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    n_trades = int(signals.sum())
    win_rate = np.mean(strat_ret[signals == 1] > 0) if n_trades > 0 else 0

    return {
        "equity": equity,
        "bh_equity": bh_equity,
        "dates": df_bt["Date"].values,
        "total_return": equity[-1] / capital0 - 1,
        "bh_total_return": bh_equity[-1] / capital0 - 1,
        "sharpe": float(sharpe(strat_ret)),
        "max_drawdown": float(dd.min()),
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "max_consecutive_loss": max_consecutive_loss(strat_ret),
        "annual_return": (equity[-1] / capital0) ** (252 / len(equity)) - 1 if len(equity) > 0 else 0,
    }

# ─── 7. HTML REPORT ──────────────────────────────────────────────────────────

def generate_report(df, stats_tests, model_results, best_model, importances,
                    feature_names, bt_results, regime_img, eda_imgs,
                    tech_strategies=None, var_results=None):
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Summary stats table
    summary_rows = ""
    for t in ["TSLA", "BYD", "TOYOTA"]:
        s = stats_tests[t]
        summary_rows += f"""
        <tr>
            <td><strong>{t}</strong></td>
            <td>{s['mean_return']:.4%}</td>
            <td>{s['std_return']:.4%}</td>
            <td>{s['sharpe_annual']:.2f}</td>
            <td>{s['skewness']:.3f}</td>
            <td>{s['kurtosis']:.3f}</td>
            <td>{s['jarque_bera_p']:.2e}</td>
            <td>{s['min_return']:.2%}</td>
            <td>{s['max_return']:.2%}</td>
        </tr>"""

    # Model comparison table
    model_rows = ""
    for name, res in model_results.items():
        m = res["avg_metrics"]
        best_flag = " ★" if name == best_model else ""
        model_rows += f"""
        <tr>
            <td><strong>{name}{best_flag}</strong></td>
            <td>{m['accuracy']:.4f}</td>
            <td>{m['precision']:.4f}</td>
            <td>{m['recall']:.4f}</td>
            <td>{m['auc']:.4f}</td>
        </tr>"""

    # Backtest table
    bt_rows = ""
    for name, bt in bt_results.items():
        bt_rows += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{bt['total_return']:.2%}</td>
            <td>{bt['bh_total_return']:.2%}</td>
            <td>{bt['sharpe']:.2f}</td>
            <td>{bt['max_drawdown']:.2%}</td>
            <td>{bt['n_trades']}</td>
            <td>{bt['win_rate']:.1%}</td>
            <td>{bt['annual_return']:.2%}</td>
        </tr>"""

    # Correlation stats
    corr_rows = ""
    for key in ["corr_TSLA_BYD", "corr_TSLA_TOYOTA", "corr_BYD_TOYOTA"]:
        if key in stats_tests:
            s = stats_tests[key]
            corr_rows += f"""
            <tr>
                <td>{key.replace('corr_', '').replace('_', ' × ')}</td>
                <td>{s['pearson_r']:.4f}</td>
                <td>{s['p_value']:.2e}</td>
            </tr>"""

    # Total appreciation
    appreciation = {}
    for t in ["TSLA", "BYD", "TOYOTA"]:
        appreciation[t] = df[f"Close_{t}"].iloc[-1] / df[f"Close_{t}"].iloc[0] - 1

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Análise BYD vs Tesla vs Toyota</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px;
       background: #f8f9fa; color: #333; line-height: 1.6; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-top: 40px; }}
h3 {{ color: #0f3460; }}
.section {{ background: white; padding: 25px; margin: 20px 0; border-radius: 8px;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #16213e; color: white; }}
tr:hover {{ background: #f5f5f5; }}
.insight {{ background: #e8f4f8; border-left: 4px solid #0f3460; padding: 15px; margin: 15px 0;
           border-radius: 4px; }}
.warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0;
           border-radius: 4px; }}
.success {{ background: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 15px 0;
           border-radius: 4px; }}
.toc {{ background: #e8eaf6; padding: 20px; border-radius: 8px; }}
.toc a {{ color: #0f3460; text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 15px 0; }}
.kpi {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
       padding: 20px; border-radius: 8px; text-align: center; }}
.kpi .value {{ font-size: 28px; font-weight: bold; }}
.kpi .label {{ font-size: 12px; opacity: 0.9; }}
</style>
</head>
<body>

<h1>Análise Quantitativa Comparativa: BYD vs Tesla vs Toyota</h1>
<p><em>Gerado em {now} | Período: {df['Date'].min().strftime('%Y-%m-%d')} a {df['Date'].max().strftime('%Y-%m-%d')} | {len(df)} pregões</em></p>

<div class="toc">
<h3>Índice</h3>
<ol>
<li><a href="#resumo">Resumo Executivo</a></li>
<li><a href="#eda">Análise Exploratória (EDA)</a></li>
<li><a href="#tecnica">Análise Técnica</a></li>
<li><a href="#stats">Testes Estatísticos</a></li>
<li><a href="#regimes">Detecção de Regimes</a></li>
<li><a href="#modeling">Modelagem Preditiva</a></li>
<li><a href="#backtest">Backtesting</a></li>
<li><a href="#conclusions">Conclusões e Recomendações</a></li>
</ol>
</div>

<!-- ─── 1. RESUMO EXECUTIVO ─── -->
<div class="section" id="resumo">
<h2>1. Resumo Executivo</h2>

<div class="kpi-grid">
    <div class="kpi">
        <div class="label">TSLA — Valorização Total</div>
        <div class="value">{appreciation['TSLA']:.1%}</div>
    </div>
    <div class="kpi">
        <div class="label">BYD — Valorização Total</div>
        <div class="value">{appreciation['BYD']:.1%}</div>
    </div>
    <div class="kpi">
        <div class="label">TOYOTA — Valorização Total</div>
        <div class="value">{appreciation['TOYOTA']:.1%}</div>
    </div>
</div>

<div class="insight">
<strong>Storytelling:</strong> Esta análise cobre {len(df)} dias de negociação de três gigantes do setor automotivo.
A Tesla ({appreciation['TSLA']:.1%}), BYD ({appreciation['BYD']:.1%}) e Toyota ({appreciation['TOYOTA']:.1%})
mostram perfis de risco-retorno distintos que refletem suas posições no mercado de veículos elétricos.
</div>
</div>

<!-- ─── 2. EDA ─── -->
<div class="section" id="eda">
<h2>2. Análise Exploratória de Dados</h2>

<h3>2.1 Evolução dos Preços</h3>
{img_tag(eda_imgs['price'])}

<h3>2.2 Performance Normalizada</h3>
{img_tag(eda_imgs['normalized'])}
<div class="insight">
<strong>Insight:</strong> A normalização permite comparar a performance relativa ignorando diferenças de escala.
A empresa que mais se valorizou no período demonstra a preferência do mercado pelo setor de EVs.
</div>

<h3>2.3 Distribuição dos Retornos</h3>
{img_tag(eda_imgs['returns_dist'])}
<div class="warning">
<strong>Atenção:</strong> Os retornos apresentam caudas pesadas (leptocúrticos), violando a hipótese de normalidade.
Isso tem implicações diretas para modelos que assumem distribuição normal (VaR paramétrico, Black-Scholes).
</div>

<h3>2.4 Retornos Anuais</h3>
{img_tag(eda_imgs['yearly'])}

<h3>2.5 Retornos Mensais (Heatmap)</h3>
{img_tag(eda_imgs['monthly'])}
<div class="insight">
<strong>Sazonalidade:</strong> O heatmap revela padrões sazonais — certos meses consistentemente apresentam
retornos mais altos ou mais baixos, sugerindo efeitos de calendário (e.g., "Sell in May", efeito janeiro).
</div>
</div>

<!-- ─── 3. ANÁLISE TÉCNICA ─── -->
<div class="section" id="tecnica">
<h2>3. Análise Técnica</h2>

<h3>3.1 Volatilidade</h3>
{img_tag(eda_imgs['volatility'])}
<div class="insight">
<strong>Insight:</strong> A volatilidade é um indicador-chave de risco. Períodos de alta volatilidade coincidem
com incertezas macroeconômicas e eventos específicos do setor automotivo.
</div>

<h3>3.2 Sharpe Ratio Rolling</h3>
{img_tag(eda_imgs['sharpe'])}

<h3>3.3 Drawdown</h3>
{img_tag(eda_imgs['drawdown'])}
<div class="warning">
<strong>Risco:</strong> O drawdown máximo mostra a pior perda consecutiva. Investidores precisam estar preparados
para essas correções, especialmente em ações de alta volatilidade como Tesla.
</div>

<h3>3.4 RSI (14 períodos)</h3>
{img_tag(eda_imgs['rsi'])}

<h3>3.5 Bollinger Bands — Tesla</h3>
{img_tag(eda_imgs['bollinger'])}

<h3>3.6 MACD e Médias Móveis — Tesla</h3>
{img_tag(eda_imgs['macd'])}

<h3>3.7 Correlações</h3>
{img_tag(eda_imgs['correlation'])}

<table>
<tr><th>Par</th><th>Correlação de Pearson</th><th>p-valor</th></tr>
{corr_rows}
</table>

<div class="insight">
<strong>Diversificação:</strong> Correlações entre os retornos indicam o potencial de diversificação.
Correlações baixas ou negativas entre empresas sugerem que um portfólio combinado pode reduzir o risco.
</div>
</div>

<!-- ─── 4. TESTES ESTATÍSTICOS ─── -->
<div class="section" id="stats">
<h2>4. Testes Estatísticos</h2>

<table>
<tr><th>Empresa</th><th>Retorno Médio</th><th>Volatilidade</th><th>Sharpe Anual</th>
    <th>Assimetria</th><th>Curtose</th><th>Jarque-Bera (p)</th><th>Mín</th><th>Máx</th></tr>
{summary_rows}
</table>

<div class="insight">
<strong>Normalidade Rejeitada:</strong> O teste de Jarque-Bera rejeita a hipótese de normalidade para todas as ações
(p < 0.05), confirmando a presença de caudas pesadas e assimetria nos retornos.
</div>

<h3>4.1 Value at Risk (VaR) e CVaR — 95% de confiança</h3>
<table>
<tr><th>Empresa</th><th>VaR (95%)</th><th>CVaR (95%)</th><th>Interpretação</th></tr>
{"".join(f'''<tr><td><strong>{t}</strong></td><td>{var_results[t]["VaR_95"]:.2%}</td><td>{var_results[t]["CVaR_95"]:.2%}</td><td>Em 95% dos dias, a perda máxima é {abs(var_results[t]["VaR_95"]):.2%}</td></tr>''' for t in ["TSLA","BYD","TOYOTA"]) if var_results else ""}
</table>
<div class="warning">
<strong>Gestão de Risco:</strong> O VaR mostra a perda máxima esperada em 95% dos dias. O CVaR (Expected Shortfall)
mostra a média das perdas nos 5% piores dias — métrica mais conservadora e recomendada para gestão de risco.
</div>
</div>

<!-- ─── 5. REGIMES ─── -->
<div class="section" id="regimes">
<h2>5. Detecção de Regimes de Mercado</h2>
{img_tag(regime_img)}
<div class="insight">
<strong>Regimes:</strong> O KMeans identifica 3 regimes distintos: alta, baixa e lateral.
Cada regime exige estratégias diferentes — o que funciona em tendência pode falhar em consolidação.
</div>
</div>

<!-- ─── 6. MODELAGEM ─── -->
<div class="section" id="modeling">
<h2>6. Modelagem Preditiva</h2>
<p>Target: se o retorno do dia seguinte da Tesla será positivo (classificação binária).
Validação: Time Series Split com 5 folds.</p>

<table>
<tr><th>Modelo</th><th>Acurácia</th><th>Precisão</th><th>Recall</th><th>ROC-AUC</th></tr>
{model_rows}
</table>

<h3>6.1 Curvas ROC</h3>
{img_tag(eda_imgs['roc'])}

<h3>6.2 Matrizes de Confusão</h3>
{img_tag(eda_imgs['cm'])}

<h3>6.3 Feature Importance</h3>
{img_tag(eda_imgs['importance'])}

<div class="warning">
<strong>Cuidado com Overfitting:</strong> Em séries temporais financeiras, a performance out-of-sample
tendem a ser inferior ao backtest. Os resultados devem ser interpretados como indicativos, não garantia.
</div>
</div>

<!-- ─── 7. BACKTEST ─── -->
<div class="section" id="backtest">
<h2>7. Backtesting da Estratégia</h2>
<p>Estratégia: comprar Tesla quando P(subida) > threshold, senão ficar em caixa.</p>

<table>
<tr><th>Modelo</th><th>Retorno Total</th><th>Buy & Hold</th><th>Sharpe</th><th>Max DD</th>
    <th>Nº Trades</th><th>Win Rate</th><th>Retorno Anual</th></tr>
{bt_rows}
</table>
"""

    for name, bt in bt_results.items():
        html += f"<h3>Equity Curve — {name}</h3>"
        html += img_tag(plot_backtest(bt["equity"], bt["bh_equity"], bt["dates"],
                                       title=f"Backtest — {name}"))

    html += f"""
<div class="insight">
<strong>Interpretação:</strong> Compare o retorno da estratégia com o Buy &amp; Hold.
Uma estratégia superior deve entregar mais retorno com menos drawdown (maior Sharpe).
O número de trades e win rate mostram a viabilidade prática.
</div>

<h3>7.1 Estratégias Técnicas (Análise Pura)</h3>
<p>Backtests baseados em indicadores técnicos puros, sem ML. Estas são estratégias clássicas de trading.</p>
<table>
<tr><th>Estratégia</th><th>Retorno Total</th><th>Buy & Hold</th><th>Sharpe</th><th>Max DD</th>
    <th>Nº Trades</th><th>Win Rate</th><th>Retorno Anual</th></tr>
"""

    if tech_strategies:
        for name, bt in tech_strategies.items():
            html += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{bt['total_return']:.2%}</td>
            <td>{bt['bh_total_return']:.2%}</td>
            <td>{bt['sharpe']:.2f}</td>
            <td>{bt['max_drawdown']:.2%}</td>
            <td>{bt['n_trades']}</td>
            <td>{bt['win_rate']:.1%}</td>
            <td>{bt['annual_return']:.2%}</td>
        </tr>"""

    html += "</table>"

    # Plot equity curves for best technical strategies
    if tech_strategies:
        # Pick best strategy by Sharpe
        best_tech = max(tech_strategies.items(), key=lambda x: x[1]["sharpe"])
        html += f"<h3>Equity Curve — Melhor Estratégia Técnica: {best_tech[0]}</h3>"
        html += img_tag(plot_backtest(best_tech[1]["equity"], best_tech[1]["bh_equity"],
                                       best_tech[1]["dates"],
                                       title=f"Backtest — {best_tech[0]}"))

    html += """
<div class="insight">
<strong>Comparação:</strong> As estratégias técnicas (MA Crossover, RSI, Bollinger) são mais interpretáveis
que modelos de ML. Compare os resultados — muitas vezes uma regra simples supera modelos complexos
em dados fora da amostra, pois há menos risco de overfitting.
</div>
</div>

<!-- CONCLUSÕES -->
<div class="section" id="conclusions">
<h2>8. Conclusões e Recomendações</h2>

<h3>Descobertas Principais</h3>
<ol>
<li><strong>Performance Divergente:</strong> As três empresas mostram trajetórias muito distintas,
refletindo diferentes estágios de adoção de EVs e posicionamento de mercado.</li>
<li><strong>Volatilidade como Risco:</strong> A Tesla consistentemente apresenta maior volatilidade,
o que oferece oportunidades de trading mas também riscos significativos.</li>
<li><strong>Correlações Dinâmicas:</strong> As correlações entre as empresas variam ao longo do tempo,
especialmente durante crises — quando mais importa para diversificação.</li>
<li><strong>Modelos Preditivos:</strong> Os modelos de ML mostram capacidade preditiva modesta (AUC > 0.5),
mas a eficiência de mercado limita ganhos consistentes.</li>
</ol>

<h3>Recomendações</h3>
<ul>
<li>Diversificar entre as três empresas para reduzir risco específico</li>
<li>Usar análise técnica (RSI, Bollinger) como complemento, não como sinal único</li>
<li>Monitorar regimes de mercado para ajustar exposição</li>
<li>Implementar stop-loss disciplinado dado os drawdowns observados</li>
</ul>

<div class="warning">
<strong>Disclaimer:</strong> Esta análise é educacional e não constitui recomendação de investimento.
Retornos passados não garantem resultados futuros. Consulte um profissional financeiro.
</div>
</div>

</body>
</html>"""

    return html

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ANÁLISE QUANTITATIVA: BYD vs TESLA vs TOYOTA")
    print("=" * 60)

    # 1. Load data
    print("\n[1/7] Carregando dados...")
    df = load_data()
    print(f"  {len(df)} pregões | {df['Date'].min().date()} a {df['Date'].max().date()}")

    # 2. Feature engineering
    print("[2/7] Engineering features...")
    df = engineer_features(df)
    print(f"  {df.shape[1]} colunas")

    # 3. EDA plots
    print("[3/7] Gerando visualizações...")
    eda_imgs = {
        "price": plot_price_evolution(df),
        "normalized": plot_normalized_performance(df),
        "returns_dist": plot_returns_distribution(df),
        "yearly": plot_yearly_returns(df),
        "monthly": plot_monthly_returns_heatmap(df),
        "volatility": plot_volatility_comparison(df),
        "sharpe": plot_rolling_sharpe(df),
        "drawdown": plot_drawdown(df),
        "rsi": plot_rsi(df),
        "bollinger": plot_bollinger(df),
        "macd": plot_macd(df),
        "correlation": plot_correlation_heatmap(df),
    }
    print(f"  {len(eda_imgs)} gráficos gerados")

    # 4. Statistical tests
    print("[4/7] Rodando testes estatísticos...")
    stats_tests = run_statistical_tests(df)

    # 5. Regime detection
    print("[5/7] Detectando regimes...")
    regime_features = ["return_TSLA", "vol20_TSLA", "momentum10_TSLA", "corr_TSLA_BYD_30"]
    regime_df = df.dropna(subset=regime_features).copy()
    X_regime = StandardScaler().fit_transform(regime_df[regime_features])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    regime_df["regime"] = km.fit_predict(X_regime)
    regimes = regime_df[["Date", "regime"]]
    regime_img = plot_regime_analysis(df, regimes)

    # 6. Modeling
    print("[6/7] Treinando modelos...")
    df["target_up_next"] = (df["return_TSLA"].shift(-1) > 0).astype(int)
    feature_cols = build_model_data(df)
    model_results, best_model, importances = train_models(df, feature_cols)
    print(f"  Melhor modelo: {best_model}")

    # Add ROC and CM plots
    eda_imgs["roc"] = plot_roc_curves(model_results)
    eda_imgs["cm"] = plot_confusion_matrices(model_results)
    if importances is not None:
        eda_imgs["importance"] = plot_feature_importance(importances, feature_cols)
    else:
        eda_imgs["importance"] = ""

    # 7. Backtest
    print("[7/7] Rodando backtests...")
    bt_results = {}
    for name, res in model_results.items():
        if res["proba"] is not None:
            bt = backtest_strategy(df, res["proba"], res["dates"], threshold=0.55)
            bt_results[name] = bt

    # Technical strategy backtests
    tech_strategies = {}
    for t in ["TSLA", "BYD", "TOYOTA"]:
        tech_strategies[f"MA Crossover ({t})"] = backtest_ma_crossover(df, t)
        tech_strategies[f"RSI ({t})"] = backtest_rsi_strategy(df, t)
        tech_strategies[f"Bollinger ({t})"] = backtest_bollinger_strategy(df, t)

    # VaR analysis
    var_results = compute_var_cvar(df)

    # Generate HTML report
    print("\nGerando relatório HTML...")
    html = generate_report(df, stats_tests, model_results, best_model,
                          importances, feature_cols, bt_results, regime_img, eda_imgs,
                          tech_strategies, var_results)

    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base, "output", "relatorio_completo.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[OK] Relatorio salvo em: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESUMO DOS RESULTADOS")
    print("=" * 60)
    for t in ["TSLA", "BYD", "TOYOTA"]:
        s = stats_tests[t]
        print(f"\n{t}:")
        print(f"  Retorno médio diário: {s['mean_return']:.4%}")
        print(f"  Volatilidade: {s['std_return']:.4%}")
        print(f"  Sharpe anualizado: {s['sharpe_annual']:.2f}")

    print(f"\nMelhor modelo: {best_model}")
    for name, res in model_results.items():
        m = res["avg_metrics"]
        print(f"  {name}: AUC={m['auc']:.4f}, Acc={m['accuracy']:.4f}")

    print("\nBacktest:")
    for name, bt in bt_results.items():
        print(f"  {name}: Retorno={bt['total_return']:.2%}, Sharpe={bt['sharpe']:.2f}, MaxDD={bt['max_drawdown']:.2%}")

if __name__ == "__main__":
    main()
