"""Generate standalone charts for README.md"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from full_analysis import load_data, engineer_features

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}")

print("Generating README charts...")
df = load_data()
df = engineer_features(df)

# 1. Price evolution
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["Date"], df["Close_TSLA"], label="Tesla (TSLA)", linewidth=1.3, color="#e74c3c")
ax.plot(df["Date"], df["Close_BYD"], label="BYD (BYDDF)", linewidth=1.3, color="#3498db")
ax.plot(df["Date"], df["Close_TOYOTA"], label="Toyota (TM)", linewidth=1.3, color="#2ecc71")
ax.set_title("Evolucao do Preco de Fechamento (2018-2026)", fontsize=14, fontweight="bold")
ax.set_ylabel("Preco (USD)")
ax.legend(fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout()
save(fig, "price_evolution.png")

# 2. Normalized performance
fig, ax = plt.subplots(figsize=(14, 6))
for t, label, color in [("TSLA", "Tesla", "#e74c3c"), ("BYD", "BYD", "#3498db"), ("TOYOTA", "Toyota", "#2ecc71")]:
    norm = df[f"Close_{t}"] / df[f"Close_{t}"].iloc[0] * 100
    ax.plot(df["Date"], norm, label=label, linewidth=1.5, color=color)
ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Performance Normalizada (Base 100)", fontsize=14, fontweight="bold")
ax.set_ylabel("Indice (Base 100)")
ax.legend(fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout()
save(fig, "normalized_performance.png")

# 3. Returns distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ["#e74c3c", "#3498db", "#2ecc71"]
for ax, t, label, color in zip(axes, ["TSLA", "BYD", "TOYOTA"], ["Tesla", "BYD", "Toyota"], colors):
    ret = df[f"return_{t}"].dropna()
    ax.hist(ret, bins=80, density=True, alpha=0.7, color=color)
    mu, sigma = ret.mean(), ret.std()
    x = np.linspace(ret.min(), ret.max(), 200)
    from scipy import stats
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=1.5)
    ax.set_title(f"Retornos - {label}", fontsize=12)
    ax.set_xlabel("Retorno Diario")
fig.tight_layout()
save(fig, "returns_distribution.png")

# 4. Volatility
fig, ax = plt.subplots(figsize=(14, 5))
for t, label, color in [("TSLA", "Tesla", "#e74c3c"), ("BYD", "BYD", "#3498db"), ("TOYOTA", "Toyota", "#2ecc71")]:
    ax.plot(df["Date"], df[f"vol20_{t}"], label=label, linewidth=1, color=color)
ax.set_title("Volatilidade Rolling 20 dias", fontsize=14, fontweight="bold")
ax.set_ylabel("Volatilidade")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout()
save(fig, "volatility.png")

# 5. Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[["return_TSLA", "return_BYD", "return_TOYOTA"]].corr()
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax,
            xticklabels=["Tesla", "BYD", "Toyota"], yticklabels=["Tesla", "BYD", "Toyota"],
            annot_kws={"size": 14})
ax.set_title("Correlacao entre Retornos Diarios", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "correlation_heatmap.png")

# 6. Drawdown
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, t, label, color in zip(axes, ["TSLA", "BYD", "TOYOTA"], ["Tesla", "BYD", "Toyota"], colors):
    dd = df[f"drawdown_{t}"]
    ax.fill_between(df["Date"], dd, 0, alpha=0.4, color=color)
    ax.plot(df["Date"], dd, color=color, linewidth=0.8)
    ax.set_ylabel("Drawdown")
    ax.set_title(f"Drawdown - {label}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout()
save(fig, "drawdown.png")

# 7. Sharpe rolling
fig, ax = plt.subplots(figsize=(14, 5))
for t, label, color in [("TSLA", "Tesla", "#e74c3c"), ("BYD", "BYD", "#3498db"), ("TOYOTA", "Toyota", "#2ecc71")]:
    roll_mean = df[f"return_{t}"].rolling(60).mean()
    roll_std = df[f"return_{t}"].rolling(60).std()
    sharpe = (roll_mean / (roll_std + 1e-12)) * np.sqrt(252)
    ax.plot(df["Date"], sharpe, label=label, linewidth=1, color=color)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Sharpe Ratio Rolling 60 dias (anualizado)", fontsize=14, fontweight="bold")
ax.set_ylabel("Sharpe Ratio")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout()
save(fig, "sharpe_ratio.png")

# 8. Yearly returns
fig, ax = plt.subplots(figsize=(12, 5))
width = 0.25
years = df["Date"].dt.year.unique()
x = np.arange(len(years))
for i, (t, label, color) in enumerate([("TSLA", "Tesla", "#e74c3c"), ("BYD", "BYD", "#3498db"), ("TOYOTA", "Toyota", "#2ecc71")]):
    yr_returns = []
    for yr in years:
        yr_data = df[df["Date"].dt.year == yr]
        if len(yr_data) > 1:
            ret = yr_data[f"Close_{t}"].iloc[-1] / yr_data[f"Close_{t}"].iloc[0] - 1
        else:
            ret = 0
        yr_returns.append(ret)
    ax.bar(x + i * width, yr_returns, width, label=label, color=color)
ax.set_xticks(x + width)
ax.set_xticklabels(years)
ax.set_title("Retornos Anuais por Empresa", fontsize=14, fontweight="bold")
ax.set_ylabel("Retorno")
ax.axhline(0, color="black", linewidth=0.5)
ax.legend()
fig.tight_layout()
save(fig, "yearly_returns.png")

print("\nDone! All charts saved to assets/")
