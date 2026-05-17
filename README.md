# Analise Quantitativa Comparativa: BYD vs Tesla vs Toyota

> **Projeto de analise de dados financeiros** comparando tres gigantes do setor automotivo global, com foco em veiculos eletricos (EVs). Pipeline completo de EDA, analise tecnica, modelagem preditiva e backtesting de estrategias.

**Periodo analisado:** Janeiro 2018 a Fevereiro 2026 (2.035 pregoes)

---

## Storytelling: A Batalha dos Eletricos

O setor automotivo vive uma transformacao historica. Enquanto a **Tesla** pioneirizou o mercado de EVs e se tornou a montadora mais valiosa do mundo, a **BYD** emergiu da China como a maior fabricante de veiculos eletricos do planeta, e a **Toyota**, gigante japonesa, busca se reinventar apos decadas dominando o mercado de combustao.

Este analise mergulha em **2.035 dias de dados de mercado** para responder:

- Qual empresa ofereceu melhor risco-retorno?
- Existe correlacao entre elas (oportunidade de diversificacao)?
- Modelos de Machine Learning conseguem prever movimentos diarios?
- Estrategias tecnicas simples superam modelos complexos?

---

## Resultados Principais

### Performance Acumulada (2018-2026)

| Empresa | Retorno Total | Retorno Medio Diario | Volatilidade | Sharpe Anual |
|---------|--------------|---------------------|--------------|--------------|
| **Tesla (TSLA)** | Alto | 0.22% | 4.00% | **0.89** |
| **BYD (BYDDF)** | Medio | 0.12% | 3.12% | 0.59 |
| **Toyota (TM)** | Conservador | 0.05% | 1.56% | 0.53 |

### Descobertas

1. **Tesla** entrega o maior retorno, mas com a maior volatilidade — Sharpe 0.89 indica bom risco-retorno ajustado
2. **BYD** mostrou crescimento consistente, refletindo a dominancia chinesa no mercado de EVs
3. **Toyota** oferece o perfil mais conservador, ideal para investidores avessos ao risco
4. **Correlacao entre TSLA e BYD** varia ao longo do tempo — oportunidades de diversificacao dinamica
5. **Modelos de ML** (AUC ~0.52) mostram que o mercado de acoes e altamente eficiente — prever movimentos diarios e extremamente dificil

---

## Visualizacoes

### Evolucao dos Precos
![Evolucao dos Precos](assets/price_evolution.png)

### Performance Normalizada (Base 100)
![Performance Normalizada](assets/normalized_performance.png)

### Distribuicao dos Retornos Diarios
![Distribuicao dos Retornos](assets/returns_distribution.png)

> Os retornos apresentam **caudas pesadas** (leptocurticos), violando a hipotese de normalidade. Isso impacta diretamente modelos que assumem distribuicao normal (VaR parametrico, Black-Scholes).

### Retornos Anuais
![Retornos Anuais](assets/yearly_returns.png)

### Volatilidade Rolling
![Volatilidade](assets/volatility.png)

### Sharpe Ratio Rolling (60 dias)
![Sharpe Ratio](assets/sharpe_ratio.png)

### Drawdown
![Drawdown](assets/drawdown.png)

> O drawdown mostra a pior perda consecutiva de cada ativo. Tesla apresentou os maiores drawdowns, exigindo disciplina e gestao de risco rigorosa.

### Correlacao entre Retornos
![Correlacao](assets/correlation_heatmap.png)

---

## Analise Tecnica Aplicada

O pipeline inclui indicadores tecnicos classicos para todas as tres empresas:

| Indicador | Descricao | Aplicacao |
|-----------|-----------|-----------|
| **RSI(14)** | Indice de Forca Relativa | Identificar sobrecompra/sobrevenda |
| **Bollinger Bands (20,2)** | Bandas de volatilidade | Detectar breakouts e reversoes |
| **MACD (12,26,9)** | Convergencia/Divergencia de Medias | Confirmar tendencias |
| **MA(50, 100, 200)** | Medias moveis simples | Identificar suporte/resistencia |
| **ATR(14)** | Amplitude Verdadeira Media | Medir volatilidade para stop-loss |

---

## Modelagem Preditiva

### Target
Prever se o retorno do dia seguinte da Tesla sera **positivo** (classificacao binaria).

### Validacao
**Time Series Split** com 5 folds — respeita a ordem temporal dos dados (sem data leakage).

### Modelos Testados

| Modelo | Acuracia | Precisao | Recall | ROC-AUC |
|--------|----------|----------|--------|---------|
| Logistic Regression | 0.5098 | - | - | 0.4987 |
| **Random Forest** | 0.5026 | - | - | **0.5202** |
| Gradient Boosting | 0.4882 | - | - | 0.4998 |

> **Interpretacao:** AUC proximo de 0.5 indica que os modelos tem dificuldade em superar o acaso. Isso e **esperado** em mercados financeiros eficientes — se fosse facil prever, todos seriam ricos.

### Features Utilizadas (142 colunas)
- Retornos defasados (lags 1-5)
- Medias moveis (5, 7, 14, 21, 30, 50, 100, 200 periodos)
- Volatilidade rolling (20, 60 dias)
- Momentum (10, 20 dias)
- RSI(14), Bollinger Bands, MACD
- ATR(14), Drawdown
- Correlacoes cross-empresa (rolling 30, 60 dias)
- Spreads e ratios entre ativos

---

## Backtesting de Estrategias

### Estrategias ML (com threshold de 55%)

| Modelo | Retorno | Buy & Hold | Sharpe | Max DD | Win Rate |
|--------|---------|------------|--------|--------|----------|
| Logistic Regression | -1.27% | - | 0.21 | -27.78% | - |
| Random Forest | -24.19% | - | -1.38 | -24.19% | - |
| Gradient Boosting | -37.09% | - | -1.57 | -37.09% | - |

### Estrategias Tecnicas (Analise Pura)

O relatorio completo inclui backtests de:
- **MA Crossover (50/200):** Estrategia classica de tendencia
- **RSI Mean Reversion:** Compra em sobrevenda, vende em sobrecompra
- **Bollinger Bands:** Breakout das bandas

---

## Estrutura do Projeto

```
byd-analise/
├── README.md                  # Este arquivo
├── requirements.txt           # Dependencias Python
├── .gitignore                 # Arquivos ignorados
├── data/
│   └── auto_company_comparison.csv   # Dados brutos (Yahoo Finance)
├── src/
│   ├── full_analysis.py       # Pipeline completo de analise
│   └── generate_readme_charts.py  # Gerador de graficos para README
└── output/
    └── relatorio_completo.html     # Relatorio interativo (gerado)
```

---

## Como Executar

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Executar o pipeline completo

```bash
python src/full_analysis.py
```

O script gera automaticamente o relatorio HTML em `output/relatorio_completo.html`.

### 3. Gerar graficos do README (opcional)

```bash
python src/generate_readme_charts.py
```

---

## Tecnologias Utilizadas

| Categoria | Ferramenta |
|-----------|------------|
| **Linguagem** | Python 3.13 |
| **Manipulacao de dados** | Pandas, NumPy |
| **Visualizacao** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression) |
| **Estatistica** | SciPy (Jarque-Bera, Pearson) |
| **Clustering** | KMeans (deteccao de regimes) |
| **Validacao** | TimeSeriesSplit (walk-forward) |

---

## Metricas de Risco Calculadas

| Metrica | O que mede | Por que importa |
|---------|------------|-----------------|
| **VaR (95%)** | Perda maxima em 95% dos dias | Limite de perda esperada |
| **CVaR (95%)** | Media das perdas nos 5% piores dias | Risco de cauda (mais conservador) |
| **Sharpe Ratio** | Retorno ajustado ao risco | Comparar estrategias justo |
| **Max Drawdown** | Maior perda consecutiva | Preparacao psicologica |
| **Volatilidade** | Dispersao dos retornos | Medida de risco basica |

---

## Limitacoes e Disclaimer

- **Retornos passados nao garantem resultados futuros**
- Modelos de ML para previsao de acoes tem eficacia limitada (mercado eficiente)
- Custos de transacao, slippage e impostos nao foram considerados no backtest
- Esta analise e **educacional** e nao constitui recomendacao de investimento
- Consulte um profissional financeiro antes de tomar decisoes de investimento

---

## Autor

**Henry** | Projeto academico EBAC (Escola Brasileira de Analise de Dados)

---

## Licenca

Este projeto e de uso educacional. Os dados de mercado foram obtidos via Yahoo Finance.
