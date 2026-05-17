# Analise Quantitativa: A Guerra dos Veiculos Eletricos

### BYD vs Tesla vs Toyota — Quem esta vencendo a transicao energetica automotiva?

> **Em 2025, a BYD ultrapassou a Tesla em vendas globais de EVs pela primeira vez.**
> A Toyota, que dominou o mercado por decadas, agora corre contra o tempo.
> Esta analise investiga 8 anos de dados de mercado para entender quem esta vencendo — e por que.

**Periodo:** Janeiro 2018 a Fevereiro 2026 | **2.035 pregoes** | **3 gigantes globais**

---

## Por que este projeto?

O setor automotivo esta vivendo a maior transformacao desde a invencao do motor a combustao. A questao nao e **se** os veiculos eletricos vao dominar, mas **quem** vai dominar.

Este projeto nasce de uma pergunta de negocio real:

> **Se voce tivesse R$10.000 para investir em janeiro de 2018, qual empresa teria gerado mais riqueza — e com quanto risco?**

Para responder isso, construi um pipeline quantitativo completo que vai muito alem de "olhar graficos". Aqui voce vai encontrar analise estatistica rigorosa, modelagem preditiva com Machine Learning, e backtesting de estrategias reais de investimento.

---

## Key Insights (O que voce vai aprender)

### Insight 1: A Tesla entregou o melhor risco-retorno, mas a historia esta mudando

A Tesla gerou retorno medio de **0.22% ao dia** (Sharpe 0.89), superando amplamente BYD e Toyota. Porem, a volatilidade de 4% ao dia significa que em qualquer dia aleatorio, voce poderia ganhar ou perder 4% do seu capital. **Nao e para todos.**

### Insight 2: A BYD e a verdadeira historia de crescimento

Enquanto a Tesla chamava atencao da midia, a BYD cresceu silenciosamente. Com retorno medio de 0.12% ao dia e volatilidade menor (3.12%), a BYD oferece um perfil mais equilibrado. **A dominancia chinesa no mercado de EVs nao e acidental — e refletida nos dados.**

### Insight 3: A Toyota e o risco da transicao lenta

Com apenas 0.05% de retorno medio diario e volatilidade de 1.56%, a Toyota e o ativo mais conservador. Mas conservadorismo tem um custo: **quem ficou so na Toyota perdeu oportunidades enormes.** A lentidao na transicao para EVs esta precificada no mercado.

### Insight 4: O mercado de acoes e brutalmente eficiente

Testamos 3 modelos de Machine Learning (Logistic Regression, Random Forest, Gradient Boosting) com 142 features. **Nenhum conseguiu prever movimentos diarios com AUC significativamente acima de 0.5.** Isso confirma a Hipotese do Mercado Eficiente: se fosse facil prever, todos seriam ricos.

### Insight 5: Diversificacao funciona — mas as correlacoes mudam

A correlacao entre Tesla e BYD varia ao longo do tempo. Em momentos de crise, elas tendem a cair juntas (correlacao sobe). **Diversificar entre EVs nao e suficiente — voce precisa de ativos de setores diferentes.**

---

## Analise Completa

### 1. Evolucao dos Precos (2018-2026)

![Evolucao dos Precos](assets/price_evolution.png)

**O que voce ve:** Tres trajetorias completamente distintas. A Tesla (vermelho) teve a maior valorizacao, seguida pela BYD (azul). A Toyota (verde) cresceu de forma mais estavel.

**O que isso significa:** O mercado precificou diferentemente o potencial de cada empresa na transicao energetica. A Tesla liderou por ser pioneira, mas a BYD esta ganhando terreno com agressividade comercial e custos mais baixos.

---

### 2. Performance Normalizada — A Comparacao Justa

![Performance Normalizada](assets/normalized_performance.png)

**O que voce ve:** Todas as acoes comecam em 100 e evoluem proporcionalmente. Isso permite comparar valorizacoes percentuais, ignorando diferencas de preco.

**O que isso significa:** Se voce tivesse investido R$10.000 em cada empresa em janeiro de 2018, o resultado acumulado ate 2026 mostra claramente quem gerou mais riqueza. A diferenca entre as curvas e a diferenca entre oportunidades aproveitadas e perdidas.

---

### 3. Distribuicao dos Retornos — O Risco Real

![Distribuicao dos Retornos](assets/returns_distribution.png)

**O que voce ve:** Os histogramas mostram como os retornos diarios se distribuem. A linha vermelha e a distribuicao normal teorica.

**O que isso significa:** **Os retornos nao seguem distribuicao normal.** Eles tem caudas pesadas — eventos extremos (ganhos ou perdas de 5%+) acontecem muito mais frequentemente do que a teoria estatistica classica prevê. Isso tem implicacoes diretas:
- VaR parametrico subestima riscos
- Modelos que assumem normalidade falham
- Stop-loss e gestao de risco sao essenciais

---

### 4. Retornos Anuais — A Historia por Ano

![Retornos Anuais](assets/yearly_returns.png)

**O que voce ve:** O desempenho de cada empresa ano a ano.

**O que isso significa:** Ha anos onde a Tesla domina (2020, 2023), anos onde a BYD surpreende (2021, 2024), e anos onde ate a Toyota se destaca. **Nenhuma empresa vence sempre.** Timing de entrada e saida importa.

---

### 5. Volatilidade — O Preço da Incerteza

![Volatilidade](assets/volatility.png)

**O que voce ve:** A volatilidade rolling de 20 dias para cada ativo. Picos indicam periodos de incerteza.

**O que isso significa:** A volatilidade nao e constante — ela muda com o contexto macro, resultados trimestrais e noticias do setor. **Periodos de alta volatilidade sao oportunidades para traders, mas pesadelos para investidores de longo prazo.** A Tesla consistentemente tem a maior volatilidade, refletindo a natureza especulativa do ativo.

---

### 6. Sharpe Ratio — Quem Paga Melhor pelo Risco?

![Sharpe Ratio](assets/sharpe_ratio.png)

**O que voce ve:** O Sharpe Ratio rolling de 60 dias (anualizado). Valores acima de 1.0 indicam retorno superior ao risco assumido.

**O que isso significa:** O Sharpe Ratio e a metrica mais importante para comparar investimentos. **Um Sharpe de 0.89 (Tesla) significa que para cada unidade de risco, voce recebe 0.89 unidades de retorno.** Isso e bom, mas nao excepcional. A BYD (0.59) e Toyota (0.53) pagam menos pelo risco.

---

### 7. Drawdown — A Verdade Sobre Perdas

![Drawdown](assets/drawdown.png)

**O que voce ve:** O drawdown mostra a distancia percentual entre o preco atual e o maximo historico. Quando esta em -50%, significa que o ativo perdeu metade do valor desde o pico.

**O que isso significa:** **Este e o grafico mais importante para investidores.** A Tesla ja perdeu mais de 60% do valor em alguns periodos. Se voce nao consegue dormir com uma perda de 30% na carteira, Tesla nao e para voce. A Toyota, com drawdowns menores, e para quem prioriza paz de espirito.

---

### 8. Correlacao — Diversificacao Real

![Correlacao](assets/correlation_heatmap.png)

**O que voce ve:** A matriz de correlacao entre os retornos diarios das tres empresas.

**O que isso significa:** Correlacao baixa entre ativos e a base da diversificacao. **Se Tesla e BYD caem juntas (alta correlacao), investir nas duas nao protege seu portfolio.** Os dados mostram que a correlacao varia ao longo do tempo — em crises, ela sobe, e a diversificacao falha quando mais precisa.

---

## Modelagem Preditiva

### Pergunta: Da para prever se a Tesla vai subir amanha?

**Abordagem:** Classificacao binaria (sobe vs desce) com validacao temporal (TimeSeriesSplit, 5 folds).

### Resultados

| Modelo | Acuracia | ROC-AUC | Veredito |
|--------|----------|---------|----------|
| Logistic Regression | 50.98% | 0.4987 | Aleatorio |
| **Random Forest** | 50.26% | **0.5202** | Levemente acima |
| Gradient Boosting | 48.82% | 0.4998 | Aleatorio |

### Interpretacao Estrategica

**AUC de 0.52 e estatisticamente indistinguivel de aleatoriedade.** Isso nao e uma falha do modelo — e uma confirmacao de que mercados financeiros sao altamente eficientes. Informacoes publicas ja estao precificadas.

**O que funciona em ML financeiro:** Nao previsao de direcao, mas sim:
- Deteccao de anomalias
- Otimizacao de portfolio
- Execucao algoritmica
- Analise de sentimento

---

## Backtesting — Estrategias Reais

### Estrategia ML: Comprar quando P(subida) > 55%

| Modelo | Retorno Total | Sharpe | Max Drawdown |
|--------|--------------|--------|--------------|
| Logistic Regression | -1.27% | 0.21 | -27.78% |
| Random Forest | -24.19% | -1.38 | -24.19% |
| Gradient Boosting | -37.09% | -1.57 | -37.09% |

### Licao Principal

**Nenhuma estrategia de ML superou o Buy & Hold de forma consistente.** Isso reforça que:
1. Overfitting e o maior risco em ML financeiro
2. Simplicidade muitas vezes vence complexidade
3. Custo de transacao destruiria qualquer edge marginal

---

## Stack Tecnico

```
Python 3.13
├── pandas / numpy          → Manipulacao de dados
├── matplotlib / seaborn    → Visualizacao
├── scikit-learn            → ML (RF, GBM, LR, KMeans, TimeSeriesSplit)
├── scipy                   → Testes estatisticos (Jarque-Bera, Pearson)
└── Pipeline customizado    → Feature engineering (142 features)
```

### Features Engineradas (142 variaveis)

| Categoria | Exemplos | Quantidade |
|-----------|----------|------------|
| Retornos | lag 1-5, pct_change | 15 |
| Medias moveis | SMA/EMA 5-200 | 48 |
| Volatilidade | rolling std 20/60 | 6 |
| Momentum | 10/20 dias | 6 |
| Indicadores tecnicos | RSI, Bollinger, MACD, ATR | 24 |
| Cross-empresa | correlacao, spread, ratio | 36 |
| Drawdown | drawdown acumulado | 3 |

---

## Como Rodar

```bash
# 1. Clone o repositorio
git clone https://github.com/SEU_USUARIO/byd-analise.git
cd byd-analise

# 2. Instale dependencias
pip install -r requirements.txt

# 3. Execute o pipeline completo
python src/full_analysis.py

# 4. Abra o relatorio gerado
start output/relatorio_completo.html
```

---

## Estrutura do Projeto

```
byd-analise/
│
├── README.md                        # Este documento
├── requirements.txt                 # Dependencias
├── .gitignore
│
├── data/
│   └── auto_company_comparison.csv  # Dados brutos (Yahoo Finance API)
│
├── src/
│   ├── full_analysis.py             # Pipeline completo (1170+ linhas)
│   └── generate_readme_charts.py    # Gerador de graficos
│
├── assets/                          # Visualizacoes do README
│   ├── price_evolution.png
│   ├── normalized_performance.png
│   ├── returns_distribution.png
│   ├── volatility.png
│   ├── correlation_heatmap.png
│   ├── drawdown.png
│   ├── sharpe_ratio.png
│   └── yearly_returns.png
│
└── output/
    └── relatorio_completo.html      # Relatorio interativo (gerado)
```

---

## Conclusao

> **A transicao para veiculos eletricos nao e uma tendencia — e uma realidade ja precificada pelo mercado.**

Os dados mostram que:
1. **BYD esta ganhando market share** com um modelo de negocio agressivo e custos mais baixos
2. **Tesla manteve a lideranca em valorizacao**, mas com risco significativamente maior
3. **Toyota esta em transicao**, e o mercado esta precificando a incerteza
4. **Prever movimentos diarios de acoes e extremamente dificil** — ate modelos sofisticados de ML falham
5. **Diversificacao e gestao de risco** sao mais importantes do que escolher a "proxima Tesla"

**Para investidores:** Os dados suportam uma abordagem de longo prazo com diversificacao entre setores, ao inves de tentar prever movimentos de curto prazo.

**Para analistas:** Este projeto demonstra que dados financeiros exigem rigor estatissional — normalidade nao existe, correlacoes mudam, e modelos lineares tem limites claros.

---

**Autor:** Henry | Projeto academico EBAC (Escola Brasileira de Analise de Dados)

**Disclaimer:** Esta analise e exclusivamente educacional e nao constitui recomendacao de investimento. Retornos passados nao garantem resultados futuros. Consulte um profissional financeiro antes de tomar decisoes de investimento.
