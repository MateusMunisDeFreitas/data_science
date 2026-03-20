# Projeto Ciência de Dados - OLIST

Projeto completo de análise de dados e ciência de dados em Python, cobrindo manipulação, engenharia de features, NLP e séries temporais.

## 🎯 Fases do Projeto

### Fase 1: Manipulação e Visualização de Dados
- **Objetivo:** Unificar, limpar e explorar os dados OLIST
- **Atividades:**
  - Unificação das tabelas: orders, items, products, reviews, customers, sellers
  - Tratamento de dados nulos, datas e duplicatas
  - Criação de features (tempo de entrega, valor total, sentimento)
  - Dashboard com filtros por estado e categoria
- **Arquivo:** `src/phase1_data_processing.py`

### Fase 2: Engenharia de Features e NLP
- **Objetivo:** Processar dados estruturados e textuais para ML
- **Atividades:**
  - Pipeline de pré-processamento (OneHotEncoder, StandardScaler)
  - Aplicar TF-IDF nos reviews
  - Identificar termos positivos/negativos por sentimento
  - Visualização de analysis de sentimento
- **Arquivo:** `src/phase2_nlp_engineering.py`

### Fase 3: Séries Temporais e Previsão de Demanda
- **Objetivo:** Prever demanda de vendas futuras
- **Atividades:**
  - Agrupar vendas por período (mensal)
  - Decomposição da série temporal (tendência, sazonalidade, ruído)
  - Treinar modelo preditivo (Polynomial Regression)
  - Validar com métricas RMSE e MAE
  - Gerar previsões para 12 meses futuros
- **Arquivo:** `src/phase3_time_series.py`

## 📊 Dashboard Streamlit

O projeto inclui um dashboard interativo que integra todas as 3 fases.

### Como Executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar o dashboard
streamlit run app.py
```

O dashboard abrirá em `http://localhost:8501` com as seguintes abas:
- **📈 Visão Geral:** Resumo das 3 fases
- **🔍 Fase 1:** Exploração e visualização dos dados
- **💡 Fase 2:** Análise de NLP e sentimentos
- **📊 Fase 3:** Previsões e séries temporais

## Estrutura do Projeto

```
├── app.py                          # Dashboard Streamlit
├── requirements.txt                # Dependências Python
├── README.md                       # Este arquivo
│
├── config/
│   └── config.yaml                # Configurações do projeto
│
├── data/
│   ├── sample_data.csv            # Dados de exemplo
│   ├── processed/                 # Dados processados
│   └── olist/                     # Dataset OLIST
│       ├── olist_orders_dataset.csv
│       ├── olist_order_items_dataset.csv
│       ├── olist_order_reviews_dataset.csv
│       ├── olist_products_dataset.csv
│       ├── olist_customers_dataset.csv
│       ├── olist_sellers_dataset.csv
│       ├── olist_geolocation_dataset.csv
│       └── product_category_name_translation.csv
│
├── models/                        # Modelos treinados
│
├── notebooks/
│   ├── analysis.ipynb            # Análise exploratória
│   └── sample_notebook.ipynb     # Notebook de exemplo
│
└── src/
    ├── __init__.py
    ├── main.py                   # Script principal (antes das fases)
    ├── utils.py                  # Utilitários gerais
    ├── phase1_data_processing.py # Fase 1
    ├── phase2_nlp_engineering.py # Fase 2
    └── phase3_time_series.py     # Fase 3
```

## Dependências

- **pandas:** Manipulação de dados
- **numpy:** Computação numérica
- **scikit-learn:** ML e pré-processamento
- **matplotlib & seaborn:** Visualizações
- **streamlit:** Dashboard web
- **scipy:** Computação científica
- **jupyter:** Notebooks

## Configuração do Ambiente

1. Certifique-se de ter Python instalado (versão 3.8 ou superior recomendada).
2. Crie um ambiente virtual: `python -m venv .venv`
3. Ative o ambiente: `.venv\Scripts\activate` (Windows) ou `source .venv/bin/activate` (Linux/Mac)
4. Instale as dependências: `pip install -r requirements.txt`

## Como Usar

### Executar Fases Individuais

```bash
# Fase 1: Processamento de dados
python src/phase1_data_processing.py

# Fase 2: NLP
python src/phase2_nlp_engineering.py

# Fase 3: Séries Temporais
python src/phase3_time_series.py
```

### Usar os Módulos em Código

```python
from src.phase1_data_processing import process_data
from src.phase2_nlp_engineering import process_features_and_nlp
from src.phase3_time_series import process_time_series

# Executar pipeline completo
df = process_data()
df, nlp = process_features_and_nlp(df)
ts_analyzer, results = process_time_series(df)
```

## Métricas e Resultados

### Fase 1
- Datasets unificados com sucesso
- Dados limpos e tratados
- Features criados para análise

### Fase 2
- TF-IDF aplicado em ~X reviews
- Termos principais identificados por sentimento
- Análise de sentimento (Negativo, Neutro, Positivo)

### Fase 3
- **RMSE (Teste):** Métrica de erro do modelo
- **MAE (Teste):** Erro absoluto médio
- **Períodos Previstos:** 12 meses futuros
- Intervalo de confiança ±15%

## Próximos Passos (Opcional)

- [ ] Aplicar BERT para análise de sentimento mais precisa
- [ ] Implementar modelos ARIMA/Prophet
- [ ] Usar PyCaret para AutoML
- [ ] Adicionar mais métricas de validação
- [ ] Deploy do dashboard em produção

## Contato

Desenvolvido como projeto de ciência de dados.