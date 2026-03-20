"""
Dashboard Streamlit Integrado
Integra as 3 fases do projeto de Ciência de Dados
- Fase 1: Manipulação e Visualização
- Fase 2: Engenharia e NLP
- Fase 3: Séries Temporais e Previsão
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend para Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Adicionar src ao path
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase1_data_processing import process_data
from phase2_nlp_engineering import process_features_and_nlp
from phase3_time_series import process_time_series


# Configuração Streamlit
st.set_page_config(
    page_title="Dashboard Ciência de Dados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo
st.markdown("""
    <style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    """Carrega e processa dados de todas as fases."""
    try:
        st.info("⏳ Carregando e processando dados...")
        
        # Fase 1
        df = process_data()
        
        # Fase 2
        df, nlp_analyzer = process_features_and_nlp(df)
        
        # Fase 3
        ts_analyzer, ts_results = process_time_series(df)
        
        return df, nlp_analyzer, ts_analyzer, ts_results
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        return None, None, None, None


# ===== SIDEBAR =====
st.sidebar.title("🎛️ Controles")
fase = st.sidebar.radio(
    "Selecione a Fase",
    ["📈 Visão Geral", "🔍 Fase 1: Exploração", "💡 Fase 2: NLP", "📊 Fase 3: Previsões"]
)

# ===== MAIN CONTENT =====
def render_overview(df, nlp_analyzer, ts_analyzer, ts_results):
    """Renderiza visão geral do projeto."""
    st.title("📊 Dashboard - Análise de Dados OLIST")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Pedidos", len(df))
    
    with col2:
        st.metric("Receita Total", f"R${df['order_total'].sum():,.0f}")
    
    with col3:
        st.metric("Ticket Médio", f"R${df['order_total'].mean():,.0f}")
    
    with col4:
        if 'review_score' in df.columns:
            avg_score = df['review_score'].mean()
            st.metric("Avaliação Média", f"{avg_score:.2f} ⭐")
    
    st.divider()
    
    # Resumo de cada fase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔍 Fase 1: Dados")
        st.write(f"**Datasets unificados:** {df.shape[0]} linhas × {df.shape[1]} colunas")
        st.write(f"**Dados limpos:** Sem valores críticos faltantes")
        st.write(f"**Features criadas:** delivery_time, order_total, sentiment")
    
    with col2:
        st.subheader("💡 Fase 2: NLP")
        st.write(f"**Reviews processados:** Com TF-IDF")
        st.write(f"**Sentimentos:** Negativo, Neutro, Positivo")
        st.write(f"**Features extraídas:** {100} termos principais")
    
    with col3:
        st.subheader("📊 Fase 3: Séries Temporais")
        if ts_results and 'metrics' in ts_results:
            metrics = ts_results['metrics']
            st.write(f"**RMSE (teste):** {metrics.get('rmse_test', 0):.2f}")
            st.write(f"**MAE (teste):** {metrics.get('mae_test', 0):.2f}")
            st.write(f"**Períodos previsto:** 12 meses futuros")


def render_fase1(df):
    """Renderiza Fase 1: Exploração e Visualização."""
    st.title("🔍 Fase 1: Manipulação e Visualização de Dados")
    
    st.subheader("1️⃣ Informações dos Dados")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Shape:** {df.shape[0]} linhas × {df.shape[1]} colunas")
    with col2:
        st.write(f"**Período:** {df['order_purchase_timestamp'].min().date()} a {df['order_purchase_timestamp'].max().date()}")
    
    # Valores nulos
    st.subheader("2️⃣ Qualidade dos Dados")
    null_data = pd.DataFrame({
        'Coluna': df.columns,
        'Valores Nulos': [df[col].isnull().sum() for col in df.columns],
        '% Nulos': [(df[col].isnull().sum() / len(df)) * 100 for col in df.columns]
    })
    null_data = null_data[null_data['Valores Nulos'] > 0].sort_values('% Nulos', ascending=False)
    
    if len(null_data) > 0:
        st.dataframe(null_data, use_container_width=True)
    else:
        st.success("✓ Nenhum valor nulo crítico encontrado")
    
    # Filtros
    st.subheader("3️⃣ Filtros e Análises")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'order_status' in df.columns:
            status_filter = st.multiselect(
                "Status do Pedido",
                options=df['order_status'].unique(),
                default=df['order_status'].unique()[:3]
            )
            df_filtered = df[df['order_status'].isin(status_filter)]
        else:
            df_filtered = df
    
    with col2:
        if 'product_category_name' in df.columns:
            categories = df['product_category_name'].unique()[:10]
            cat_filter = st.multiselect(
                "Categorias de Produtos",
                options=categories,
                default=categories[:3]
            )
            df_filtered = df_filtered[df_filtered['product_category_name'].isin(cat_filter)]
    
    # Visualizações
    st.subheader("4️⃣ Visualizações")
    
    col1, col2 = st.columns(2)
    
    # Gráfico 1: Vendas por status
    with col1:
        if 'order_status' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            status_counts = df_filtered['order_status'].value_counts()
            status_counts.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title("Pedidos por Status", fontsize=12, fontweight='bold')
            ax.set_xlabel("Status")
            ax.set_ylabel("Quantidade")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # Gráfico 2: Distribuição de preços
    with col2:
        if 'price' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            df_filtered['price'].hist(bins=30, ax=ax, color='coral', edgecolor='black')
            ax.set_title("Distribuição de Preços", fontsize=12, fontweight='bold')
            ax.set_xlabel("Preço (R$)")
            ax.set_ylabel("Frequência")
            st.pyplot(fig)
    
    # Tabela de dados
    st.subheader("5️⃣ Amostra de Dados")
    columns_to_show = [
        col for col in ['order_id', 'order_purchase_timestamp', 'price', 
                       'order_status', 'product_category_name', 'review_score']
        if col in df_filtered.columns
    ]
    st.dataframe(df_filtered[columns_to_show].head(10), use_container_width=True)


def render_fase2(df, nlp_analyzer):
    """Renderiza Fase 2: NLP e Engenharia de Features."""
    st.title("💡 Fase 2: Engenharia de Features e NLP")
    
    st.subheader("1️⃣ Features Criados")
    features_info = {
        'Feature': ['Numéricos (Escalados)', 'Categóricos (One-Hot Encoded)', 'TF-IDF (Reviews)'],
        'Descrição': [
            'Price, Freight, Delivery Time',
            'Order Status, Product Category',
            'Termos mais frequentes em reviews'
        ]
    }
    st.dataframe(pd.DataFrame(features_info), use_container_width=True)
    
    # Análise de Sentimento
    st.subheader("2️⃣ Análise de Sentimento")
    
    if 'review_sentiment' in df.columns:
        sentiment_counts = df['review_sentiment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sentiment_counts.plot(kind='bar', ax=ax, color=['#d62728', '#ff7f0e', '#2ca02c'])
            ax.set_title("Distribuição de Sentimentos", fontsize=12, fontweight='bold')
            ax.set_xlabel("Sentimento")
            ax.set_ylabel("Quantidade")
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
        with col2:
            sentiment_pct = (sentiment_counts / sentiment_counts.sum()) * 100
            fig, ax = plt.subplots(figsize=(8, 5))
            sentiment_pct.plot(kind='pie', ax=ax, autopct='%1.1f%%',
                             colors=['#d62728', '#ff7f0e', '#2ca02c'])
            ax.set_title("% de Sentimentos", fontsize=12, fontweight='bold')
            ax.set_ylabel("")
            st.pyplot(fig)
    
    # Top Termos TF-IDF
    st.subheader("3️⃣ Termos Principais (TF-IDF)")
    
    if nlp_analyzer and nlp_analyzer.feature_names is not None:
        top_terms = nlp_analyzer.get_top_terms(top_n=15)
        
        if 'top_termos' in top_terms:
            terms_df = pd.DataFrame(
                top_terms['top_termos'],
                columns=['Termo', 'Score TF-IDF']
            )
            st.dataframe(terms_df, use_container_width=True)
            
            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            terms_df_sorted = terms_df.sort_values('Score TF-IDF')
            ax.barh(terms_df_sorted['Termo'], terms_df_sorted['Score TF-IDF'], color='teal')
            ax.set_title("Top 15 Termos em Reviews", fontsize=12, fontweight='bold')
            ax.set_xlabel("Score TF-IDF")
            st.pyplot(fig)
    
    # Estatísticas por sentimento
    st.subheader("4️⃣ Análise por Sentimento")
    
    if 'review_sentiment' in df.columns and 'review_score' in df.columns:
        sentiment_stats = df.groupby('review_sentiment')['review_score'].agg(['count', 'mean', 'min', 'max'])
        sentiment_stats.columns = ['Quantidade', 'Score Médio', 'Score Mín', 'Score Máx']
        st.dataframe(sentiment_stats, use_container_width=True)


def render_fase3(df, ts_analyzer, ts_results):
    """Renderiza Fase 3: Séries Temporais e Previsões."""
    st.title("📊 Fase 3: Séries Temporais e Previsão de Demanda")
    
    if ts_results is None or not ts_results:
        st.warning("⚠️ Nenhum resultado de série temporal disponível")
        return
    
    # Métricas
    st.subheader("1️⃣ Métricas do Modelo")
    
    if 'metrics' in ts_results:
        metrics = ts_results['metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE (Treino)", f"{metrics.get('rmse_train', 0):.2f}")
        with col2:
            st.metric("RMSE (Teste)", f"{metrics.get('rmse_test', 0):.2f}")
        with col3:
            st.metric("MAE (Teste)", f"{metrics.get('mae_test', 0):.2f}")
        with col4:
            st.metric("Tamanho Teste", f"{metrics.get('test_size', 0)} períodos")
    
    # Série Temporal
    st.subheader("2️⃣ Série Temporal Histórica")
    
    if ts_analyzer and ts_analyzer.sales_ts is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ts_data = ts_analyzer.sales_ts.sort_values('order_month_name')
        ax.plot(ts_data['order_month_name'], ts_data['total_vendas'], 
               marker='o', linewidth=2, label='Vendas Histórico', color='steelblue')
        ax.set_title("Série Temporal de Vendas", fontsize=12, fontweight='bold')
        ax.set_xlabel("Período")
        ax.set_ylabel("Total de Vendas (R$)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Estatísticas
        st.write(f"**Período:** {ts_data['order_month_name'].min().date()} a {ts_data['order_month_name'].max().date()}")
        st.write(f"**Vendas médias:** R${ts_data['total_vendas'].mean():,.0f}")
        st.write(f"**Vendas máximas:** R${ts_data['total_vendas'].max():,.0f}")
        st.write(f"**Vendas mínimas:** R${ts_data['total_vendas'].min():,.0f}")
    
    # Previsões
    st.subheader("3️⃣ Previsão de Demanda (12 meses)")
    
    if 'forecast' in ts_results and ts_results['forecast'] is not None:
        forecast_df = ts_results['forecast']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Dados históricos
        if ts_analyzer and ts_analyzer.sales_ts is not None:
            ts_data = ts_analyzer.sales_ts.sort_values('order_month_name')
            ax.plot(ts_data['order_month_name'], ts_data['total_vendas'],
                   marker='o', linewidth=2, label='Histórico', color='steelblue')
        
        # Previsões
        ax.plot(forecast_df['data'], forecast_df['vendas_previstas'],
               marker='s', linewidth=2, label='Previsão', color='coral', linestyle='--')
        
        # Intervalo de confiança
        ax.fill_between(forecast_df['data'],
                        forecast_df['intervalo_inferior'],
                        forecast_df['intervalo_superior'],
                        alpha=0.2, color='coral', label='Intervalo ±15%')
        
        ax.set_title("Previsão de Vendas (12 meses)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Período")
        ax.set_ylabel("Vendas Previstas (R$)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Tabela de previsões
        st.subheader("4️⃣ Detalhes das Previsões")
        forecast_display = forecast_df.copy()
        forecast_display['data'] = forecast_display['data'].dt.strftime('%Y-%m')
        st.dataframe(forecast_display, use_container_width=True)
    
    # Decomposição
    st.subheader("5️⃣ Decomposição da Série")
    
    if ts_analyzer and ts_analyzer.decomposed:
        decomp = ts_analyzer.decomposed
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original
        axes[0].plot(decomp['original'], color='steelblue', linewidth=1.5)
        axes[0].set_ylabel('Original')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(decomp['trend'], color='orange', linewidth=1.5)
        axes[1].set_ylabel('Tendência')
        axes[1].grid(True, alpha=0.3)
        
        # Sazonalidade
        axes[2].plot(decomp['seasonality'], color='green', linewidth=1.5)
        axes[2].set_ylabel('Sazonalidade')
        axes[2].grid(True, alpha=0.3)
        
        # Ruído
        axes[3].plot(decomp['noise'], color='red', linewidth=1.5)
        axes[3].set_ylabel('Ruído')
        axes[3].set_xlabel('Período')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)


# ===== MAIN LOGIC =====
def main():
    """Função principal do dashboard."""
    
    # Carregar dados
    try:
        df, nlp_analyzer, ts_analyzer, ts_results = load_all_data()
        
        # Validar carregamento
        if df is None:
            st.error("❌ Falha ao carregar dados do projeto")
            return
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        return
    
    # Renderizar página selecionada
    if fase == "📈 Visão Geral":
        render_overview(df, nlp_analyzer, ts_analyzer, ts_results)
    
    elif fase == "🔍 Fase 1: Exploração":
        render_fase1(df)
    
    elif fase == "💡 Fase 2: NLP":
        render_fase2(df, nlp_analyzer)
    
    elif fase == "📊 Fase 3: Previsões":
        render_fase3(df, ts_analyzer, ts_results)


if __name__ == "__main__":
    main()
