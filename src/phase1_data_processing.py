"""
Fase 1: Manipulação e Visualização de Dados
- Unificar tabelas (orders, items, products, reviews)
- Tratar dados nulos, datas e duplicatas
- Preparar dados para dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_olist_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Carrega todos os datasets OLIST."""
    logger.info("Carregando dados OLIST...")
    
    data_files = {
        'orders': data_dir / 'olist' / 'olist_orders_dataset.csv',
        'items': data_dir / 'olist' / 'olist_order_items_dataset.csv',
        'products': data_dir / 'olist' / 'olist_products_dataset.csv',
        'reviews': data_dir / 'olist' / 'olist_order_reviews_dataset.csv',
        'customers': data_dir / 'olist' / 'olist_customers_dataset.csv',
        'sellers': data_dir / 'olist' / 'olist_sellers_dataset.csv',
        'geolocation': data_dir / 'olist' / 'olist_geolocation_dataset.csv',
        'category_translation': data_dir / 'product_category_name_translation.csv',
    }
    
    datasets = {}
    for name, path in data_files.items():
        if path.exists():
            datasets[name] = pd.read_csv(path)
            logger.info(f"✓ {name}: {len(datasets[name])} linhas")
        else:
            logger.warning(f"✗ Arquivo não encontrado: {path}")
    
    return datasets


def unify_tables(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Unifica orders, items, products e reviews."""
    logger.info("Unificando tabelas...")
    
    # Garantir que as colunas de data estão em datetime
    datasets['orders']['order_purchase_timestamp'] = pd.to_datetime(
        datasets['orders']['order_purchase_timestamp']
    )
    datasets['orders']['order_delivered_customer_date'] = pd.to_datetime(
        datasets['orders']['order_delivered_customer_date'],
        errors='coerce'
    )
    
    datasets['reviews']['review_creation_date'] = pd.to_datetime(
        datasets['reviews']['review_creation_date']
    )
    
    # Merge: orders + items
    df_merged = datasets['orders'].merge(
        datasets['items'],
        on='order_id',
        how='left'
    )
    
    # Merge: + products
    df_merged = df_merged.merge(
        datasets['products'][['product_id', 'product_category_name']],
        on='product_id',
        how='left'
    )
    
    # Merge: + category translation
    if 'category_translation' in datasets:
        df_merged = df_merged.merge(
            datasets['category_translation'],
            left_on='product_category_name',
            right_on='product_category_name',
            how='left'
        )
    
    # Merge: + reviews
    df_merged = df_merged.merge(
        datasets['reviews'][['order_id', 'review_score', 'review_comment_message', 
                             'review_creation_date']],
        on='order_id',
        how='left'
    )
    
    logger.info(f"✓ Tabelas unificadas: {len(df_merged)} linhas, {len(df_merged.columns)} colunas")
    
    return df_merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Trata dados nulos, datas e duplicatas."""
    logger.info("Limpando dados...")
    
    initial_rows = len(df)
    
    # Remover duplicatas completas
    df = df.drop_duplicates()
    logger.info(f"✓ Removidas {initial_rows - len(df)} linhas duplicadas")
    
    # Tratar nulos
    logger.info(f"✓ Valores nulos por coluna:")
    null_counts = df.isnull().sum()
    for col, count in null_counts[null_counts > 0].items():
        pct = (count / len(df)) * 100
        logger.info(f"  - {col}: {count} ({pct:.1f}%)")
        
        # Estratégias de preenchimento
        if col in ['review_score', 'price', 'freight_value']:
            df[col] = df[col].fillna(df[col].median())
        elif col in ['review_comment_message', 'product_category_name']:
            df[col] = df[col].fillna('Não informado')
        elif 'date' in col:
            # Para datas, preencher com forward fill ou valor padrão
            try:
                df[col] = df[col].fillna(df[col].max())
            except:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        else:
            # Para outras colunas, tentar preencher com moda
            try:
                df[col] = df[col].fillna(df[col].mode()[0])
            except:
                df[col] = df[col].fillna('Unknown')
    
    # Remover linhas com valores críticos faltantes
    critical_cols = ['order_id', 'product_id', 'customer_id']
    df = df.dropna(subset=critical_cols)
    
    logger.info(f"✓ Dados limpos: {len(df)} linhas")
    
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features úteis para análise."""
    logger.info("Adicionando features...")
    
    # Tempo de entrega
    df['delivery_time_days'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days
    
    # Valor total do pedido
    df['order_total'] = df['price'] + df['freight_value']
    
    # Categorizar score de review
    df['review_sentiment'] = pd.cut(
        df['review_score'],
        bins=[0, 2, 4, 5],
        labels=['Negativo', 'Neutro', 'Positivo'],
        include_lowest=True
    )
    
    # Mês e ano
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    df['order_year'] = df['order_purchase_timestamp'].dt.year
    df['order_month_name'] = df['order_purchase_timestamp'].dt.strftime('%Y-%m')
    
    logger.info(f"✓ Features adicionadas: {len(df.columns)} colunas agora")
    
    return df


def process_data(data_dir: Path = None) -> pd.DataFrame:
    """Pipeline completo de processamento da Fase 1."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    
    # 1. Carregar
    datasets = load_olist_data(data_dir)
    
    # 2. Unificar
    df = unify_tables(datasets)
    
    # 3. Limpar
    df = clean_data(df)
    
    # 4. Adicionar features
    df = add_features(df)
    
    logger.info("✓ Fase 1 completa!")
    
    return df


if __name__ == "__main__":
    df = process_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColunas: {df.columns.tolist()}")
