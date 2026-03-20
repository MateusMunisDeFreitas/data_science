"""
Fase 2: Engenharia de Features e NLP
- Pré-processamento com OneHotEncoder e Scaler
- TF-IDF nos reviews para análise de sentimento
- Identificar termos positivos/negativos
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Realizaprocessamento e engenharia de features."""
    
    def __init__(self):
        self.preprocessor = None
        self.scaler = None
        self.encoder = None
        
    def create_preprocessing_pipeline(self, numeric_features: List[str], 
                                     categorical_features: List[str]):
        """Cria pipeline de pré-processamento."""
        logger.info("Criando pipeline de pré-processamento...")
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor
    
    def fit_transform(self, X: pd.DataFrame, numeric_features: List[str],
                     categorical_features: List[str]) -> np.ndarray:
        """Ajusta e aplica transformações."""
        self.create_preprocessing_pipeline(numeric_features, categorical_features)
        return self.preprocessor.fit_transform(X)


class NLPAnalyzer:
    """Análise de NLP em reviews usando TF-IDF."""
    
    def __init__(self, max_features: int = 100, ngram_range: Tuple = (1, 2)):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        self.lda = None
        self.tfidf_matrix = None
        self.feature_names = None
    
    def fit_tfidf(self, texts: List[str]) -> np.ndarray:
        """Aplica TF-IDF nos textos."""
        logger.info(f"Aplicando TF-IDF em {len(texts)} documentos...")
        
        # Remover NaN
        texts_clean = [str(t) for t in texts if pd.notna(t)]
        
        self.tfidf_matrix = self.tfidf.fit_transform(texts_clean)
        self.feature_names = self.tfidf.get_feature_names_out()
        
        logger.info(f"✓ TF-IDF: {self.tfidf_matrix.shape}")
        
        return self.tfidf_matrix
    
    def get_top_terms(self, sentiment: str = None, df: pd.DataFrame = None,
                     top_n: int = 15) -> Dict[str, List[Tuple]]:
        """Identifica termos top por sentimento."""
        if df is None or sentiment is None:
            return self._get_global_top_terms(top_n)
        
        logger.info(f"Obtendo top termos para sentimento: {sentiment}")
        
        try:
            # Filtrar índices por sentimento
            sentiment_mask = df['review_sentiment'] == sentiment
            sentiment_indices = np.where(sentiment_mask.values)[0]
            
            if len(sentiment_indices) == 0:
                logger.warning(f"Nenhum documento com sentimento: {sentiment}")
                return {}
            
            # Somar TF-IDF scores para este sentimento
            tfidf_slice = self.tfidf_matrix[sentiment_indices]
            scores = np.asarray(tfidf_slice.sum(axis=0)).flatten()
            
            top_indices = np.argsort(scores)[-top_n:][::-1]
            
            top_terms = [
                (self.feature_names[idx], float(scores[idx]))
                for idx in top_indices
            ]
            
            return {sentiment: top_terms}
        
        except Exception as e:
            logger.warning(f"Erro ao extrair termos para {sentiment}: {e}")
            return {}
    
    def _get_global_top_terms(self, top_n: int = 15) -> Dict[str, List[Tuple]]:
        """Overview geral dos top termos."""
        scores = np.asarray(self.tfidf_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(scores)[-top_n:][::-1]
        
        top_terms = [
            (self.feature_names[idx], scores[idx])
            for idx in top_indices
        ]
        
        return {'top_termos': top_terms}


def process_features_and_nlp(df: pd.DataFrame) -> Tuple[pd.DataFrame, NLPAnalyzer]:
    """Pipeline completo da Fase 2."""
    logger.info("=== Iniciando Fase 2: Engenharia e NLP ===")
    
    # 1. Engenharia de features
    feature_engineer = FeatureEngineer()
    
    numeric_features = ['price', 'freight_value', 'delivery_time_days']
    categorical_features = ['order_status', 'product_category_name', 'payment_type']
    
    # Selecionar apenas colunas existentes
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    if numeric_features and categorical_features:
        try:
            X_processed = feature_engineer.fit_transform(
                df[numeric_features + categorical_features],
                numeric_features,
                categorical_features
            )
            logger.info(f"✓ Features processadas: {X_processed.shape}")
        except Exception as e:
            logger.warning(f"Erro ao processar features: {e}")
    
    # 2. NLP - TF-IDF nos reviews
    nlp_analyzer = NLPAnalyzer(max_features=100)
    
    # Filtrar reviews válidos
    reviews = df['review_comment_message'].dropna()
    if len(reviews) > 0:
        nlp_analyzer.fit_tfidf(reviews.tolist())
        
        # Análise por sentimento
        if 'review_sentiment' in df.columns:
            for sentiment in df['review_sentiment'].unique():
                if pd.notna(sentiment):
                    terms = nlp_analyzer.get_top_terms(
                        sentiment=sentiment,
                        df=df,
                        top_n=10
                    )
                    logger.info(f"  Top 10 termos para '{sentiment}':")
                    for term, score in terms.get(sentiment, []):
                        logger.info(f"    - {term}: {score:.4f}")
    
    logger.info("✓ Fase 2 completa!")
    
    return df, nlp_analyzer


if __name__ == "__main__":
    from phase1_data_processing import process_data
    
    df = process_data()
    df_processed, nlp = process_features_and_nlp(df)
    
    print("\n=== Resumo NLP ===")
    top_terms = nlp.get_top_terms()
    for term, score in top_terms.get('top_termos', [])[:10]:
        print(f"{term}: {score:.4f}")
