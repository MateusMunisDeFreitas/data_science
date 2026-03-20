"""
Fase 3: Séries Temporais e Previsão de Demanda
- Agrupar vendas por período
- Decomposição de séries temporais
- Modelo preditivo de demanda (ARIMA/Prophet)
- Métricas: RMSE, MAE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """Análise e previsão de séries temporais."""
    
    def __init__(self):
        self.sales_ts = None
        self.decomposed = None
        self.model = None
        self.forecast = None
        
    def prepare_time_series(self, df: pd.DataFrame, 
                           time_col: str = 'order_month_name',
                           value_col: str = 'order_total') -> pd.DataFrame:
        """Prepara série temporal agregada por período."""
        logger.info("Preparando série temporal...")
        
        # Agrupar por período (mês)
        ts_data = df.groupby(time_col)[value_col].agg(['sum', 'mean', 'count'])
        ts_data.columns = ['total_vendas', 'ticket_medio', 'num_transacoes']
        ts_data = ts_data.reset_index()
        ts_data[time_col] = pd.to_datetime(ts_data[time_col])
        ts_data = ts_data.sort_values(time_col)
        
        logger.info(f"✓ Série temporal: {len(ts_data)} períodos")
        logger.info(f"  Período: {ts_data[time_col].min()} a {ts_data[time_col].max()}")
        
        self.sales_ts = ts_data
        return ts_data
    
    def basic_decomposition(self) -> Dict:
        """Decomposição básica de tendência e sazonalidade."""
        if self.sales_ts is None:
            logger.error("Execute prepare_time_series primeiro")
            return {}
        
        logger.info("Realizando decomposição...")
        
        sales = self.sales_ts['total_vendas'].values
        
        # Tendência: média móvel com janela de 3 períodos
        trend = pd.Series(sales).rolling(window=3, center=True).mean().values
        
        # Sazonalidade: diferença entre valor real e tendência
        seasonality = sales - trend
        
        # Ruído: o restante
        noise = np.zeros_like(sales)
        
        decomposition = {
            'trend': trend,
            'seasonality': seasonality,
            'noise': noise,
            'original': sales
        }
        
        self.decomposed = decomposition
        
        logger.info("✓ Decomposição concluída")
        logger.info(f"  Trend range: [{trend[~np.isnan(trend)].min():.2f}, {trend[~np.isnan(trend)].max():.2f}]")
        logger.info(f"  Sazonalidade range: [{seasonality.min():.2f}, {seasonality.max():.2f}]")
        
        return decomposition
    
    def train_forecast_model(self, test_size: float = 0.2,
                           polynomial_degree: int = 2) -> Dict[str, float]:
        """Treina modelo preditivo de demanda."""
        if self.sales_ts is None:
            logger.error("Execute prepare_time_series primeiro")
            return {}
        
        logger.info("Treinando modelo de previsão...")
        
        # Preparar dados
        X = np.arange(len(self.sales_ts)).reshape(-1, 1)
        y = self.sales_ts['total_vendas'].values
        
        # Split teste/treino
        split_point = int(len(self.sales_ts) * (1 - test_size))
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Pipeline: Polinomial + LinearRegression
        model = LinearRegression()
        poly = PolynomialFeatures(degree=polynomial_degree)
        
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Treinar
        model.fit(X_train_poly, y_train)
        
        # Prever
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
        
        # Métricas
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        metrics = {
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'train_size': len(y_train),
            'test_size': len(y_test)
        }
        
        self.model = {
            'model': model,
            'poly': poly,
            'metrics': metrics,
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'X_test': X_test
        }
        
        logger.info("✓ Modelo treinado")
        logger.info(f"  RMSE (treino): {rmse_train:.2f}")
        logger.info(f"  RMSE (teste): {rmse_test:.2f}")
        logger.info(f"  MAE (teste): {mae_test:.2f}")
        
        return metrics
    
    def forecast_future(self, periods: int = 12) -> pd.DataFrame:
        """Gera previsões para períodos futuros."""
        if self.model is None:
            logger.error("Execute train_forecast_model primeiro")
            return pd.DataFrame()
        
        logger.info(f"Gerando previsões para {periods} períodos...")
        
        # Dados históricos para contexto
        last_date = self.sales_ts['order_month_name'].iloc[-1]
        last_idx = len(self.sales_ts)
        
        # Índices para previsão
        future_indices = np.arange(last_idx, last_idx + periods).reshape(-1, 1)
        future_dates = pd.date_range(
            start=last_date + timedelta(days=30),
            periods=periods,
            freq='MS'
        )
        
        # Prever
        future_X_poly = self.model['poly'].transform(future_indices)
        future_pred = self.model['model'].predict(future_X_poly)
        
        # Dataframe previsão
        forecast_df = pd.DataFrame({
            'data': future_dates,
            'vendas_previstas': future_pred,
            'intervalo_superior': future_pred * 1.15,
            'intervalo_inferior': future_pred * 0.85
        })
        
        self.forecast = forecast_df
        
        logger.info(f"✓ Previsões geradas")
        logger.info(f"  Intervalo: {forecast_df['vendas_previstas'].min():.2f} a {forecast_df['vendas_previstas'].max():.2f}")
        
        return forecast_df


def process_time_series(df: pd.DataFrame) -> Tuple[TimeSeriesAnalyzer, Dict]:
    """Pipeline completo da Fase 3."""
    logger.info("\n=== Iniciando Fase 3: Séries Temporais ===")
    
    # Validar coluna de data
    if 'order_month_name' not in df.columns:
        logger.error("Coluna 'order_month_name' não encontrada")
        return TimeSeriesAnalyzer(), {}
    
    # 1. Preparar série temporal
    ts_analyzer = TimeSeriesAnalyzer()
    ts_data = ts_analyzer.prepare_time_series(df)
    
    # 2. Decomposição
    decomposition = ts_analyzer.basic_decomposition()
    
    # 3. Treinar modelo
    metrics = ts_analyzer.train_forecast_model()
    
    # 4. Gerar previsões
    forecast = ts_analyzer.forecast_future(periods=12)
    
    logger.info("✓ Fase 3 completa!")
    
    return ts_analyzer, {
        'metrics': metrics,
        'forecast': forecast,
        'decomposition': decomposition
    }


if __name__ == "__main__":
    from phase1_data_processing import process_data
    
    df = process_data()
    ts_analyzer, results = process_time_series(df)
    
    print("\n=== Resumo Previsões ===")
    print(results['forecast'].head(12))
