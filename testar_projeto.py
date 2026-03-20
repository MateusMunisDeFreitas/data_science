#!/usr/bin/env python
"""
Script de teste para validar o projeto completo
Executa as 3 fases e mostra resumo dos resultados
"""

import sys
from pathlib import Path

# Adicionar src ao path
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def test_project():
    """Testa o projeto inteiro."""
    
    print("\n" + "="*60)
    print("🧪 TESTE DO PROJETO - CIÊNCIA DE DADOS OLIST")
    print("="*60 + "\n")
    
    # ===== FASE 1 =====
    print("📝 Testando Fase 1: Manipulação e Visualização...")
    try:
        from phase1_data_processing import process_data
        df = process_data()
        print(f"✅ Fase 1 OK")
        print(f"   - Dados: {df.shape[0]} linhas × {df.shape[1]} colunas")
        print(f"   - Período: {df['order_purchase_timestamp'].min()} a {df['order_purchase_timestamp'].max()}")
    except Exception as e:
        print(f"❌ Erro Fase 1: {e}")
        return False
    
    # ===== FASE 2 =====
    print("\n📝 Testando Fase 2: Engenharia e NLP...")
    try:
        from phase2_nlp_engineering import process_features_and_nlp
        df, nlp = process_features_and_nlp(df)
        print(f"✅ Fase 2 OK")
        if nlp.feature_names is not None:
            print(f"   - Features TF-IDF: {len(nlp.feature_names)} termos")
        else:
            print(f"   - Reviews processados com TF-IDF")
    except Exception as e:
        print(f"❌ Erro Fase 2: {e}")
        return False
    
    # ===== FASE 3 =====
    print("\n📝 Testando Fase 3: Séries Temporais...")
    try:
        from phase3_time_series import process_time_series
        ts_analyzer, ts_results = process_time_series(df)
        print(f"✅ Fase 3 OK")
        if ts_results and 'metrics' in ts_results:
            metrics = ts_results['metrics']
            print(f"   - RMSE (teste): {metrics.get('rmse_test', 0):.2f}")
            print(f"   - MAE (teste): {metrics.get('mae_test', 0):.2f}")
        if ts_results and 'forecast' in ts_results:
            print(f"   - Previsões: 12 meses futuros gerados")
    except Exception as e:
        print(f"❌ Erro Fase 3: {e}")
        return False
    
    # ===== RESUMO =====
    print("\n" + "="*60)
    print("✅ PROJETO TESTADO COM SUCESSO!")
    print("="*60)
    print("\n🚀 Como executar:")
    print("   1. Dashboard Streamlit:")
    print("      python -m streamlit run app.py")
    print("\n   2. Fases individuais:")
    print("      python src/phase1_data_processing.py")
    print("      python src/phase2_nlp_engineering.py")
    print("      python src/phase3_time_series.py")
    print("\n" + "="*60 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_project()
    sys.exit(0 if success else 1)
