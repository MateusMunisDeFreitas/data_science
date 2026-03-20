import sys
from pathlib import Path

# Garantia de import local a partir da pasta src, caso execute do root do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from utils import create_preprocessing_pipeline, run_preprocessing, train_nlp_model


def demo_preprocessing() -> None:
    print("--- Pipeline 1: Pré-processamento de variáveis ---")
    path = ROOT_DIR / "data" / "sample_data.csv"
    df = pd.read_csv(path)

    numeric_features = ["age", "salary"]
    categorical_features = ["city"]

    _, df_processed = run_preprocessing(df, numeric_features, categorical_features)
    print(df_processed.head())
    print("Shape:", df_processed.shape)


def demo_nlp() -> None:
    print("\n--- Pipeline 2: NLP com TF-IDF ---")

    categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
    dataset = fetch_20newsgroups(subset="train", categories=categories, remove=("headers", "footers", "quotes"), random_state=42)

    # utilizar subset reduzido para demo rápida
    texts = dataset.data[:1500]
    targets = dataset.target[:1500]

    _, report = train_nlp_model(texts, targets, model_name="logistic")
    print(report)


if __name__ == "__main__":
    demo_preprocessing()
    demo_nlp()