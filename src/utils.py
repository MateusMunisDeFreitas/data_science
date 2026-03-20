import logging 
from typing import List, Tuple, Union

import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """Cria um ColumnTransformer para pré-processamento.

    - Numéricos: imputação média + escalonamento (StandardScaler)
    - Categóricos: imputação de constante + OneHotEncoding
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ],
        memory=None,
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ],
        memory=None,
    )

    transformer = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return transformer


def run_preprocessing(
    df: Union[pd.DataFrame, any],
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """Executa fit + transform em um DataFrame de entrada."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df precisa ser um pandas.DataFrame")

    pipeline = create_preprocessing_pipeline(numeric_features, categorical_features)
    x_processed = pipeline.fit_transform(df)

    # Retornar DataFrame com colunas transformadas
    num_cols = numeric_features
    cat_cols = list(pipeline.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features))
    result = pd.DataFrame(x_processed, columns=num_cols + cat_cols, index=df.index)

    logger.info("Pré-processamento concluído: %s linhas / %s colunas", result.shape[0], result.shape[1])

    return pipeline, result


def create_nlp_pipeline(
    model_name: str = "logistic",
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: Union[str, None] = "english",
) -> Pipeline:
    """Cria pipeline de NLP com TF-IDF + classificador."""
    model_name = model_name.lower()
    if model_name == "logistic":
        clf = LogisticRegression(max_iter=1500, random_state=42)
    elif model_name == "svm":
        clf = LinearSVC(random_state=42, max_iter=5000)
    elif model_name == "rf" or model_name == "randomforest":
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=1,
            max_features="sqrt",
        )
    else:
        raise ValueError("model_name deve ser 'logistic', 'svm' ou 'rf'")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words=stop_words,
                ),
            ),
            ("clf", clf),
        ],
        memory=None,
    )

    return pipeline


def train_nlp_model(
    texts: List[str],
    targets: Union[List[int], pd.Series],
    model_name: str = "logistic",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, str]:
    """Treina o pipeline NLP e retorna o relatório de classificação."""
    if len(texts) == 0:
        raise ValueError("texts não pode estar vazio")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, targets, test_size=test_size, random_state=random_state, stratify=targets
    )

    pipeline = create_nlp_pipeline(model_name=model_name)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)

    logger.info("NLP model (%s) treinado. \n%s", model_name, report)
    return pipeline, report
