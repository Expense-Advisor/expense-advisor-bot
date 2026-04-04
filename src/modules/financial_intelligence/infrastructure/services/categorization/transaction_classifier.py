import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from src.modules.financial_intelligence.domain.interfaces.pipeline_item import (
    PipelineItem,
)


class OtherTransactionClassifier(PipelineItem):
    CONFIDENCE_THRESHOLD = 0.60
    MIN_TRAIN_ROWS = 30

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self, **kwargs) -> pd.DataFrame:
        artifacts = self.train_category_model(self.df)

        if artifacts is None:
            self.df["ml_category"] = None
            self.df["ml_confidence"] = 0.0
            return self.df

        return self.classify_other_operations(self.df, artifacts)

    @staticmethod
    def _normalize_text_series(series: pd.Series) -> pd.Series:
        return (
            series.fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"\d+", " ", regex=True)
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def _build_feature_matrix(
        self, df: pd.DataFrame, artifacts: dict | None = None, fit: bool = False
    ):
        text = self._normalize_text_series(df["description"])

        aux = pd.DataFrame(
            {
                "mcc": df["mcc"].fillna(-1).astype(int).astype(str),
                "mcc_category": df["mcc_category"].fillna("NO_MCC").astype(str),
                "direction": np.where(df["amount"] < 0, "expense", "income"),
            }
        )

        if fit:
            word_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                max_features=8000,
                sublinear_tf=True,
            )
            char_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=12000,
                sublinear_tf=True,
            )
            aux_encoder = OneHotEncoder(handle_unknown="ignore")

            X_word = word_vectorizer.fit_transform(text)
            X_char = char_vectorizer.fit_transform(text)
            X_aux = aux_encoder.fit_transform(aux)

            artifacts = {
                "word_vectorizer": word_vectorizer,
                "char_vectorizer": char_vectorizer,
                "aux_encoder": aux_encoder,
            }
        else:
            word_vectorizer = artifacts["word_vectorizer"]
            char_vectorizer = artifacts["char_vectorizer"]
            aux_encoder = artifacts["aux_encoder"]

            X_word = word_vectorizer.transform(text)
            X_char = char_vectorizer.transform(text)
            X_aux = aux_encoder.transform(aux)

        X = hstack([X_word, X_char, X_aux], format="csr")
        return X, artifacts

    def train_category_model(self, df: pd.DataFrame):
        train_mask = (
            (~df["is_money"])
            & df["final_category"].notna()
            & (df["final_category"] != "Прочие операции")
        )

        train_df = df.loc[train_mask].copy()

        if len(train_df) < self.MIN_TRAIN_ROWS:
            return None

        if train_df["final_category"].nunique() < 2:
            return None

        X_train, artifacts = self._build_feature_matrix(train_df, fit=True)
        y_train = train_df["final_category"].astype(str)

        model = LogisticRegression(max_iter=2000, class_weight="balanced", C=3.0)
        model.fit(X_train, y_train)

        artifacts["model"] = model
        return artifacts

    def classify_other_operations(self, df: pd.DataFrame, artifacts: dict):
        df = df.copy()

        mask = (
            (df["category"] == "Прочие операции")
            & (~df["is_money"])
            & (df["final_category"] == "Прочие операции")
        )

        df["ml_category"] = None
        df["ml_confidence"] = 0.0

        if not mask.any():
            return df

        target_df = df.loc[mask].copy()
        X_test, _ = self._build_feature_matrix(
            target_df, artifacts=artifacts, fit=False
        )

        model = artifacts["model"]
        proba = model.predict_proba(X_test)

        best_idx = proba.argmax(axis=1)
        best_score = proba.max(axis=1)
        best_pred = model.classes_[best_idx]

        df.loc[target_df.index, "ml_category"] = best_pred
        df.loc[target_df.index, "ml_confidence"] = best_score

        confident_mask = best_score >= self.CONFIDENCE_THRESHOLD
        confident_index = target_df.index[confident_mask]

        df.loc[confident_index, "final_category"] = best_pred[confident_mask]

        return df
