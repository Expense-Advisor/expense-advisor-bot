import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.modules.financial_intelligence.domain.entities.category import (
    CATEGORY_PROTOTYPES,
    FINANCE_KEYWORDS,
)
from src.modules.financial_intelligence.domain.entities.mcc import MCCHelper
from src.modules.financial_intelligence.domain.interfaces.pipeline_item import (
    PipelineItem,
)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def is_money_movement(desc: str) -> bool:
    desc = str(desc).lower()
    return any(k in desc for k in FINANCE_KEYWORDS)


def embed(texts: str | list[str] | np.ndarray) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True)


class TransactionCategorizer(PipelineItem):
    SEMANTIC_CONFIDENCE_THRESHOLD = 0.42
    TRUSTED_BANK_CATEGORIES = {"Прочие операции", "Другое", "", "nan", "None"}

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> pd.DataFrame:
        self.smart_category()
        return self.df

    @staticmethod
    def _normalize_semantic_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def build_category_vectors(
        cls, extra_labels: list[str] | None = None
    ) -> tuple[list[str], np.ndarray]:
        labels: list[str] = list(CATEGORY_PROTOTYPES.keys())

        if extra_labels:
            for label in extra_labels:
                label = str(label).strip()
                if (
                    label
                    and label not in cls.TRUSTED_BANK_CATEGORIES
                    and label not in labels
                ):
                    labels.append(label)

        texts: list[str] = []
        for label in labels:
            if label in CATEGORY_PROTOTYPES:
                texts.append(CATEGORY_PROTOTYPES[label])
            else:
                texts.append(cls._normalize_semantic_text(label))

        vecs = embed(texts)
        return labels, vecs

    def _semantic_classify(self, df: pd.DataFrame) -> pd.DataFrame:
        extra_labels = (
            df["category"].dropna().astype(str).str.strip().unique().tolist()
            if "category" in df.columns
            else []
        )

        labels, cat_vecs = self.build_category_vectors(extra_labels=extra_labels)

        semantic_text = (
            df["description"]
            .fillna("")
            .astype(str)
            .apply(self._normalize_semantic_text)
        )

        desc_vecs = embed(semantic_text.tolist())
        sims = desc_vecs @ cat_vecs.T

        best_idx = sims.argmax(axis=1)
        best_score = sims.max(axis=1)

        df["semantic_category"] = [labels[i] for i in best_idx]
        df["semantic_score"] = best_score

        low_conf_mask = df["semantic_score"] < self.SEMANTIC_CONFIDENCE_THRESHOLD
        df.loc[low_conf_mask, "semantic_category"] = "Прочие операции"

        return df

    def smart_category(self) -> pd.DataFrame:
        self.df["is_money"] = (
            self.df["description"].astype(str).apply(is_money_movement)
        )
        self.df["mcc_category"] = self.df["mcc"].apply(MCCHelper.classify_by_mcc)

        self.df = self._semantic_classify(self.df)

        bank_category = (
            self.df["category"].fillna("Прочие операции").astype(str).str.strip()
        )

        trusted_bank_mask = ~bank_category.isin(self.TRUSTED_BANK_CATEGORIES)

        self.df["final_category"] = np.where(
            self.df["is_money"],
            "Финансовые операции",
            np.where(
                trusted_bank_mask,
                bank_category,
                np.where(
                    self.df["mcc_category"].notna(),
                    self.df["mcc_category"],
                    np.where(
                        self.df["semantic_score"] >= self.SEMANTIC_CONFIDENCE_THRESHOLD,
                        self.df["semantic_category"],
                        "Прочие операции",
                    ),
                ),
            ),
        )

        return self.df
