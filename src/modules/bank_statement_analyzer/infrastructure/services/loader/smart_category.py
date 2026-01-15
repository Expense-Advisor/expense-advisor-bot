import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.modules.bank_statement_analyzer.domain.entites.category import CATEGORY_PROTOTYPES, FINANCE_KEYWORDS
from src.modules.bank_statement_analyzer.domain.entites.mcc import MCCHelper
from src.modules.bank_statement_analyzer.domain.interfaces.pipeline_item import PipelineItem

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def is_money_movement(desc: str) -> bool:
    desc = desc.lower()
    return any(k in desc for k in FINANCE_KEYWORDS)


def embed(texts):
    return model.encode(texts, normalize_embeddings=True)


class SmartCategory(PipelineItem):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self, **kwargs) -> pd.DataFrame:
        self.smart_category()
        return self.df

    @staticmethod
    def build_category_vectors():
        labels = list(CATEGORY_PROTOTYPES.keys())
        texts = list(CATEGORY_PROTOTYPES.values())
        vecs = embed(texts)

        return labels, vecs

    def _semantic_classify(self, df):
        labels, cat_vecs = self.build_category_vectors()

        desc_vecs = embed(df["description"].astype(str).tolist())

        sims = desc_vecs @ cat_vecs.T  # cosine similarity

        best = sims.argmax(axis=1)

        df["semantic_category"] = [labels[i] for i in best]
        df["semantic_score"] = sims.max(axis=1)

        return df

    def smart_category(self) -> pd.DataFrame:
        self.df["is_money"] = self.df["description"].astype(str).apply(is_money_movement)

        self.df["mcc_category"] = self.df["mcc"].apply(MCCHelper.classify_by_mcc)
        self.df = self._semantic_classify(self.df)

        self.df["final_category"] = np.where(
            self.df["is_money"],
            "Финансовые операции",  # ← выше всего
            np.where(
                self.df["category"] != "Прочие операции",
                self.df["category"],  # ← доверяем банку
                np.where(
                    self.df["mcc_category"].notna(),
                    self.df["mcc_category"],
                    self.df["semantic_category"]
                )
            )
        )