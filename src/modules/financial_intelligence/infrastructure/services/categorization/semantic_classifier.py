import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.modules.financial_intelligence.domain.entities.category import CATEGORY_PROTOTYPES, FINANCE_KEYWORDS
from src.modules.financial_intelligence.domain.entities.mcc import MCCHelper
from src.modules.financial_intelligence.domain.interfaces.pipeline_item import PipelineItem

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def is_money_movement(desc: str) -> bool:
    """
    Определяет, является ли операция финансовым переводом

    Args:
        desc (str): Описание транзакции

    Returns:
        bool: True, если операция является переводом или пополнением
    """
    desc = desc.lower()
    return any(k in desc for k in FINANCE_KEYWORDS)


def embed(texts: str | list[str] | np.ndarray) -> np.ndarray:
    """
    Преобразует тексты в векторные эмбеддинги

    Args:
        texts (Sequence[str]): Список текстов

    Returns:
        np.ndarray: Матрица эмбеддингов размером (n_texts, embedding_dim)
    """
    return model.encode(texts, normalize_embeddings=True)


class TransactionCategorizer(PipelineItem):
    """
    Интеллектуальный классификатор категорий транзакций.

    Определяет финальную категорию каждой операции, комбинируя:
        - категории банка
        - MCC-коды
        - семантический анализ текста (NLP)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> pd.DataFrame:
        """
        Запускает интеллектуальную категоризацию

        Returns:
            pd.DataFrame: Таблица транзакций с колонкой `final_category`
        """
        self.smart_category()
        return self.df

    @staticmethod
    def build_category_vectors() -> tuple[list[str], np.ndarray]:
        """
        Строит эмбеддинги для прототипов категорий

        Returns:
            tuple[list[str], np.ndarray]:
                - список названий категорий
                - матрица эмбеддингов этих категорий
        """
        labels = list(CATEGORY_PROTOTYPES.keys())
        texts = list(CATEGORY_PROTOTYPES.values())
        vecs = embed(texts)

        return labels, vecs

    def _semantic_classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Присваивает семантические категории на основе NLP

        Args:
            df (pd.DataFrame): Таблица транзакций

        Returns:
            pd.DataFrame: Таблица с колонками `semantic_category` и `semantic_score`
        """
        labels, cat_vecs = self.build_category_vectors()

        desc_vecs = embed(df["description"].astype(str).tolist())

        sims = desc_vecs @ cat_vecs.T  # косинусное сходство

        best = sims.argmax(axis=1)

        df["semantic_category"] = [labels[i] for i in best]
        df["semantic_score"] = sims.max(axis=1)

        return df

    def smart_category(self) -> pd.DataFrame:
        """
        Формирует финальную категорию каждой транзакции.

        Логика:
            1. Финансовые переводы → "Финансовые операции"
            2. Если банк дал категорию → используем её
            3. Если есть MCC → используем категорию по MCC
            4. Иначе → используем NLP-классификацию

        Returns:
            pd.DataFrame: Таблица транзакций с колонкой `final_category`
        """
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
