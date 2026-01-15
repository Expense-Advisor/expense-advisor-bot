import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.modules.bank_statement_analyzer.domain.interfaces.pipeline_item import PipelineItem


class ClassificationOtherOperation(PipelineItem):
    """
    ML-классификатор операций категории «Прочие операции».

    Обучает модель на уже известных категориях и использует её
    для автоматической переклассификации операций, которые банк
    не смог отнести ни к одной категории.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self, **kwargs) -> pd.DataFrame:
        """
        Запускает ML-переклассификацию «Прочих операций».

        Метод обучает текстовую модель на существующих категориях
        и применяет её для операций с категорией «Прочие операции».

        Args:
            **kwargs: Дополнительные параметры пайплайна (не используются).

        Returns:
            pd.DataFrame: Таблица транзакций с обновлённой колонкой `final_category`.
        """
        model, vec = self.train_category_model(self.df)
        return self.classify_other_operations(self.df, model, vec)

    def train_category_model(self, df: pd.DataFrame):
        """
        Обучает ML-модель для предсказания категорий по тексту описания.

        В качестве обучающей выборки используются все операции,
        у которых категория уже известна (кроме «Прочие операции»).

        Args:
            df (pd.DataFrame): Таблица транзакций.

        Returns:
            tuple[LogisticRegression, TfidfVectorizer]:
                - обученная логистическая регрессия
                - TF-IDF векторизатор текста
        """
        train_df = df[df["final_category"] != "Прочие операции"]

        X = train_df["description"].astype(str)
        y = train_df["final_category"]

        vectorizer = TfidfVectorizer(max_features=1000)
        X_vec = vectorizer.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_vec, y)

        return model, vectorizer

    def classify_other_operations(self, df: pd.DataFrame, model, vectorizer):
        """
        Переклассифицирует операции категории «Прочие операции» с помощью ML.

        Использует обученную модель для предсказания реальной категории
        по тексту описания транзакции.

        Args:
            df (pd.DataFrame): Таблица транзакций.
            model (LogisticRegression): Обученная модель классификации.
            vectorizer (TfidfVectorizer): TF-IDF векторизатор.

        Returns:
            pd.DataFrame: Таблица с обновлённой колонкой `final_category`.
        """
        mask = df["category"] == "Прочие операции"

        X = vectorizer.transform(df.loc[mask, "description"].astype(str))
        predictions = model.predict(X)

        df.loc[mask, "final_category"] = predictions
        df.loc[~mask, "final_category"] = df.loc[~mask, "category"]

        return df
