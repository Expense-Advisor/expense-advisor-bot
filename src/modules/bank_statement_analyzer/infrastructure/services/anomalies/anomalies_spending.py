import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.modules.bank_statement_analyzer.domain.interfaces.pipeline_item import PipelineItem


class AnomaliesSpendingAnalyzer(PipelineItem):
    """
    Детектор аномальных трат пользователя.

    Использует модель Isolation Forest для выявления операций,
    которые сильно выбиваются из общего распределения по сумме.
    Такие операции обычно соответствуют импульсивным, редким
    или необычно крупным покупкам.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> pd.DataFrame:
        """
        Запускает детекцию аномальных транзакций.

        Метод добавляет в таблицу колонку `anomaly`, где:
            - 1 — операция считается аномальной
            - 0 — операция находится в пределах нормы

        Returns:
            pd.DataFrame: DataFrame с добавленной колонкой `anomaly`.
        """
        return self._detect_anomalies()

    def _detect_anomalies(self) -> pd.DataFrame:
        """
        Выполняет ML-детекцию аномалий на основе суммы транзакции.

        Использует Isolation Forest, который ищет редкие и изолированные
        точки в распределении сумм операций.

        Returns:
            pd.DataFrame: Таблица транзакций с добавленным бинарным признаком `anomaly`.
        """
        features = self.df[["amount"]].copy()

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        model = IsolationForest(contamination=0.05, random_state=42)
        self.df["anomaly"] = model.fit_predict(X)

        # -1 means anomaly -> convert to 1
        self.df["anomaly"] = (self.df["anomaly"] == -1).astype(int)

        return self.df
