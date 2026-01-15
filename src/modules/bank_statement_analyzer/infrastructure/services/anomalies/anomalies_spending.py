import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.modules.bank_statement_analyzer.domain.interfaces.pipeline_item import PipelineItem


class AnomaliesSpendingAnalyzer(PipelineItem):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> pd.DataFrame:
        return self._detect_anomalies()

    def _detect_anomalies(self) -> pd.DataFrame:
        """
        Detects anomalous transactions using Isolation Forest.

        Args:
            df: Normalized dataframe.

        Returns:
            DataFrame with an 'anomaly' column (1 = anomaly).
        """
        features = self.df[["amount"]].copy()

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        model = IsolationForest(contamination=0.05, random_state=42)
        self.df["anomaly"] = model.fit_predict(X)

        # -1 means anomaly -> convert to 1
        self.df["anomaly"] = (self.df["anomaly"] == -1).astype(int)

        return self.df
