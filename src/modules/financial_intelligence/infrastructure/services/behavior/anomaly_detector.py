import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.modules.financial_intelligence.domain.interfaces.pipeline_item import (
    PipelineItem,
)


class SpendingAnomalyDetector(PipelineItem):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> pd.DataFrame:
        return self._detect_anomalies()

    @staticmethod
    def _safe_mad(values: pd.Series) -> float:
        arr = np.asarray(values, dtype=float)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return float(mad) if mad > 1e-9 else 1.0

    @staticmethod
    def _normalize_merchant(series: pd.Series) -> pd.Series:
        return (
            series.fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"\d+", " ", regex=True)
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def _detect_anomalies(self) -> pd.DataFrame:
        df = self.df.copy()

        if "is_money" in df.columns:
            is_money = df["is_money"].fillna(False)
        else:
            is_money = pd.Series(False, index=df.index)

        df["anomaly"] = 0
        df["anomaly_score"] = 0.0

        work_mask = (df["amount"] < 0) & (~is_money)
        work = df.loc[work_mask].copy()

        if len(work) < 20:
            self.df = df
            return self.df

        work["amount_abs"] = work["amount"].abs()
        work["merchant_norm"] = self._normalize_merchant(work["description"])
        work["merchant_freq"] = work["merchant_norm"].map(
            work["merchant_norm"].value_counts()
        )
        work["category_freq"] = work["final_category"].map(
            work["final_category"].value_counts()
        )
        work["day_of_week"] = work["date"].dt.dayofweek
        work["day_of_month"] = work["date"].dt.day

        work["cat_median"] = work.groupby("final_category")["amount_abs"].transform(
            "median"
        )
        work["cat_p90"] = work.groupby("final_category")["amount_abs"].transform(
            lambda s: s.quantile(0.90)
        )
        work["cat_p95"] = work.groupby("final_category")["amount_abs"].transform(
            lambda s: s.quantile(0.95)
        )
        work["cat_mad"] = work.groupby("final_category")["amount_abs"].transform(
            self._safe_mad
        )

        work["robust_z"] = (
            0.6745 * (work["amount_abs"] - work["cat_median"]) / work["cat_mad"]
        )
        work["robust_z"] = work["robust_z"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        features = pd.DataFrame(
            {
                "log_amount_abs": np.log1p(work["amount_abs"]),
                "merchant_freq": work["merchant_freq"].astype(float),
                "category_freq": work["category_freq"].astype(float),
                "day_of_week": work["day_of_week"].astype(float),
                "day_of_month": work["day_of_month"].astype(float),
                "robust_z": work["robust_z"].astype(float),
            }
        )

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        contamination = min(0.08, max(0.02, 10 / len(work)))
        model = IsolationForest(contamination=contamination, random_state=42)
        work["iforest_flag"] = (model.fit_predict(X) == -1).astype(int)

        work["rule_flag"] = (
            (work["robust_z"] >= 3.5)
            | ((work["merchant_freq"] <= 1) & (work["amount_abs"] >= work["cat_p95"]))
            | ((work["iforest_flag"] == 1) & (work["amount_abs"] >= work["cat_p90"]))
        ).astype(int)

        work["anomaly_score"] = (
            work["robust_z"].clip(lower=0.0)
            + 1.5 * work["iforest_flag"]
            + 1.0 * (work["merchant_freq"] <= 1).astype(int)
        )

        df.loc[work.index, "anomaly"] = work["rule_flag"]
        df.loc[work.index, "anomaly_score"] = work["anomaly_score"].round(3)

        self.df = df
        return self.df
