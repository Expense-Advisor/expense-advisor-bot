import numpy as np
import pandas as pd


class UserBehaviorModel(object):
    MIN_MONTHS = 4

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build(self) -> tuple[pd.DataFrame, list[str]]:
        profile, _, _ = self.build_user_profile()
        return profile, self.explain_monthly_anomalies(profile)

    @staticmethod
    def _safe_mad(series: pd.Series) -> float:
        arr = np.asarray(series, dtype=float)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return float(mad) if mad > 1e-9 else 1.0

    def build_user_profile(
        self,
    ) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
        df = self.df.copy()

        if "is_money" in df.columns:
            df = df[~df["is_money"]].copy()

        df = df[df["amount"] < 0].copy()

        df["month"] = df["date"].dt.to_period("M")

        pivot = (
            df.groupby(["month", "final_category"])["amount"]
            .sum()
            .unstack(fill_value=0)
            .abs()
        )

        if len(pivot) < self.MIN_MONTHS:
            pivot["month_anomaly_score"] = 0.0
            pivot["is_abnormal_month"] = False
            return pivot, None, None

        baseline = pivot.median(axis=0)
        mad = pivot.apply(self._safe_mad, axis=0)

        robust_z = 0.6745 * (pivot - baseline) / mad
        robust_z = robust_z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        positive_z = robust_z.clip(lower=0.0)

        if positive_z.shape[1] >= 2:
            scores = np.sort(positive_z.values, axis=1)[:, -2:].mean(axis=1)
        else:
            scores = positive_z.values[:, 0]

        month_score = pd.Series(scores, index=pivot.index, name="month_anomaly_score")
        threshold = max(2.5, float(month_score.quantile(0.80)))

        pivot["month_anomaly_score"] = month_score
        pivot["is_abnormal_month"] = month_score >= threshold

        return pivot, baseline, mad

    def explain_monthly_anomalies(self, profile: pd.DataFrame) -> list[str]:
        advice: list[str] = []

        if profile.empty:
            return ["Недостаточно данных для анализа финансового поведения."]

        abnormal = profile[profile["is_abnormal_month"] == True]
        normal = profile[profile["is_abnormal_month"] == False]

        if len(abnormal) == 0:
            return ["Ваш стиль расходов стабилен — резких сбоев не обнаружено."]

        value_cols = [
            col
            for col in profile.columns
            if col not in {"is_abnormal_month", "month_anomaly_score"}
        ]

        if len(normal) > 0:
            baseline = normal[value_cols].median()
        else:
            baseline = profile[value_cols].median()

        for month, row in abnormal.iterrows():
            diff = row[value_cols] - baseline
            top = diff.sort_values(ascending=False).head(3)

            for cat, value in top.items():
                if value > 0:
                    advice.append(
                        f"В {month} траты по категории '{cat}' были выше вашего обычного уровня на {value:.0f} ₽. "
                        f"Это одна из главных точек для оптимизации."
                    )

        return advice
