from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class BuildUserProfile(object):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build(self) -> tuple[Any, list[str]]:
        profile, _, _ = self.build_user_profile()
        return profile, self.explain_monthly_anomalies(profile)

    def build_user_profile(self):
        """
        Learns user's normal monthly spending behavior.
        """
        self.df["month"] = self.df["date"].dt.to_period("M")

        pivot = (
            self.df.groupby(["month", "final_category"])["amount"]
            .sum()
            .unstack(fill_value=0)
        )

        # приводим к абсолютным тратам
        pivot = pivot.abs()

        scaler = StandardScaler()
        X = scaler.fit_transform(pivot)

        model = IsolationForest(contamination=0.2, random_state=42)
        scores = model.fit_predict(X)

        pivot["is_abnormal_month"] = scores == -1

        return pivot, model, scaler

    def explain_monthly_anomalies(self, profile: pd.DataFrame) -> list[str]:
        advice: list[str] = []

        normal = profile[profile["is_abnormal_month"] == False]
        abnormal = profile[profile["is_abnormal_month"] == True]

        if len(abnormal) == 0:
            advice.append("Ваш стиль расходов стабилен — резких сбоев не обнаружено.")
            return advice

        baseline = normal.mean()

        for month, row in abnormal.iterrows():
            diff = row.drop("is_abnormal_month") - baseline

            top = diff.sort_values(ascending=False).head(3)

            for cat, value in top.items():
                if value > 0:
                    advice.append(
                        f"В {month} траты по категории '{cat}' были выше нормы на {value:.0f} ₽. "
                        "Это ключевая точка для оптимизации."
                    )

        return advice
