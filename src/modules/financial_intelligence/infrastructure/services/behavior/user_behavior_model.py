from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class UserBehaviorModel(object):
    """
    Строитель поведенческой модели пользователя.

    Анализирует помесячные расходы по категориям и выявляет
    месяцы, в которых поведение пользователя значительно
    отклоняется от его обычного финансового стиля.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build(self) -> tuple[Any, list[str]]:
        """
        Строит профиль финансового поведения пользователя
        и генерирует текстовые объяснения аномальных месяцев.

        Returns:
            tuple[pd.DataFrame, list[str]]:
                - DataFrame с помесячным профилем и признаком аномальности
                - список текстовых рекомендаций
        """
        profile, _, _ = self.build_user_profile()
        return profile, self.explain_monthly_anomalies(profile)

    def build_user_profile(self) -> tuple[pd.DataFrame, IsolationForest, StandardScaler]:
        """
        Обучает ML-модель нормального финансового поведения пользователя.

        Метод агрегирует расходы по месяцам и категориям, после чего
        использует Isolation Forest для выявления месяцев,
        которые выбиваются из общего шаблона.

        Returns:
            tuple[pd.DataFrame, IsolationForest, StandardScaler]:
                - pivot-таблица (месяц × категория) с признаком `is_abnormal_month`
                - обученная модель IsolationForest
                - использованный StandardScaler
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
        """
        Формирует текстовые объяснения для аномальных месяцев.

        Сравнивает расходы в аномальных месяцах с нормальным
        базовым уровнем пользователя и выделяет категории,
        где перерасход был максимальным.

        Args:
            profile (pd.DataFrame): Помесячный профиль пользователя,
                полученный из `build_user_profile`.

        Returns:
            list[str]: Список текстовых рекомендаций и пояснений.
        """
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
