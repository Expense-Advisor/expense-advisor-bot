import re

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.modules.bank_statement_analyzer.domain.interfaces.pipeline_item import PipelineItem


class SearchForRegularExpenses(PipelineItem):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> pd.DataFrame:
        return self._detect_recurring_payments(self.df)

    def _filter_real_expenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Оставляем только реальные траты:
        - не переводы
        - не пополнения
        - только списания
        """
        return df[
            (~df["is_money"]) &
            (df["amount"] < 0)
            ].copy()

    def _normalize_description(self, text: str) -> str:
        """
        Removes noise from bank descriptions to make clustering work.
        """
        text = text.lower()

        # убрать цифры, номера транзакций, телефоны
        text = re.sub(r"\d+", " ", text)

        # убрать мусор
        text = re.sub(r"[^\w\s]", " ", text)

        # убрать лишние пробелы
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _build_merchant_id(self, row: pd.Series) -> str:
        """
        Формирует устойчивый идентификатор получателя платежа
        """
        base = self._normalize_description(row["description"])
        mcc = str(row["mcc"]) if not pd.isna(row["mcc"]) else ""
        return f"{base}|{mcc}"

    def _make_time_features(self, dates: list[pd.Timestamp], amounts: list[float]) -> np.ndarray:
        """
        Строит ML-признаки регулярности платежей
        """
        dates = sorted(dates)
        days = [d.toordinal() for d in dates]

        deltas = np.diff(days)

        return np.array([
            np.mean(deltas),
            np.std(deltas),
            np.mean(amounts),
            np.std(amounts)
        ])

    def _detect_recurring_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ML-детектор регулярных платежей (подписок, сервисов, связи и т.д.)
        """
        df = self._filter_real_expenses(df)

        df["merchant_id"] = df.apply(self._build_merchant_id, axis=1)

        groups = (
            df.groupby("merchant_id")
            .agg(
                description=("description", lambda x: x.iloc[0][:60]),
                dates=("date", list),
                amounts=("amount", list),
                count=("amount", "count"),
                total=("amount", "sum")
            )
            .reset_index()
        )

        groups = groups[groups["count"] >= 3]

        if len(groups) == 0:
            return groups

        # строим ML-вектора
        features = np.vstack([
            self._make_time_features(row["dates"], row["amounts"])
            for _, row in groups.iterrows()
        ])

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        model = DBSCAN(eps=0.9, min_samples=1)
        labels = model.fit_predict(X)

        groups["cluster"] = labels
        groups["features"] = list(features)

        # строгий фильтр подписок
        def is_recurring(row):
            dates = pd.to_datetime(row["dates"])

            months = dates.to_period("M")

            n_months = months.nunique()
            n_payments = len(dates)

            # сколько месяцев в покрываемом интервале
            span = (months.max() - months.min()).n + 1

            coverage = n_months / span

            # стабильность суммы (но мягко)
            mean_amt = np.mean(row["amounts"])
            std_amt = np.std(row["amounts"])

            return (
                    n_months >= 3 and  # минимум 3 месяца
                    coverage >= 0.5 and  # платили хотя бы в половине месяцев
                    abs(std_amt / mean_amt) < 0.7  # цена может расти
            )

        groups["is_recurring"] = groups.apply(is_recurring, axis=1)

        return groups[groups["is_recurring"]]
