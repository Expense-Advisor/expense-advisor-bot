import pandas as pd


class EstimateSavings(object):
    """
    Оценщик потенциальной экономии пользователя.

    Анализирует:
        - перерасход в аномальных месяцах
        - сумму регулярных платежей (подписок)

    и вычисляет, сколько денег пользователь мог бы сохранить,
    если оптимизировал свои финансовые привычки.
    """

    def __init__(self, recurring_groups, profile):
        self.recurring_groups = recurring_groups
        self.profile = profile

    def estimate(self, **kwargs) -> float:
        """
        Запускает расчёт потенциальной экономии

        Returns:
            float: Оценка возможной экономии за анализируемый период
        """
        return self._estimate_savings(self.recurring_groups, self.profile)

    def _estimate_savings(
            self,
            recurring_groups: pd.DataFrame,
            profile: pd.DataFrame
    ) -> float:
        """
        Выполняет расчёт экономии на основе финансового поведения пользователя.

        Алгоритм:
            1. Вычисляет нормальный (базовый) уровень расходов пользователя
            2. Считает перерасход в аномальных месяцах
            3. Добавляет потенциальную экономию от отключения подписок

        Args:
            recurring_groups (pd.DataFrame): Таблица регулярных платежей.
            profile (pd.DataFrame): Помесячный профиль пользователя.

        Returns:
            float: Итоговая оценка возможной экономии.
        """
        savings = 0.0

        normal = profile[profile["is_abnormal_month"] == False]
        abnormal = profile[profile["is_abnormal_month"] == True]

        if len(normal) == 0 or len(abnormal) == 0:
            return 0.0

        baseline = normal.drop(columns=["is_abnormal_month"]).mean()

        # 1️⃣ Перерасход по месяцам
        for month, row in abnormal.iterrows():
            diff = row.drop("is_abnormal_month") - baseline

            for value in diff:
                if value > 0:
                    # считаем, что 50% перерасхода можно оптимизировать
                    savings += value * 0.5

        # 2️⃣ Регулярные платежи (подписки и сервисы)
        if len(recurring_groups) > 0:
            # считаем, что 60% подписок можно отключить
            savings += abs(recurring_groups["total"].sum()) * 0.6

        return round(float(savings), 2)
