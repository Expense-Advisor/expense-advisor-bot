class EstimateSavings(object):
    def __init__(self, recurring_groups, profile):
        self.recurring_groups = recurring_groups
        self.profile = profile

    def estimate(self, **kwargs) -> float:
        return self._estimate_savings(self.recurring_groups, self.profile)

    def _estimate_savings(self, recurring_groups, profile) -> float:
        """
        Estimates potential savings based on personal spending behavior.
        Uses user's own baseline instead of hardcoded categories.
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
