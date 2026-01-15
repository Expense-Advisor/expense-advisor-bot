import pandas as pd

from src.modules.financial_intelligence.infrastructure.services.behavior.anomaly_detector import SpendingAnomalyDetector
from src.modules.financial_intelligence.infrastructure.services.behavior.recurring_payment_detector import \
    RecurringPaymentDetector
from src.modules.financial_intelligence.infrastructure.services.behavior.user_behavior_model import UserBehaviorModel
from src.modules.financial_intelligence.infrastructure.services.categorization.semantic_classifier import \
    TransactionCategorizer
from src.modules.financial_intelligence.infrastructure.services.categorization.transaction_classifier import \
    OtherTransactionClassifier
from src.modules.financial_intelligence.infrastructure.services.ingestion.bank_statement_loader import \
    BankStatementIngestor
from src.modules.financial_intelligence.infrastructure.services.optimization.savings_estimator import \
    SavingsOpportunityEstimator

pd.set_option("display.max_colwidth", None)


class FinancialIntelligencePipeline(object):
    """
    Главный ML-пайплайн анализа банковских транзакций.

    Выполняет полный цикл финансового анализа:
        1. Загрузка и очистка банковской выписки
        2. Определение реальных категорий трат (банк + MCC + NLP)
        3. Поиск регулярных платежей (подписки, связь, сервисы)
        4. Выявление аномальных транзакций
        5. Построение модели финансовых привычек пользователя
        6. Расчёт потенциальной экономии
        7. Формирование текстового отчёта

    Attributes:
        content (bytes): Контент excel-файла банковской выписки.
    """

    def __init__(self, content: bytes):
        self.content: bytes = content

        self.bank_statement_loader = BankStatementIngestor(self.content)

        self.smart_category: TransactionCategorizer | None = None
        self.classification_other_operation: OtherTransactionClassifier | None = None
        self.search_for_regular_expenses: RecurringPaymentDetector | None = None
        self.anomalies_spending: SpendingAnomalyDetector | None = None

        self.build_user_profile: UserBehaviorModel | None = None
        self.estimation_savings: SavingsOpportunityEstimator | None = None

    def run(self) -> list[str]:
        """
        Запускает полный анализ банковских транзакций.

        Выполняет все этапы ML-пайплайна: загрузку данных, категоризацию,
        поиск подписок, аномалий, анализ поведения и расчёт экономии.

        Returns:
            list[str]: Сформированный текстовый финансовый отчёт для пользователя.
        """
        df: pd.DataFrame = self.bank_statement_loader.run()

        self.smart_category = TransactionCategorizer(df)
        df: pd.DataFrame = self.smart_category.run()

        self.classification_other_operation = OtherTransactionClassifier(df)
        df: pd.DataFrame = self.classification_other_operation.run()

        self.search_for_regular_expenses = RecurringPaymentDetector(df)
        recurring_groups: pd.DataFrame = self.search_for_regular_expenses.run()

        self.anomalies_spending = SpendingAnomalyDetector(df)
        df: pd.DataFrame = self.anomalies_spending.run()
        anomalies = df[df["anomaly"] == 1]

        self.build_user_profile = UserBehaviorModel(df)
        profile, profile_advice = self.build_user_profile.build()

        self.estimation_savings = SavingsOpportunityEstimator(recurring_groups, profile)
        savings = self.estimation_savings.estimate()

        return self._format_user_report(
            df,
            recurring_groups,
            anomalies,
            savings,
            profile_advice
        )

    def _format_user_report(
            self,
            df: pd.DataFrame,
            recurring_groups: pd.DataFrame,
            anomalies: pd.DataFrame,
            savings: float,
            profile_advice: list[str]
    ) -> list[str]:
        """
        Формирует итоговый текстовый финансовый отчёт.

        Args:
            df (pd.DataFrame): Полная таблица транзакций с финальными категориями.
            recurring_groups (pd.DataFrame): Обнаруженные регулярные платежи.
            anomalies (pd.DataFrame): Таблица аномальных операций.
            savings (float): Оценка потенциальной экономии.
            profile_advice (list[str]): Рекомендации на основе поведенческой модели.

        Returns:
            str: Готовый отчёт для вывода пользователю.
        """
        pages: list[str] = []

        # Куда уходят деньги
        block = ["<b>КУДА УХОДЯТ ДЕНЬГИ</b>\n"]

        by_cat = (
            df.groupby("final_category")["amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
        )

        total = by_cat.sum()

        for cat, value in by_cat.items():
            share = value / total * 100
            block.append(f"- {cat}: {value:,.0f} ₽ ({share:.1f}%)")

        pages.append("\n".join(block))

        # Регулярные платежи
        block = ["<b>ВАШИ РЕГУЛЯРНЫЕ ПЛАТЕЖИ</b>\n"]

        if len(recurring_groups) == 0:
            block.append("Регулярных платежей не найдено.")
        else:
            for _, row in recurring_groups.sort_values("total").iterrows():
                avg = abs(row["total"]) / row["count"]
                block.append(
                    f"- {row['description']} → {row['count']} раз, "
                    f"≈ {avg:.0f} ₽, всего {abs(row['total']):,.0f} ₽"
                )

        pages.append("\n".join(block))

        # Аномалии
        block = ["<b>НЕОБЫЧНЫЕ ТРАТЫ</b>\n"]

        if len(anomalies) == 0:
            block.append("Аномальных операций не обнаружено.")
        else:
            for _, row in anomalies.sort_values("amount").head(10).iterrows():
                desc = str(row["description"])
                block.append(
                    f"- {row['date'].date()} | {desc} → {row['amount']} ₽"
                )

        pages.append("\n".join(block))

        # Анализ финансов
        block = ["<b>АНАЛИЗ ФИНАНСОВОГО ПОВЕДЕНИЯ</b>\n"]

        for line in profile_advice:
            block.append(f"- {line}")

        pages.append("\n".join(block))

        # Экономия
        block = [
            "<b>ПОТЕНЦИАЛ ЭКОНОМИИ</b>\n",
            f"Если оптимизировать выявленные привычки, можно сохранить около {abs(savings):,.0f} ₽ за этот период."
        ]

        pages.append("\n".join(block))

        return pages
