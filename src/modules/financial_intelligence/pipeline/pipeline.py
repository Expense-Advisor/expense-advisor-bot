import re
from html import escape

import pandas as pd

from src.modules.financial_intelligence.infrastructure.services.behavior.anomaly_detector import (
    SpendingAnomalyDetector,
)
from src.modules.financial_intelligence.infrastructure.services.behavior.recurring_payment_detector import (
    RecurringPaymentDetector,
)
from src.modules.financial_intelligence.infrastructure.services.behavior.user_behavior_model import (
    UserBehaviorModel,
)
from src.modules.financial_intelligence.infrastructure.services.categorization.semantic_classifier import (
    TransactionCategorizer,
)
from src.modules.financial_intelligence.infrastructure.services.categorization.transaction_classifier import (
    OtherTransactionClassifier,
)
from src.modules.financial_intelligence.infrastructure.services.ingestion.bank_statement_loader import (
    BankStatementIngestor,
)
from src.modules.financial_intelligence.infrastructure.services.optimization.savings_estimator import (
    SavingsOpportunityEstimator,
)

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

        print("Transaction categorizer")
        self.smart_category = TransactionCategorizer(df)
        df: pd.DataFrame = self.smart_category.run()

        print("Other transaction classifier")
        self.classification_other_operation = OtherTransactionClassifier(df)
        df: pd.DataFrame = self.classification_other_operation.run()

        print("Recurring payment detector")
        self.search_for_regular_expenses = RecurringPaymentDetector(df)
        recurring_groups: pd.DataFrame = self.search_for_regular_expenses.run()

        print("spending anomaly detector")
        self.anomalies_spending = SpendingAnomalyDetector(df)
        df: pd.DataFrame = self.anomalies_spending.run()
        anomalies = df[df["anomaly"] == 1]

        print("User behavior model")
        self.build_user_profile = UserBehaviorModel(df)
        profile, profile_advice = self.build_user_profile.build()

        print("Savings opportunity estimator")
        self.estimation_savings = SavingsOpportunityEstimator(recurring_groups, profile)
        savings = self.estimation_savings.estimate()

        return self._format_user_report(
            df, recurring_groups, anomalies, savings, profile_advice
        )

    def _format_user_report(
        self,
        df: pd.DataFrame,
        recurring_groups: pd.DataFrame,
        anomalies: pd.DataFrame,
        savings: float,
        profile_advice: list[str],
    ) -> list[str]:
        pages: list[str] = []

        # 1. Куда уходят деньги
        block = ["<b>КУДА УХОДЯТ ДЕНЬГИ</b>", ""]

        by_cat = (
            df.groupby("final_category")["amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
        )

        total = by_cat.sum()

        for i, (cat, value) in enumerate(by_cat.items(), start=1):
            share = value / total * 100 if total else 0
            block.append(
                f"{i}. {escape(str(cat))} — {self._format_money(value)} ₽ ({share:.1f}%)"
            )

        pages.append("\n".join(block))

        # 2. Регулярные платежи
        block = ["<b>ВАШИ РЕГУЛЯРНЫЕ ПЛАТЕЖИ</b>", ""]

        if len(recurring_groups) == 0:
            block.append("Регулярных платежей не найдено.")
        else:
            for i, (_, row) in enumerate(
                recurring_groups.sort_values("total").iterrows(),
                start=1,
            ):
                desc = self._simplify_description(row["description"])
                avg = abs(row["total"]) / row["count"]

                count = int(row["count"])
                block.append(
                    f"{i}. {desc} — "
                    f"{count} платеж., "
                    f"средний чек {self._format_money(avg)} ₽, "
                    f"всего {self._format_money(abs(row['total']))} ₽"
                )

        pages.append("\n".join(block))

        # 3. Необычные траты
        block = ["<b>НЕОБЫЧНЫЕ ТРАТЫ</b>", ""]

        if len(anomalies) == 0:
            block.append("Аномальных операций не обнаружено.")
        else:
            for i, (_, row) in enumerate(
                anomalies.sort_values(
                    ["anomaly_score", "amount"],
                    ascending=[False, True],
                ).iterrows(),
                start=1,
            ):
                desc = self._simplify_description(row["description"])
                block.append(
                    f"{i}. {row['date'].date()} — {desc} — "
                    f"{self._format_money(abs(row['amount']))} ₽"
                )

        pages.append("\n".join(block))

        # 4. Анализ финансового поведения
        block = ["<b>АНАЛИЗ ФИНАНСОВОГО ПОВЕДЕНИЯ</b>", ""]

        if not profile_advice:
            block.append("Выраженных отклонений от обычного поведения не найдено.")
        else:
            for i, line in enumerate(profile_advice, start=1):
                cleaned = self._clean_spaces(line)

                m = re.search(
                    r"В\s+([0-9]{4}-[0-9]{2})\s+траты по категории\s+'(.+?)'\s+были выше.*?на\s+([0-9]+)\s*₽",
                    cleaned,
                )

                if m:
                    month, category, value = m.groups()
                    month_label = self._format_month_label(month)
                    block.append(
                        f"{i}. {month_label} • {escape(category)} • "
                        f"+{self._format_money(float(value))} ₽ к обычному уровню"
                    )
                else:
                    block.append(f"{i}. {escape(cleaned)}")

        pages.append("\n".join(block))

        # 5. Потенциал экономии
        block = [
            "<b>ПОТЕНЦИАЛ ЭКОНОМИИ</b>",
            "",
            f"Оценка за период: {self._format_money(abs(savings))} ₽",
        ]

        pages.append("\n".join(block))

        return pages

    @staticmethod
    def _format_money(value: float, digits: int = 0) -> str:
        value = float(value)
        formatted = f"{value:,.{digits}f}".replace(",", " ")
        if digits == 0:
            formatted = formatted.split(".")[0]
        return formatted

    @staticmethod
    def _clean_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()

    def _simplify_description(self, description: str) -> str:
        """
        Упрощает длинное описание операции, не теряя данные.
        Было:
        Операция по карте: ..., дата создания транзакции: ..., место совершения операции: ..., MCC: ...
        Стало:
        Карта ... • дата ... • место ... • MCC ...
        """
        text = self._clean_spaces(description)

        replacements = [
            ("Операция по карте: ", "Карта "),
            (", дата создания транзакции: ", " • дата "),
            (", место совершения операции: ", " • место "),
            (", MCC: ", " • MCC "),
            ("Категория: ", ""),
            (" Период проживания: ", " • период проживания: "),
            (" в банк ", " • банк "),
        ]

        for old, new in replacements:
            text = text.replace(old, new)

        text = re.sub(r"\s*•\s*", " • ", text)
        text = self._clean_spaces(text)

        return escape(text)

    @staticmethod
    def _format_month_label(raw_month) -> str:
        """
        Преобразует 2025-08 -> 08.2025
        """
        text = str(raw_month)
        m = re.match(r"(\d{4})-(\d{2})", text)
        if m:
            year, month = m.groups()
            return f"{month}.{year}"
        return text
