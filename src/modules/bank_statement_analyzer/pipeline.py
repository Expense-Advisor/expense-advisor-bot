import pandas as pd

from src.modules.bank_statement_analyzer.infrastructure.services.anomalies.anomalies_spending import \
    AnomaliesSpendingAnalyzer
from src.modules.bank_statement_analyzer.infrastructure.services.building_user_profile.build_user_profile import \
    BuildUserProfile
from src.modules.bank_statement_analyzer.infrastructure.services.classification_other_operations.classifications import \
    ClassificationOtherOperation
from src.modules.bank_statement_analyzer.infrastructure.services.estimation_savings.estimate_savings import \
    EstimateSavings
from src.modules.bank_statement_analyzer.infrastructure.services.loader.loader import BankStatementLoader
from src.modules.bank_statement_analyzer.infrastructure.services.loader.smart_category import SmartCategory
from src.modules.bank_statement_analyzer.infrastructure.services.regular_expenses.search_for_regular_expenses import \
    SearchForRegularExpenses


class AnalyzerPipeline(object):
    def __init__(self, path):
        self.path = path

        self.bank_statement_loader = BankStatementLoader(self.path)

        self.smart_category: SmartCategory | None = None
        self.classification_other_operation: ClassificationOtherOperation | None = None
        self.search_for_regular_expenses: SearchForRegularExpenses | None = None
        self.anomalies_spending: AnomaliesSpendingAnalyzer | None = None

        self.build_user_profile: BuildUserProfile | None = None
        self.estimation_savings: EstimateSavings | None = None

    def run(self) -> str:
        df: pd.DataFrame = self.bank_statement_loader.run()

        self.smart_category = SmartCategory(df)
        df: pd.DataFrame = self.smart_category.run()

        self.classification_other_operation = ClassificationOtherOperation(df)
        df: pd.DataFrame = self.classification_other_operation.run()

        self.search_for_regular_expenses = SearchForRegularExpenses(df)
        recurring_groups: pd.DataFrame = self.search_for_regular_expenses.run()

        self.anomalies_spending = AnomaliesSpendingAnalyzer(df)
        df: pd.DataFrame = self.anomalies_spending.run()
        anomalies = df[df["anomaly"] == 1]

        self.build_user_profile = BuildUserProfile(df)
        profile, profile_advice = self.build_user_profile.build()

        self.estimation_savings = EstimateSavings(recurring_groups, profile)
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
    ) -> str:
        text: list[str] = []

        # ----------------------------
        # 1. Куда уходят деньги
        # ----------------------------
        text.append("📊 КУДА УХОДЯТ ДЕНЬГИ\n")

        by_cat = (
            df.groupby("final_category")["amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
        )

        total = by_cat.sum()

        for cat, value in by_cat.items():
            share = value / total * 100
            text.append(f"- {cat}: {value:,.0f} ₽ ({share:.1f}%)")

        # ----------------------------
        # 2. Регулярные траты
        # ----------------------------
        text.append("\n🔁 ВАШИ РЕГУЛЯРНЫЕ ПЛАТЕЖИ\n")

        if len(recurring_groups) == 0:
            text.append("Регулярных платежей не найдено.")
        else:
            for _, row in recurring_groups.sort_values("total").iterrows():
                avg = abs(row["total"]) / row["count"]
                text.append(
                    f"- {row['description']} → {row['count']} раз, "
                    f"≈ {avg:.0f} ₽, всего {abs(row['total']):,.0f} ₽"
                )

        # ----------------------------
        # 3. Аномальные операции
        # ----------------------------
        text.append("\n⚠️ НЕОБЫЧНЫЕ ТРАТЫ\n")

        if len(anomalies) == 0:
            text.append("Аномальных операций не обнаружено.")
        else:
            for _, row in anomalies.sort_values("amount").head(10).iterrows():
                text.append(
                    f"- {row['date'].date()} | {row['description'][:50]}… → {row['amount']} ₽"
                )

        # ----------------------------
        # 4. Поведенческий анализ (ML)
        # ----------------------------
        text.append("\n🧠 АНАЛИЗ ВАШЕГО ФИНАНСОВОГО ПОВЕДЕНИЯ\n")

        for line in profile_advice:
            text.append(f"- {line}")

        # ----------------------------
        # 5. Итог по экономии
        # ----------------------------
        text.append("\n💰 ПОТЕНЦИАЛ ЭКОНОМИИ\n")
        text.append(
            f"Если оптимизировать выявленные привычки, можно сохранить около {abs(savings):,.0f} ₽ за этот период.")

        return "\n".join(text)