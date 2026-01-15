import pandas as pd

from src.modules.bank_statement_analyzer.domain.entites.mcc import MCCHelper
from src.modules.bank_statement_analyzer.domain.interfaces.pipeline_item import PipelineItem


class BankStatementLoader(PipelineItem):
    """
    Загрузчик и нормализатор банковских Excel-выписок.

    Преобразует «грязкие» файлы банков в стандартную таблицу транзакций
    со следующей схемой:

        date | description | amount | category | mcc
    """

    def __init__(self, file_path):
        self.path = file_path

    def run(self, **kwargs) -> pd.DataFrame:
        """
        Загружает и нормализует банковскую выписку

        Returns:
            pd.DataFrame: Нормализованная таблица транзакций
        """
        df = self._load_bank_xlsx()
        df = self._normalize_columns(df)
        return df

    def _find_header_row(self, df_raw: pd.DataFrame, required_cols: list[str], max_scan: int = 200) -> int:
        """
        Находит строку с заголовками в грязном Excel-файле.

        Банковские выписки часто содержат перед таблицей служебные строки
        (название клиента, период и т.д.). Этот метод ищет строку,
        в которой присутствуют ключевые слова заголовков.

        Args:
            df_raw (pd.DataFrame): DataFrame без заголовков.
            required_cols (List[str]): Ключевые слова для поиска колонок.
            max_scan (int): Максимальное количество строк для анализа.

        Returns:
            int: Индекс строки, содержащей заголовки таблицы.

        Raises:
            ValueError: Если строка заголовков не найдена.
        """
        best_i, best_score = 0, -1

        for i in range(min(len(df_raw), max_scan)):
            row = df_raw.iloc[i].astype(str).str.lower().tolist()

            score = 0
            for col in required_cols:
                for cell in row:
                    if col.lower() in cell:
                        score += 1
                        break

            if score > best_score:
                best_score = score
                best_i = i

            if score >= len(required_cols) - 1:
                return i

        if best_score <= 0:
            raise ValueError("Не удалось найти строку заголовков.")

        return best_i

    def _load_bank_xlsx(self) -> pd.DataFrame:
        """
        Загружает Excel-файл банка и извлекает таблицу транзакций.

        Returns:
            pd.DataFrame: Таблица транзакций с исходными колонками.
        """
        df_raw = pd.read_excel(self.path, header=None)

        required = ["дата", "сум", "опис", "катег"]
        header_row = self._find_header_row(df_raw, required_cols=required)

        df = pd.read_excel(self.path, header=header_row)
        df = df.dropna(how="all").reset_index(drop=True)

        df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Приводит колонки банка к стандартной схеме.

        Итоговая схема:
            date | description | amount | category | mcc

        Args:
            df (pd.DataFrame): Исходная таблица банка.

        Returns:
            pd.DataFrame: Нормализованная таблица транзакций.
        """
        detected: dict[str, list] = {
            "date": [],
            "description": [],
            "amount": [],
            "category": []
        }

        for col in df.columns:
            name = str(col).lower()

            if "дата" in name:
                detected["date"].append(col)
            elif "опис" in name or "назнач" in name:
                detected["description"].append(col)
            elif "сум" in name or "amount" in name:
                detected["amount"].append(col)
            elif "катег" in name:
                detected["category"].append(col)

        final_map = {cols[0]: key for key, cols in detected.items() if len(cols) > 0}
        df = df.rename(columns=final_map)

        if "category" not in df.columns:
            df["category"] = "Другое"

        df = df[["date", "description", "amount", "category"]]
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            dayfirst=True
        )

        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        df = df.dropna(subset=["date", "amount"])
        df["mcc"] = df["description"].astype(str).apply(MCCHelper.extract_mcc)

        return df
