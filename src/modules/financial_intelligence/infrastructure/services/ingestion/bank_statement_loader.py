import re
from io import BytesIO

import numpy as np
import pandas as pd

from src.modules.financial_intelligence.domain.entities.mcc import MCCHelper
from src.modules.financial_intelligence.domain.interfaces.pipeline_item import (
    PipelineItem,
)


class BankStatementIngestor(PipelineItem):
    """
    Загрузчик и нормализатор банковских Excel-выписок.

    Преобразует «грязные» файлы банков в стандартную таблицу транзакций:

        date | description | amount | category | mcc
    """

    def __init__(self, content: bytes):
        self.content = content

    def run(self, **kwargs) -> pd.DataFrame:
        df = self._load_bank_xlsx()
        df = self._normalize_columns(df)

        print("rows:", len(df))
        print("expense_sum:", df.loc[df["amount"] < 0, "amount"].sum())
        print("income_sum:", df.loc[df["amount"] > 0, "amount"].sum())

        return df

    def _find_header_row(
        self, df_raw: pd.DataFrame, required_cols: list[str], max_scan: int = 200
    ) -> int:
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
        bytes_io = BytesIO(self.content)
        df_raw = pd.read_excel(io=bytes_io, header=None)

        required = ["дата", "сум", "опис", "катег"]
        header_row = self._find_header_row(df_raw, required_cols=required)

        bytes_io.seek(0)
        df = pd.read_excel(io=bytes_io, header=header_row)

        df = df.dropna(how="all").reset_index(drop=True)
        df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", regex=True)]

        return df

    @staticmethod
    def _parse_amount(value) -> float:
        """
        Корректно парсит денежные суммы вида:
        -1 200
        1 587,78
        -4 407,40
        3 000,00 RUR
        """
        if pd.isna(value):
            return np.nan

        s = str(value)

        s = s.replace("\xa0", "")
        s = s.replace(" ", "")
        s = s.replace(",", ".")
        s = re.sub(r"[^0-9.\-]", "", s)

        if s in {"", "-", ".", "-."}:
            return np.nan

        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _clean_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        detected: dict[str, list] = {
            "date": [],
            "description": [],
            "amount": [],
            "category": [],
        }

        for col in df.columns:
            name = str(col).lower().strip()

            if "дата" in name:
                detected["date"].append(col)
            elif "опис" in name or "назнач" in name:
                detected["description"].append(col)
            elif "сум" in name or "amount" in name:
                detected["amount"].append(col)
            elif "катег" in name:
                detected["category"].append(col)

        final_map = {cols[0]: key for key, cols in detected.items() if cols}
        df = df.rename(columns=final_map)

        required = ["date", "description", "amount"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Не найдены обязательные колонки: {missing}")

        if "category" not in df.columns:
            df["category"] = "Прочие операции"

        df = df[["date", "description", "amount", "category"]].copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df["description"] = df["description"].apply(self._clean_text)
        df["category"] = (
            df["category"].fillna("Прочие операции").astype(str).str.strip()
        )
        df["amount"] = df["amount"].apply(self._parse_amount)

        df = df.dropna(subset=["date", "amount"]).copy()

        # удаляем возможные мусорные строки, если они вдруг пережили чтение
        df = df[
            ~df["description"].str.contains(
                r"страница\s+\d+|подпись|выписка по счету",
                case=False,
                na=False,
                regex=True,
            )
        ].copy()

        df["mcc"] = df["description"].astype(str).apply(MCCHelper.extract_mcc)

        return df.reset_index(drop=True)
