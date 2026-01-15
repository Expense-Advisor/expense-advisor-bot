import pandas as pd

from src.modules.bank_statement_analyzer.domain.entites.mcc import MCCHelper
from src.modules.bank_statement_analyzer.domain.interfaces.pipeline_item import PipelineItem


class BankStatementLoader(PipelineItem):
    def __init__(self, file_path):
        self.path = file_path

    def run(self, **kwargs) -> pd.DataFrame:
        df = self._load_bank_xlsx()
        df = self._normalize_columns(df)
        return df

    def _find_header_row(self, df_raw: pd.DataFrame, required_cols: list[str], max_scan: int = 200) -> int:
        """
        Searches for the header row in messy bank Excel files.

        Args:
            df_raw: Raw DataFrame without headers.
            required_cols: Column name keywords to search for.
            max_scan: Max rows to scan.

        Returns:
            Index of header row.
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
        Loads messy bank Excel file and extracts the real transaction table.

        Args:
            path: Path to Excel file.

        Returns:
            DataFrame with bank transactions.
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
        Converts bank columns to standard schema:
        date, description, amount, category
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
