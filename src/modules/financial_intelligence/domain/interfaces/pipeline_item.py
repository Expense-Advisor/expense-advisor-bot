from typing import Protocol

import pandas as pd


class PipelineItem(Protocol):
    """
    Контракт одного шага ML-пайплайна.

    Любой элемент пайплайна обязан:
        - принимать DataFrame с транзакциями
        - возвращать либо DataFrame, либо числовой результат
    """

    def run(self) -> pd.DataFrame:
        """
        Выполняет логику шага.

        Returns:
            Union[pd.DataFrame, float]:
                - обновлённый DataFrame
                - или числовой результат (например, сумма экономии).
        """
        ...
