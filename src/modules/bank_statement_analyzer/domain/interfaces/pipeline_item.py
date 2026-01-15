from typing import Protocol, Union

import pandas as pd


class PipelineItem(Protocol):
    def __init__(self, df: pd.DataFrame):
        ...

    def run(self, **kwargs) -> Union[pd.DataFrame, float]:
        ...
