import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.modules.bank_statement_analyzer.domain.interfaces.pipeline_item import PipelineItem


class ClassificationOtherOperation(PipelineItem):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self, **kwargs) -> pd.DataFrame:
        model, vec = self.train_category_model(self.df)
        return self.classify_other_operations(self.df, model, vec)

    def train_category_model(self, df: pd.DataFrame):
        """
        Trains ML model to predict categories from descriptions.
        """
        train_df = df[df["final_category"] != "Прочие операции"]

        X = train_df["description"].astype(str)
        y = train_df["final_category"]

        vectorizer = TfidfVectorizer(max_features=1000)
        X_vec = vectorizer.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_vec, y)

        return model, vectorizer

    def classify_other_operations(self, df: pd.DataFrame, model, vectorizer):
        """
        Reclassifies 'Прочие операции' using ML.
        """
        mask = df["category"] == "Прочие операции"

        X = vectorizer.transform(df.loc[mask, "description"].astype(str))
        predictions = model.predict(X)

        df.loc[mask, "final_category"] = predictions
        df.loc[~mask, "final_category"] = df.loc[~mask, "category"]

        return df
