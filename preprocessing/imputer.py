from pyspark.ml.feature import Imputer as SparkImputer

class Imputer:
    def __init__(self, **kwargs):
        self.imputer = SparkImputer(**kwargs)

    def fit(self, df, input_cols, strategy):
        self.imputer.setInputCols([input_cols]).setOutputCols([f"{input_cols}_imputed"]).setStrategy(strategy)
        self.model = self.imputer.fit(df)
        return self

    def transform(self, df):
        return self.model.transform(df)
