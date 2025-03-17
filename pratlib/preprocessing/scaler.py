# pratlib/preprocessing/scaler.py
from pyspark.ml.feature import StandardScaler as SparkStandardScaler, MinMaxScaler as SparkMinMaxScaler

class StandardScaler:
    def __init__(self, **kwargs):
        self.scaler = SparkStandardScaler(**kwargs)

    def fit(self, df, input_col, output_col):
        self.scaler.setInputCol(input_col).setOutputCol(output_col)
        self.model = self.scaler.fit(df)
        return self

    def transform(self, df):
        return self.model.transform(df)

class MinMaxScaler:
    def __init__(self, **kwargs):
        self.scaler = SparkMinMaxScaler(**kwargs)

    def fit(self, df, input_col, output_col):
        self.scaler.setInputCol(input_col).setOutputCol(output_col)
        self.model = self.scaler.fit(df)
        return self

    def transform(self, df):
        return self.model.transform(df)
