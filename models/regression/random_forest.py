# pratlib/models/regression/random_forest.py
from pyspark.ml.regression import RandomForestRegressor as SparkRandomForestRegressor

class RandomForestRegressor:
    def __init__(self, **kwargs):
        self.model = SparkRandomForestRegressor(**kwargs)

    def fit(self, df, features_col='features', label_col='label'):
        self.model.setFeaturesCol(features_col).setLabelCol(label_col)
        self.fitted_model = self.model.fit(df)
        return self

    def predict(self, df):
        return self.fitted_model.transform(df)
