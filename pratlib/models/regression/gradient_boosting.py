# pratlib/models/regression/gradient_boosting.py
from pyspark.ml.regression import GBTRegressor as SparkGBTRegressor

class GradientBoostedTreeRegressor:
    def __init__(self, **kwargs):
        self.model = SparkGBTRegressor(**kwargs)

    def fit(self, df, features_col='features', label_col='label'):
        self.model.setFeaturesCol(features_col).setLabelCol(label_col)
        self.fitted_model = self.model.fit(df)
        return self

    def predict(self, df):
        return self.fitted_model.transform(df)
