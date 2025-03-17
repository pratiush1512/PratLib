# pratlib/models/classification/logistic_regression.py
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression

class LogisticRegression:
    def __init__(self, **kwargs):
        self.model = SparkLogisticRegression(**kwargs)

    def fit(self, df, features_col='features', label_col='label'):
        self.model.setFeaturesCol(features_col).setLabelCol(label_col)
        self.fitted_model = self.model.fit(df)
        return self

    def predict(self, df):
        return self.fitted_model.transform(df)
