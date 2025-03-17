# pratlib/models/classification/random_forest.py
from pyspark.ml.classification import RandomForestClassifier as SparkRandomForestClassifier

class RandomForestClassifier:
    def __init__(self, **kwargs):
        self.model = SparkRandomForestClassifier(**kwargs)

    def fit(self, df, features_col='features', label_col='label'):
        self.model.setFeaturesCol(features_col).setLabelCol(label_col)
        self.fitted_model = self.model.fit(df)
        return self

    def predict(self, df):
        return self.fitted_model.transform(df)
