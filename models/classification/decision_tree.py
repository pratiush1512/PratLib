# pratlib/models/classification/decision_tree.py
from pyspark.ml.classification import DecisionTreeClassifier as SparkDecisionTreeClassifier

class DecisionTreeClassifier:
    def __init__(self, **kwargs):
        self.model = SparkDecisionTreeClassifier(**kwargs)

    def fit(self, df, features_col='features', label_col='label'):
        self.model.setFeaturesCol(features_col).setLabelCol(label_col)
        self.fitted_model = self.model.fit(df)
        return self

    def predict(self, df):
        return self.fitted_model.transform(df)
