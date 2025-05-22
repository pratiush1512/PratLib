from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier as SparkDecisionTreeClassifier

class DecisionTreeClassifier:
    def __init__(self, **kwargs):
        self.model = SparkDecisionTreeClassifier(**kwargs)
        self.assembler = None
        self.fitted_model = None
        self.feature_cols = None

    def fit(self, df, label_col='label'):
        self.feature_cols = [col for col in df.columns if col != label_col]
        self.assembler = VectorAssembler(inputCols=self.feature_cols, outputCol='features')
        df_vectorized = self.assembler.transform(df)
        self.model.setFeaturesCol('features').setLabelCol(label_col)
        self.fitted_model = self.model.fit(df_vectorized)
        return self

    def predict(self, df):
        if not self.fitted_model or not self.assembler:
            raise ValueError("Model must be fitted before predicting.")
        df_vectorized = self.assembler.transform(df)
        return self.fitted_model.transform(df_vectorized)
