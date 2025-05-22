from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor as SparkDecisionTreeRegressor

class DecisionTreeRegressor:
    def __init__(self, **kwargs):
        """
        kwargs: parameters for Spark's DecisionTreeRegressor
        """
        self.model = SparkDecisionTreeRegressor(**kwargs)
        self.assembler = None
        self.fitted_model = None
        self.feature_cols = None

    def fit(self, df, label_col='label'):
        """
        Fit the model. Automatically infers feature columns from DataFrame by excluding label_col.
        """
        # Infer feature columns
        self.feature_cols = [col for col in df.columns if col != label_col]
        self.assembler = VectorAssembler(inputCols=self.feature_cols, outputCol='features')

        # Transform dataframe
        df_vectorized = self.assembler.transform(df)
        self.model.setFeaturesCol('features').setLabelCol(label_col)
        self.fitted_model = self.model.fit(df_vectorized)
        return self

    def predict(self, df):
        """
        Predict using the fitted model. Assumes df has the same feature columns as used during training.
        """
        if self.fitted_model is None or self.assembler is None:
            raise ValueError("Model has not been fitted yet.")
        
        df_vectorized = self.assembler.transform(df)
        return self.fitted_model.transform(df_vectorized)
