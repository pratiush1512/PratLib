from pyspark.ml.feature import VectorAssembler

class Vectorizer:
    def __init__(self,**kwargs):
        self.model = VectorAssembler(**kwargs)

    def vectorize(self,df, input_col):
        self.model.setInputCols(input_col).setOutputCol("Independent Features")
        transformed = self.model.transform(df)
        return transformed