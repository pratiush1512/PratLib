## Module 1
from pyspark.ml.feature import OneHotEncoder as SparkOneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import when,col
import warnings

class OneHotEncoder:
    def __init__(self, **kwargs):
        self.encoder = None
        self.indexer = None
        self.indexer_output_col = None
        self.encoder_output_col = None

    def fit(self, df, input_col):
        self.indexer_output_col = f"{input_col}_index"
        self.encoder_output_col = f"{input_col}_encoded"
        
        self.indexer = StringIndexer(inputCol=input_col, outputCol=self.indexer_output_col).fit(df)
        indexed_df = self.indexer.transform(df)
        
        self.encoder = SparkOneHotEncoder(inputCol=self.indexer_output_col, outputCol=self.encoder_output_col).fit(indexed_df)
        return self

    def transform(self, df):
        if not self.encoder or not self.indexer:
            raise AttributeError("The encoder has not been fitted yet.")
        
        indexed_df = self.indexer.transform(df)
        new_df = self.encoder.transform(indexed_df)
        new_df = new_df.drop(self.indexer_output_col)
        return new_df

class LabelEncoder:
    def __init__(self, **kwargs):
        self.indexer = StringIndexer(**kwargs)

    def fit(self, df, input_col):
        self.indexer.setInputCol(input_col).setOutputCol(f"{input_col}_new")
        self.model = self.indexer.fit(df)
        return self
            
    def transform(self, df):
        return self.model.transform(df)

class map:
    def __init__(self):
        pass
    def mapper(self, df, column_name, dic):
        for key, value in dic.items():
            df = df.withColumn(
                column_name,
                when(df[column_name] == key, value).otherwise(df[column_name])
            )
        return df
        
class Cast:
    def __init__(self):
        pass

    def cast_col(self, df, feature, datatype):
        df = df.withColumn(
            feature, col(feature).cast(datatype) 
        )
        return df