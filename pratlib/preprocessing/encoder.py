from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder as SparkOHE
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from pyspark.sql import DataFrame

class OneHotEncoder:
    def __init__(self):
        self.indexer = None
        self.encoder = None
        self.categories = []
        self.input_col = None
        self.index_col = None
        self.encoded_col = None

    def fit(self, df: DataFrame, input_col: str):
        self.input_col = input_col
        self.index_col = f"{input_col}_index_temp"
        self.encoded_col = f"{input_col}_encoded_temp"

        # Indexer
        self.indexer = StringIndexer(inputCol=input_col, outputCol=self.index_col, handleInvalid="keep").fit(df)
        self.categories = self.indexer.labels  # Store labels for later column names

        # Encoder
        indexed_df = self.indexer.transform(df)
        self.encoder = SparkOHE(inputCol=self.index_col, outputCol=self.encoded_col).fit(indexed_df)
        return self

    def transform(self, df: DataFrame):
        # Apply indexer and encoder
        df = self.indexer.transform(df)
        df = self.encoder.transform(df)

        # Convert vector to array
        df = df.withColumn(f"{self.encoded_col}_arr", vector_to_array(col(self.encoded_col)))

        # Explode into individual binary columns
        for i, category in enumerate(self.categories):
            new_col = f"{self.input_col}_{category}"
            df = df.withColumn(new_col, col(f"{self.encoded_col}_arr")[i])

        # Drop temp columns and original
        df = df.drop(self.input_col, self.index_col, self.encoded_col, f"{self.encoded_col}_arr")
        return df

    def fit_transform(self, df: DataFrame, input_col: str):
        self.fit(df, input_col)
        return self.transform(df)


class LabelEncoder:
    def __init__(self):
        self.model = None
        self.input_col = None
        self.output_col = None

    def fit(self, df, input_col):
        self.input_col = input_col
        self.output_col = f"{input_col}_label"
        indexer = StringIndexer(inputCol=input_col, outputCol=self.output_col, handleInvalid="keep")
        self.model = indexer.fit(df)
        return self

    def transform(self, df):
        # Apply transformation to get the temporary label column
        transformed_df = self.model.transform(df)
        # Drop the original column and rename the encoded one
        transformed_df = (
            transformed_df
            .drop(self.input_col)
            .withColumnRenamed(self.output_col, self.input_col)
        )
        return transformed_df

    def fit_transform(self, df, input_col):
        self.fit(df, input_col)
        return self.transform(df)

class ValueMapper:
    def __init__(self):
        pass

    def map_values(self, df, column_name, mapping_dict):
        expr = None
        for key, val in mapping_dict.items():
            condition = (col(column_name) == key)
            expr = when(condition, val) if expr is None else expr.when(condition, val)
        df = df.withColumn(column_name, expr.otherwise(col(column_name)))
        return df

class TypeCaster:
    def __init__(self):
        pass

    def cast_column(self, df, column, dtype):
        return df.withColumn(column, col(column).cast(dtype))
