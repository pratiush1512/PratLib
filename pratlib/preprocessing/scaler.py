from pyspark.ml.feature import StandardScaler as SparkStandardScaler, MinMaxScaler as SparkMinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

class StandardScaler:
    def __init__(self, **kwargs):
        self.scaler = SparkStandardScaler(**kwargs)
        self.model = None
        self.input_col = None
        self.output_col = None
        self.temp_vec_col = None

    def fit(self, df, input_col):
        self.input_col = input_col
        self.temp_vec_col = f"{input_col}_vec"
        self.output_col = f"{input_col}_scaled"

        # Assemble into vector (required by Spark)
        assembler = VectorAssembler(inputCols=[input_col], outputCol=self.temp_vec_col)
        df_vec = assembler.transform(df)

        self.scaler.setInputCol(self.temp_vec_col).setOutputCol(self.output_col)
        self.model = self.scaler.fit(df_vec)
        return self

    def transform(self, df):
        assembler = VectorAssembler(inputCols=[self.input_col], outputCol=self.temp_vec_col)
        df_vec = assembler.transform(df)
        df_scaled = self.model.transform(df_vec)

        # Convert vector to scalar
        df_scaled = df_scaled.withColumn(self.input_col, vector_to_array(col(self.output_col))[0])
        return df_scaled.drop(self.temp_vec_col, self.output_col)

    def fit_transform(self, df, input_col):
        self.fit(df, input_col)
        return self.transform(df)

class MinMaxScaler:
    def __init__(self, **kwargs):
        self.scaler = SparkMinMaxScaler(**kwargs)
        self.model = None
        self.input_col = None
        self.output_col = None
        self.temp_vec_col = None

    def fit(self, df, input_col):
        self.input_col = input_col
        self.temp_vec_col = f"{input_col}_vec"
        self.output_col = f"{input_col}_scaled"

        assembler = VectorAssembler(inputCols=[input_col], outputCol=self.temp_vec_col)
        df_vec = assembler.transform(df)

        self.scaler.setInputCol(self.temp_vec_col).setOutputCol(self.output_col)
        self.model = self.scaler.fit(df_vec)
        return self

    def transform(self, df):
        assembler = VectorAssembler(inputCols=[self.input_col], outputCol=self.temp_vec_col)
        df_vec = assembler.transform(df)
        df_scaled = self.model.transform(df_vec)

        df_scaled = df_scaled.withColumn(self.input_col, vector_to_array(col(self.output_col))[0])
        return df_scaled.drop(self.temp_vec_col, self.output_col)

    def fit_transform(self, df, input_col):
        self.fit(df, input_col)
        return self.transform(df)
