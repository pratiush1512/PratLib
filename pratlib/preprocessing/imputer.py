from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, Any

class SimpleImputer:
    def __init__(self, imputations: Dict[str, Dict[str, Any]]):
        """
        imputations: A dictionary specifying imputation strategy per column.
        Format:
        {
            'column_name': {
                'strategy': 'mean' | 'median' | 'mode' | 'constant',
                'fill_value': <value>  # required only for 'constant'
            },
            ...
        }
        """
        self.imputations = imputations
        self.fill_values = {}

    def fit(self, df: DataFrame):
        for col, config in self.imputations.items():
            strategy = config['strategy']
            if strategy == 'mean':
                self.fill_values[col] = df.select(F.mean(F.col(col))).first()[0]
            elif strategy == 'median':
                self.fill_values[col] = df.approxQuantile(col, [0.5], 0.001)[0]
            elif strategy == 'mode':
                self.fill_values[col] = (
                    df.groupBy(col).count().orderBy(F.desc("count")).first()[0]
                )
            elif strategy == 'constant':
                self.fill_values[col] = config['fill_value']
            else:
                raise ValueError(f"Unsupported strategy '{strategy}' for column '{col}'")

    def transform(self, df: DataFrame) -> DataFrame:
        for col, value in self.fill_values.items():
            df = df.withColumn(
                col,
                F.when(F.col(col).isNull(), F.lit(value)).otherwise(F.col(col))
            )
        return df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        self.fit(df)
        return self.transform(df)
