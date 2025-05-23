from pyspark.sql import DataFrame

def train_test_split(df: DataFrame, test_size=0.25, seed=None):
    """
    Split a Spark DataFrame into train and test sets.

    Args:
        df (DataFrame): input Spark DataFrame
        test_size (float): fraction of data to be used for testing (between 0 and 1)
        seed (int or None): random seed for reproducibility

    Returns:
        train_df, test_df (DataFrame, DataFrame)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    train_fraction = 1 - test_size
    train_df, test_df = df.randomSplit([train_fraction, test_size], seed=seed)
    return train_df, test_df
