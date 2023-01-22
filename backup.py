from io import StringIO

from pandas import concat, read_csv
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession
from requests import get

def load_wine_quality():
    WINE_QUALITY_BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/'
    RED_WINES_FILENAME = 'winequality-red.csv'
    WHITE_WINES_FILENAME = 'winequality-white.csv'

    red_wines_csv = get(
        f'{WINE_QUALITY_BASE_URL}{RED_WINES_FILENAME}'
    )
    white_wines_csv = get(
        f'{WINE_QUALITY_BASE_URL}{WHITE_WINES_FILENAME}'
    )
    red_wines_df = read_csv(StringIO(red_wines_csv.text), sep=';')
    red_wines_df['color'] = 'red'
    white_wines_df = read_csv(StringIO(white_wines_csv.text), sep=';')
    white_wines_df['color'] = 'white'
    return concat(
        [red_wines_df, white_wines_df]
    )

def load_census_income():
    ADULT_BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/'
    TRAIN_DATA = 'adult.data'
    TEST_DATA = 'adult.test'
    adult_train_csv = get(f'{CENSUS_INCOME_URL}{TRAIN_DATA})
    adult_test_csv = get(f'{CENSUS_INCOME_URL}{TEST_DATA})

if __name__ == '__main__':
    NUMERIC = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    spark = SparkSession.builder.master(
                "local[1]"
            ).appName(
                "Juran"
            ).getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    wines_df = load_wine_quality()
    categorical_columns = wines_df.select_dtypes(
        exclude=NUMERIC
    ).columns.to_list()
    indexed_columns = [
        f'{column}_index' for column in categorical_columns
    ]
    one_hot_encoded_columns = [
        f'{column}_ohe' for column in categorical_columns
    ]
    string_indexer = StringIndexer(
        inputCols=categorical_columns,
        outputCols=indexed_columns
    )
    one_hot_encoder = OneHotEncoder(
        inputCols=indexed_columns,
        outputCols=one_hot_encoded_columns
    )
    wines_sparkdf = spark.createDataFrame(wines_df)