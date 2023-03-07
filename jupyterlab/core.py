from decimal import Decimal
from functools import reduce
from io import StringIO
from logging import getLogger, basicConfig
from typing import Tuple

from matplotlib import pyplot
from pandas import DataFrame as PandasDataFrame
from pandas import concat, read_csv
from pyspark.ml import Estimator, Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator,
                                   RegressionEvaluator)
from pyspark.ml.feature import (Imputer, OneHotEncoder, QuantileDiscretizer,
                                SQLTransformer, StringIndexer, VectorAssembler)
from pyspark.ml.regression import (DecisionTreeRegressor, GBTRegressor,
                                   GeneralizedLinearRegression,
                                   RandomForestRegressor)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, length, levenshtein, rand,
                                   regexp_replace, stddev, udf, when)
from pyspark.sql.types import BooleanType
from requests import get

logger = getLogger(__name__)
logger.setLevel(level='INFO')
basicConfig()

UNIVERSAL_SEED = 420

test_pollution = False

class Juran:
    
    NUMERIC = ('bigint','int', 'double', 'float')

    def __init__(self, standard_deviation_range: int=6) -> None:
        self.standard_deviation_range = standard_deviation_range

    def calculate_data_quality(self, dataframe: SparkDataFrame) -> float:
        completeness_score = self.calculate_completeness_score(dataframe)
        accurateness_score = self.calculate_accurateness_score(dataframe)
        return completeness_score * accurateness_score
        
    def calculate_completeness_score(self, dataframe: SparkDataFrame) -> float:
        return 1 - self.percentage_of_missing_values(dataframe)
    
    def calculate_accurateness_score(self, dataframe: SparkDataFrame) -> float:
        return 1 - self.percentage_of_strange_values(dataframe)
    
    @staticmethod
    def filter_out_missing(dataframe: SparkDataFrame, column: str) -> SparkDataFrame:
        return dataframe.filter(
            ~(
                col(column).isNull() 
                | (length(regexp_replace(column, r"[\t\n ]+", "")) == 0)
            )
        )
    
    def percentage_of_missing_values(self, dataframe: SparkDataFrame) -> float:
        non_missing_count = 0
        for column in dataframe.columns:
            non_missing_count += self.filter_out_missing(dataframe, column).count()
        total_cells = dataframe.count() * len(dataframe.columns)
        return (1 - (non_missing_count / total_cells))

    def percentage_of_outliers_without_missing_values(
        self, dataframe: SparkDataFrame, column: str
    ) -> float:
        filtered = self.filter_out_missing(dataframe, column)

        mean_value = filtered.select(column).agg({"*": "avg"}).collect()[0][0]
        standard_deviation = filtered.select(stddev(column)).collect()[0][0]
        upper_limit = mean_value + (self.standard_deviation_range/2) * standard_deviation
        lower_limit = mean_value - (self.standard_deviation_range/2) * standard_deviation
        total_count = filtered.count()

        outliers_count = filtered.filter(
            (col(column) > upper_limit) | (col(column) < lower_limit)
        ).count()

        return outliers_count / total_count * 100

    def percentage_of_potential_typos_without_missing_values(
        self,
        dataframe: SparkDataFrame,
        column: str,
        edit_distance_threshold: int=2
    ) -> float:

        levenshtein_udf = udf(lambda str1, str2: levenshtein(str1, str2) <= 2, BooleanType())

        non_missing = self.filter_out_missing(dataframe, column)
        filtered = non_missing.filter(length(column) > 3)

        total_cells = filtered.select(column).count()

        matching_cells = filtered.alias('d1') \
            .join(filtered.alias('d2'), on=col('d1.' + column) != col('d2.' + column)) \
            .filter(levenshtein_udf(col('d1.' + column), col('d2.' + column))) \
            .select(col('d1.' + column)) \
            .distinct() \
            .count()

        percentage = (matching_cells / total_cells) * 100

        return percentage

    def percentage_of_strange_values(self, dataframe: SparkDataFrame) -> float:
        result = 1
        for column, data_type in dataframe.dtypes:
            if data_type in self.NUMERIC:
                result *= self.percentage_of_outliers_without_missing_values(
                    dataframe, column
                )
            else:
                result *= self.percentage_of_potential_typos_without_missing_values(
                    dataframe, column
                )
        return result

    @staticmethod
    def pollute_with_nulls(dataframe, null_ratio):
        for column in dataframe.columns:
            dataframe = dataframe.withColumn(
                column, when(rand() < null_ratio, None).otherwise(
                    col(column)
                )
            )
        
        return dataframe


def load_stratified_wine_quality() -> Tuple[SparkDataFrame, SparkDataFrame]:
    logger.info('Loading wine data')
    WINE_QUALITY_BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/'
    RED_WINES_FILENAME = 'winequality-red.csv'
    WHITE_WINES_FILENAME = 'winequality-white.csv'
    TARGET_COLUMN = 'quality'

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
    data = concat(
        [red_wines_df, white_wines_df]
    )
    data.rename(columns={TARGET_COLUMN: 'target'}, inplace=True)
    spark_dataframe = spark.createDataFrame(data)
    discretizer = QuantileDiscretizer(
        numBuckets=10,
        inputCol='target',
        outputCol='bucket'
    )
    spark_dataframe = discretizer.fit(spark_dataframe).transform(spark_dataframe)
    fractions = spark_dataframe.groupBy('bucket').count().withColumn(
        'fraction', col('count') / spark_dataframe.count()
    ).select('bucket', 'fraction').rdd.collectAsMap()
    train_df = spark_dataframe.stat.sampleBy('bucket', fractions, seed=UNIVERSAL_SEED)
    test_df = spark_dataframe.subtract(train_df)
    logger.info('Finished loading wine data')
    return train_df, test_df
    

def load_census_income() -> Tuple[SparkDataFrame, SparkDataFrame]:
    ADULT_BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/'
    TRAIN_DATA = 'adult.data'
    TEST_DATA = 'adult.test'
    TARGET_COLUMN = 'income'

    adult_train_csv = get(f'{CENSUS_INCOME_URL}{TRAIN_DATA}')
    adult_test_csv = get(f'{CENSUS_INCOME_URL}{TEST_DATA}')
    adult_train_df = read_csv(StringIO(adult_train_csv.text), sep=',')
    adult_test_df = read_csv(StringIO(adult_test_csv.text), sep=',')
    adult_train_df.rename(columns={TARGET_COLUMN: 'target'}, inplace=True)
    adult_test_df.rename(columns={TARGET_COLUMN: 'target'}, inplace=True)
    train_df = spark.createDataFrame(adult_train_df)
    test_df = spark.createDataFrame(adult_test_df)
    return train_df, test_df

    
def display_head(spark_df: SparkDataFrame, number_of_rows: int=5) -> PandasDataFrame:
    return spark_df.limit(number_of_rows).toPandas().head(number_of_rows)

def get_regression_estimators(target_column: str='target') -> set:
    return {
        regressor(labelCol=target_column, **options) for regressor, options in {
            GeneralizedLinearRegression: {'regParam': .05}, 
            DecisionTreeRegressor: {'minInfoGain': .05},
            RandomForestRegressor: {'maxDepth': 10},
            GBTRegressor: {},
        }.items()
    }

def get_dataset_pipeline(
    dataframe: SparkDataFrame,
    estimator: Estimator,
    task: str='regression',
    target_column: str='target',
    number_of_folds: int=5,
    metric_name: str='rmse'
) -> Pipeline:
    NUMERIC = ('bigint','int', 'double', 'float')
    evaluation_mapper = {
        'regression': RegressionEvaluator,
        'multiclass_classification': MulticlassClassificationEvaluator,
        'binary_classification': BinaryClassificationEvaluator,
    }
    numerical_columns = [
        column for column, data_type in dataframe.dtypes if data_type in NUMERIC
    ]
    logger.debug(f'Numerical columns: {numerical_columns}')
    categorical_columns = [
        column for column in dataframe.columns if column not in numerical_columns
    ]
    logger.debug(f'Categorical columns: {categorical_columns}')
    indexed_columns = [
        f'{column}_index' for column in categorical_columns
    ]
    one_hot_encoded_columns = [
        f'{column}_ohe' for column in categorical_columns
    ]
    features = [
        *[
            column for column in numerical_columns if target_column not in column
        ],
        *[
            column for column in one_hot_encoded_columns if target_column not in column
        ],
    ]
    
    string_indexer = StringIndexer(
        inputCols=categorical_columns,
        outputCols=indexed_columns
    )
    one_hot_encoder = OneHotEncoder(
        inputCols=indexed_columns,
        outputCols=one_hot_encoded_columns
    )
    sql_transformer = SQLTransformer(
        statement="SELECT *, IF(EXISTS(SELECT target_ohe FROM __THIS__), target_ohe AS target, target) AS target FROM __THIS__"
    )
    assembler = VectorAssembler(
        inputCols=features,
        outputCol="features"
    )
    cross_validator = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=ParamGridBuilder().build(),
        evaluator=evaluation_mapper[task](
            labelCol=target_column
        ),
        numFolds=number_of_folds
    )

    return Pipeline(
        stages=[
            string_indexer,
            one_hot_encoder,
            assembler,
            cross_validator,
        ]
    )

if __name__ == '__main__':
    
    spark = SparkSession.builder.master(
                "local[1]"
            ).appName(
                "Juran"
            ).getOrCreate()
            
    juran = Juran()

    train_data, test_data = load_stratified_wine_quality()
    
    original_train_data_quality_score = juran.calculate_data_quality(train_data)
    
    polluted_train_data = [
        {
            'data': juran.pollute_with_nulls(train_data, ratio),
            'quality_score': juran.calculate_data_quality(train_data)
        } for ratio in (factor * .05 for factor in range(1, 11))
    ]
    
estimator_results = dict()

for estimator in get_regression_estimators():
    estimator_name = estimator.__class__.__name__
    logger.info(f'Executing analysis for {estimator_name}')
    model = get_dataset_pipeline(
        train_data, estimator, task='regression', metric_name='rmse'
    )
    logger.info('Executing cross validation')
    trained_model = model.fit(train_data)
    logger.info('Cross validation executed')
    average_rmse = round(trained_model.stages[-1].avgMetrics[0],4)
    average_stddev = round(trained_model.stages[-1].stdMetrics[0],4)
    logger.info(f'Average root mean squared error: {average_rmse}')
    logger.info(f'Average standard deviation: {average_stddev}')
    estimator_results[estimator_name] = {
        'cross_validation_average_rmse': average_rmse,
        'cross_vaidation_average_stddev': average_stddev
    }

    train_sizes = tuple(
        float(
            Decimal(value) * Decimal('0.05') + Decimal('0.1')
        ) for value in range(16)
    )

    train_errors = list()
    test_errors = list()

    for size in train_sizes:
        test_evaluator = RegressionEvaluator(labelCol='target')
        train_data_current, test_data_current = train_data.randomSplit(
            [size, 1-size], seed=UNIVERSAL_SEED
        )
        trained_model = model.fit(train_data_current)
        train_error = trained_model.stages[-1].avgMetrics[0]
        test_error = test_evaluator.evaluate(
            trained_model.transform(test_data_current)
        )
        logger.debug(f'Train score: {train_error}')
        logger.debug(f'Validation score: {test_error}')
        train_errors.append(train_error)
        test_errors.append(test_error)

    logger.info(f'Plotting learning curve for {estimator_name}')
    pyplot.plot(train_sizes, train_errors, '-o', label='Train')
    pyplot.plot(train_sizes, test_errors, '-o', label='Test')
    pyplot.title(f'Learning Curve for {estimator_name}')
    pyplot.xlabel('Training Set Size')
    pyplot.ylabel('RMSE Score')
    pyplot.legend(loc='best')
    pyplot.show()

    if test_pollution:
        pollution_results = list()
        for current_train_data in polluted_train_data:
            pollution_evaluator = RegressionEvaluator(labelCol='target')
            polluted_model = model.fit(current_train_data['data'])
            current_train_data['error'] = pollution_evaluator.evaluate(
                polluted_model.transform(test_data)
            )
            current_train_data.pop('data')
            pollution_results.append(current_train_data)
    estimator_results[estimator_name]['pollution_test'] = pollution_results
print(estimator_results)