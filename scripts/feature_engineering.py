from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import StringIndexer
import yaml
import logging
import os
from datetime import datetime
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_spark_session():
    """Create or get Spark session for feature engineering."""
    spark = (
        SparkSession.builder.appName("IPLFeatureEngineering")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # Disable Arrow for Java 21 compatibility
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_config():
    """Load model configuration."""
    with open('configs/model_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def engineer_features(spark, df):
    """Engineer advanced features for the IPL win prediction model using Spark."""
    config = load_config()

    logging.info("Starting advanced feature engineering with Spark...")

    # Add row number for ordering (assuming data is already sorted by date/match_id)
    # If not sorted, we'll need to sort by date first
    if 'date' in df.columns:
        # Try to convert date to proper format if it's a string
        try:
            df = df.withColumn('date_parsed', F.to_date('date', 'yyyy-MM-dd'))
            df = df.drop('date').withColumnRenamed('date_parsed', 'date')
        except:
            # If date is already in correct format or can't parse, keep as is
            pass
        df = df.orderBy('date', 'id')

    # Add a row number for window functions using row_number() for proper ordering
    window_order = Window.orderBy('date', 'id')
    df = df.withColumn('row_num', F.row_number().over(window_order))

    # 1. Recent Form (last 5 matches) using Spark Window Functions
    logging.info("Calculating recent form features using Spark window functions...")

    # Window for team1 recent form (last 5 matches)
    window_team1 = Window.partitionBy('Team1').orderBy('row_num').rowsBetween(-4, 0)
    # Calculate rolling mean - winner == Team1 (encoded as 1, else 0)
    df = df.withColumn(
        'winner_team1_flag',
        F.when(F.col('winner') == F.col('Team1'), 1.0).otherwise(0.0)
    )
    df = df.withColumn(
        'team1_recent_form',
        F.avg('winner_team1_flag').over(window_team1)
    )

    # Window for team2 recent form (last 5 matches)
    window_team2 = Window.partitionBy('Team2').orderBy('row_num').rowsBetween(-4, 0)
    df = df.withColumn(
        'winner_team2_flag',
        F.when(F.col('winner') == F.col('Team2'), 1.0).otherwise(0.0)
    )
    df = df.withColumn(
        'team2_recent_form',
        F.avg('winner_team2_flag').over(window_team2)
    )

    # 2. Head-to-Head Performance
    logging.info("Calculating head-to-head features using Spark aggregations...")
    h2h_stats = df.groupBy('Team1', 'Team2').agg(
        F.count('*').alias('h2h_matches'),
        F.sum(F.when(F.col('winner') == F.col('Team1'), 1).otherwise(0)).alias('h2h_wins_team1')
    )
    h2h_stats = h2h_stats.withColumn(
        'h2h_win_rate_team1',
        F.col('h2h_wins_team1') / F.col('h2h_matches')
    )

    # Merge head-to-head stats
    df = df.join(h2h_stats, on=['Team1', 'Team2'], how='left')

    # Create team2 head-to-head stats (swap Team1 and Team2)
    h2h_stats_team2 = h2h_stats.select(
        F.col('Team2').alias('Team1'),
        F.col('Team1').alias('Team2'),
        F.col('h2h_matches').alias('h2h_matches_swap'),
        F.col('h2h_wins_team1').alias('h2h_wins_swap'),
        (1 - F.col('h2h_win_rate_team1')).alias('h2h_win_rate_team2')
    )

    df = df.join(
        h2h_stats_team2.select('Team1', 'Team2', 'h2h_win_rate_team2'),
        on=['Team1', 'Team2'],
        how='left'
    )

    # 3. Venue Performance
    logging.info("Calculating venue performance features using Spark aggregations...")
    venue_stats = df.groupBy('Team1', 'Venue').agg(
        F.count('*').alias('venue_matches_team1'),
        F.sum(F.when(F.col('winner') == F.col('Team1'), 1).otherwise(0)).alias('venue_wins_team1')
    )
    venue_stats = venue_stats.withColumn(
        'venue_win_rate_team1',
        F.col('venue_wins_team1') / F.col('venue_matches_team1')
    )

    df = df.join(venue_stats, on=['Team1', 'Venue'], how='left')

    # 4. Season Performance
    logging.info("Calculating season performance features using Spark aggregations...")
    # Extract season from date (handle both string and date types)
    try:
        df = df.withColumn('Season', F.year(F.col('date')))
    except:
        # If date is a string, try parsing it
        df = df.withColumn('Season', F.year(F.to_date('date', 'yyyy-MM-dd')))

    season_stats = df.groupBy('Team1', 'Season').agg(
        F.count('*').alias('season_matches_team1'),
        F.sum(F.when(F.col('winner') == F.col('Team1'), 1).otherwise(0)).alias('season_wins_team1')
    )
    season_stats = season_stats.withColumn(
        'season_win_rate_team1',
        F.col('season_wins_team1') / F.col('season_matches_team1')
    )

    df = df.join(season_stats, on=['Team1', 'Season'], how='left')

    # 5. Categorical Encoding using StringIndexer
    logging.info("Encoding categorical features using Spark StringIndexer...")
    label_encoders = {}
    categorical_features = config['features']['categorical_features']

    for feature in categorical_features:
        if feature in df.columns:
            # Get distinct values to create mapping
            distinct_values = df.select(feature).distinct().orderBy(feature).collect()
            value_to_index = {str(row[feature]): idx for idx, row in enumerate(distinct_values)}

            # Create encoded column
            def encode_value(value):
                return value_to_index.get(str(value), -1)

            encode_udf = F.udf(encode_value, IntegerType())
            df = df.withColumn(f'{feature}_encoded', encode_udf(F.col(feature)))

            # Store mapping for saving
            label_encoders[feature] = value_to_index

    # 6. Fill missing values with median
    logging.info("Filling missing values using Spark aggregations...")
    numerical_features = config['features']['numerical_features']

    for feature in numerical_features:
        if feature in df.columns:
            # Calculate median using percentile_approx
            median_value = df.select(
                F.expr(f'percentile_approx({feature}, 0.5)').alias('median')
            ).collect()[0]['median']

            if median_value is not None:
                df = df.fillna({feature: median_value})

    # 7. Create interaction features
    logging.info("Creating interaction features...")

    # Validate required columns exist before creating interactions
    required_cols = ['team1_win_percentage', 'team2_win_percentage', 'team1_recent_form', 'team2_recent_form']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for interaction features: {missing_cols}")

    df = df.withColumn(
        'win_percentage_diff',
        F.col('team1_win_percentage') - F.col('team2_win_percentage')
    )
    df = df.withColumn(
        'recent_form_diff',
        F.col('team1_recent_form') - F.col('team2_recent_form')
    )
    df = df.withColumn(
        'h2h_advantage',
        F.coalesce(F.col('h2h_win_rate_team1'), F.lit(0.0)) -
        F.coalesce(F.col('h2h_win_rate_team2'), F.lit(0.0))
    )

    # Drop temporary columns
    df = df.drop('row_num', 'winner_team1_flag', 'winner_team2_flag')

    # Get column names for feature importance
    columns = df.columns

    # Save feature importance metrics
    feature_importance = {
        'features_created': columns,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'total_features': len(columns),
        'timestamp': datetime.now().isoformat()
    }

    os.makedirs('metrics', exist_ok=True)
    with open('metrics/feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)

    logging.info(f"Feature engineering complete. Created {len(columns)} features.")

    return df, label_encoders

def save_features(df, label_encoders, output_dir):
    """Save engineered features using Spark."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to pandas for single-file parquet output (compatible with downstream scripts)
    logging.info("Converting Spark DataFrame to Pandas for saving...")
    df_pandas = df.toPandas()

    # Save as parquet
    df_pandas.to_parquet(f'{output_dir}/engineered_features.parquet', index=False)

    # Save label encoders
    with open(f'{output_dir}/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    logging.info(f"Features saved to {output_dir}")

if __name__ == "__main__":
    spark = get_spark_session()

    try:
        # Load processed data using Spark
        processed_data_path = "data/processed/processed_ipl_data.parquet"
        logging.info(f"Loading processed data from {processed_data_path} using Spark...")
        df = spark.read.parquet(processed_data_path)
        logging.info(f"Loaded data with {df.count()} rows and {len(df.columns)} columns")

        # Engineer features
        df_engineered, label_encoders = engineer_features(spark, df)

        # Save features
        save_features(df_engineered, label_encoders, "data/features")

        logging.info("Feature engineering pipeline completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred during feature engineering: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")