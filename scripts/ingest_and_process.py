from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, count, lit, udf
from pyspark.sql.types import StringType
import logging
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def get_spark_session():
    """Create and configure Spark session for distributed data processing."""
    spark = (
        SparkSession.builder.appName("IPLDataProcessing")
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

def load_data(spark, matches_path: str, deliveries_path: str):
    """Load raw IPL data using Spark."""
    logging.info(f"Loading raw matches data from {matches_path} using Spark...")
    matches_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(matches_path)
    )

    logging.info(f"Loading raw ball-by-ball data from {deliveries_path} using Spark...")
    deliveries_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(deliveries_path)
    )

    matches_count = matches_df.count()
    deliveries_count = deliveries_df.count()

    logging.info(f"Loaded with Spark - Matches: {matches_count} rows, Deliveries: {deliveries_count} rows")

    return matches_df, deliveries_df

def clean_team_names(df):
    """Standardize team names across datasets using Spark UDF."""
    logging.info("Cleaning team names using Spark UDF...")

    # Define team name mappings
    team_mappings = {
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Mumbai Indians': 'Mumbai Indians',
        'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
        'Chennai Super Kings': 'Chennai Super Kings',
        'Kolkata Knight Riders': 'Kolkata Knight Riders',
        'Rajasthan Royals': 'Rajasthan Royals',
        'Sunrisers Hyderabad': 'Sunrisers Hyderabad',
        'Gujarat Lions': 'Gujarat Lions',
        'Pune Warriors': 'Pune Warriors',
        'Kochi Tuskers Kerala': 'Kochi Tuskers Kerala',
        'Deccan Chargers': 'Deccan Chargers',
        'Rising Pune Supergiants': 'Rising Pune Supergiants'
    }

    # Create UDF for team name mapping (more memory efficient than nested when/otherwise)
    def map_team_name(team_name):
        if team_name is None:
            return None
        return team_mappings.get(team_name, team_name)

    map_team_udf = udf(map_team_name, StringType())

    # Apply UDF to each team column
    team_columns = ['team1', 'team2', 'winner', 'batting_team', 'bowling_team']

    for col_name in team_columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, map_team_udf(col(col_name)))

    return df

def calculate_win_percentages(matches_df):
    """Calculate win percentages for each team using Spark aggregations."""
    logging.info("Calculating win percentages using Spark aggregations...")

    from pyspark.sql.functions import sum as spark_sum, count as spark_count, when, col, lit, coalesce

    # Calculate matches and wins for team1
    team1_stats = matches_df.groupBy("team1").agg(
        spark_count(lit(1)).alias("matches_team1"),
        spark_sum(when(col("winner") == col("team1"), 1).otherwise(0)).alias("wins_team1")
    ).withColumnRenamed("team1", "team")

    # Calculate matches and wins for team2
    team2_stats = matches_df.groupBy("team2").agg(
        spark_count(lit(1)).alias("matches_team2"),
        spark_sum(when(col("winner") == col("team2"), 1).otherwise(0)).alias("wins_team2")
    ).withColumnRenamed("team2", "team")

    # Combine and calculate win percentage
    team_stats = team1_stats.join(team2_stats, on="team", how="full_outer").fillna(0)
    team_stats = team_stats.withColumn(
        "total_matches",
        coalesce(col("matches_team1"), lit(0)) + coalesce(col("matches_team2"), lit(0))
    ).withColumn(
        "total_wins",
        coalesce(col("wins_team1"), lit(0)) + coalesce(col("wins_team2"), lit(0))
    ).withColumn(
        "win_percentage",
        when(col("total_matches") > 0, col("total_wins") / col("total_matches")).otherwise(0.0)
    ).select("team", "win_percentage")

    return team_stats

def engineer_features(matches_df, deliveries_df):
    """Engineer features for the dataset using Spark."""
    logging.info("Starting feature engineering with Spark...")

    # Calculate team statistics
    team_stats = calculate_win_percentages(matches_df)

    # Join matches with team stats for team1
    features_df = matches_df.join(
        team_stats.alias("team1_stats"),
        matches_df.team1 == col("team1_stats.team"),
        "left"
    ).withColumnRenamed("win_percentage", "team1_win_percentage")

    # Join matches with team stats for team2
    features_df = features_df.join(
        team_stats.alias("team2_stats"),
        matches_df.team2 == col("team2_stats.team"),
        "left"
    ).withColumnRenamed("win_percentage", "team2_win_percentage")

    # Create winner column (1 if team1 wins, 0 otherwise)
    features_df = features_df.withColumn(
        "winner",
        when(col("winner") == col("team1"), 1).otherwise(0)
    )

    # Select and rename columns
    features_df = features_df.select(
        col("id"),
        col("date"),
        col("team1").alias("Team1"),
        col("team2").alias("Team2"),
        col("city").alias("City"),
        col("venue").alias("Venue"),
        col("team1_win_percentage"),
        col("team2_win_percentage"),
        col("winner")
    )

    # Fill null values
    features_df = features_df.fillna(0.0, subset=["team1_win_percentage", "team2_win_percentage"])

    # Cache the DataFrame before counting to avoid recomputation
    features_df.cache()
    match_count = features_df.count()
    logging.info(f"Feature engineering complete. Processed {match_count} matches.")
    features_df.unpersist()  # Release cache after use

    return features_df

def generate_metrics(final_df, matches_df, deliveries_df):
    """Generate data quality and processing statistics metrics using Spark."""
    logging.info("Generating metrics using Spark...")

    os.makedirs("metrics", exist_ok=True)

    from pyspark.sql.functions import countDistinct, sum as spark_sum

    # Calculate data quality metrics
    matches_count = matches_df.count()
    deliveries_count = deliveries_df.count()

    # Unique teams
    unique_teams_team1 = matches_df.select("team1").distinct()
    unique_teams_team2 = matches_df.select("team2").distinct()
    unique_teams = unique_teams_team1.union(unique_teams_team2).distinct().count()

    # Unique venues and cities
    unique_venues = matches_df.select("venue").distinct().count()
    unique_cities = matches_df.select("city").distinct().count()

    # Missing values count - only check isnan() for numeric columns
    matches_columns = matches_df.columns
    deliveries_columns = deliveries_df.columns

    # Get schema to check column types
    matches_schema = {f.name: str(f.dataType) for f in matches_df.schema.fields}
    deliveries_schema = {f.name: str(f.dataType) for f in deliveries_df.schema.fields}

    # Numeric types in Spark
    numeric_types = ['IntegerType', 'LongType', 'DoubleType', 'FloatType', 'DecimalType', 'ShortType', 'ByteType']

    def count_nulls(df, columns, schema):
        """Count null and NaN values, only checking isnan for numeric columns."""
        total = 0
        for col_name in columns:
            col_expr = col(col_name).isNull()
            # Only check isnan for numeric columns
            if col_name in schema and any(numeric_type in schema[col_name] for numeric_type in numeric_types):
                col_expr = col_expr | isnan(col_name)
            total += df.filter(col_expr).count()
        return total

    matches_null_count = count_nulls(matches_df, matches_columns, matches_schema)
    deliveries_null_count = count_nulls(deliveries_df, deliveries_columns, deliveries_schema)

    # Data completeness
    matches_total_cells = matches_count * len(matches_columns)
    deliveries_total_cells = deliveries_count * len(deliveries_columns)

    matches_completeness = (1 - matches_null_count / matches_total_cells) * 100 if matches_total_cells > 0 else 100
    deliveries_completeness = (1 - deliveries_null_count / deliveries_total_cells) * 100 if deliveries_total_cells > 0 else 100

    data_quality = {
        "total_matches": matches_count,
        "total_deliveries": deliveries_count,
        "unique_teams": unique_teams,
        "unique_venues": unique_venues,
        "unique_cities": unique_cities,
        "missing_values": {
            "matches": matches_null_count,
            "deliveries": deliveries_null_count
        },
        "data_completeness": {
            "matches": matches_completeness,
            "deliveries": deliveries_completeness
        }
    }

    # Processing statistics
    processed_count = final_df.count()
    feature_columns = final_df.columns

    # Calculate statistics for win percentages
    from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max

    team1_stats = final_df.agg(
        mean("team1_win_percentage").alias("mean"),
        stddev("team1_win_percentage").alias("std"),
        spark_min("team1_win_percentage").alias("min"),
        spark_max("team1_win_percentage").alias("max")
    ).collect()[0]

    team2_stats = final_df.agg(
        mean("team2_win_percentage").alias("mean"),
        stddev("team2_win_percentage").alias("std"),
        spark_min("team2_win_percentage").alias("min"),
        spark_max("team2_win_percentage").alias("max")
    ).collect()[0]

    # Target distribution
    winner_dist = final_df.groupBy("winner").count().collect()
    winner_dict = {row["winner"]: row["count"] for row in winner_dist}

    processing_stats = {
        "processed_matches": processed_count,
        "features_created": len(feature_columns),
        "target_distribution": winner_dict,
        "feature_statistics": {
            "team1_win_percentage": {
                "mean": float(team1_stats["mean"] or 0),
                "std": float(team1_stats["std"] or 0),
                "min": float(team1_stats["min"] or 0),
                "max": float(team1_stats["max"] or 0)
            },
            "team2_win_percentage": {
                "mean": float(team2_stats["mean"] or 0),
                "std": float(team2_stats["std"] or 0),
                "min": float(team2_stats["min"] or 0),
                "max": float(team2_stats["max"] or 0)
            }
        }
    }

    # Save metrics
    with open("metrics/data_quality.json", "w") as f:
        json.dump(data_quality, f, indent=2)

    with open("metrics/processing_stats.json", "w") as f:
        json.dump(processing_stats, f, indent=2)

    logging.info("Metrics generated and saved successfully.")

def process_ipl_data(matches_path: str, deliveries_path: str, output_path: str):
    """Main function to process IPL data using Spark."""
    spark = None
    try:
        # Initialize Spark
        spark = get_spark_session()

        # Load data
        matches_df, deliveries_df = load_data(spark, matches_path, deliveries_path)

        # Clean data
        logging.info("Starting data cleaning and preprocessing with Spark...")
        matches_df = clean_team_names(matches_df)
        deliveries_df = clean_team_names(deliveries_df)

        # Engineer features
        final_df = engineer_features(matches_df, deliveries_df)

        # Generate metrics
        generate_metrics(final_df, matches_df, deliveries_df)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to pandas for single-file parquet output (for compatibility with downstream pandas scripts)
        # All processing was done in Spark - this is just for the final write
        logging.info("Converting Spark DataFrame to pandas for parquet output...")
        final_pandas_df = final_df.toPandas()

        # Save processed data
        logging.info(f"Saving processed data to {output_path}...")
        final_pandas_df.to_parquet(output_path, index=False)

        logging.info("Data processing complete and saved successfully using Spark.")

    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")
        raise
    finally:
        if spark:
            spark.stop()
            logging.info("Spark session stopped.")

if __name__ == "__main__":
    # Define paths
    raw_matches_path = "data/raw/matches.csv"
    raw_balls_path = "data/raw/deliveries.csv"
    processed_output_path = "data/processed/processed_ipl_data.parquet"

    # Process data
    process_ipl_data(raw_matches_path, raw_balls_path, processed_output_path)
