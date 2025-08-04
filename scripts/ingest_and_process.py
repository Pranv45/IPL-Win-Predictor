import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lit, desc, window
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_spark_session():
    """Creates and returns a Spark session."""
    try:
        spark = SparkSession.builder \
            .appName("IPLDataProcessing") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        logging.info("Spark session created successfully.")
        return spark
    except Exception as e:
        logging.error(f"Error creating Spark session: {e}")
        raise

def process_ipl_data(spark, raw_matches_path, raw_balls_path, output_path):
    """
    Reads raw IPL data, processes it, engineers features, and saves the result.

    :param spark: The SparkSession object.
    :param raw_matches_path: Path to the raw matches CSV file.
    :param raw_balls_path: Path to the raw ball-by-ball CSV file.
    :param output_path: Path to save the processed Parquet file.
    """
    try:
        # --- 1. Load Raw Data ---
        logging.info(f"Loading raw matches data from {raw_matches_path}")
        matches_df = spark.read.csv(raw_matches_path, header=True, inferSchema=True)

        logging.info(f"Loading raw ball-by-ball data from {raw_balls_path}")
        balls_df = spark.read.csv(raw_balls_path, header=True, inferSchema=True)

        # --- 2. Data Cleaning & Preprocessing ---
        logging.info("Starting data cleaning and preprocessing...")

        # FIX: The column is named 'winner', not 'WinningTeam'. Update all references.
        # Standardize team names that have changed over the years
        matches_df = matches_df.withColumn("Team1",
            when(col("Team1") == "Rising Pune Supergiant", "Rising Pune Supergiants")
            .when(col("Team1") == "Delhi Daredevils", "Delhi Capitals")
            .when(col("Team1") == "Deccan Chargers", "Sunrisers Hyderabad")
            .otherwise(col("Team1")))
        matches_df = matches_df.withColumn("Team2",
            when(col("Team2") == "Rising Pune Supergiant", "Rising Pune Supergiants")
            .when(col("Team2") == "Delhi Daredevils", "Delhi Capitals")
            .when(col("Team2") == "Deccan Chargers", "Sunrisers Hyderabad")
            .otherwise(col("Team2")))
        matches_df = matches_df.withColumn("winner",
            when(col("winner") == "Rising Pune Supergiant", "Rising Pune Supergiants")
            .when(col("winner") == "Delhi Daredevils", "Delhi Capitals")
            .when(col("winner") == "Deccan Chargers", "Sunrisers Hyderabad")
            .otherwise(col("winner")))

        # --- 3. Feature Engineering ---
        logging.info("Starting feature engineering...")

        # Calculate total matches played by each team
        total_matches_team1 = matches_df.groupBy("Team1").agg(count("ID").alias("total_played"))
        total_matches_team2 = matches_df.groupBy("Team2").agg(count("ID").alias("total_played"))

        # Union the counts and sum them up
        total_matches = total_matches_team1.union(total_matches_team2.withColumnRenamed("Team2", "Team1")) \
            .groupBy("Team1") \
            .sum("total_played") \
            .withColumnRenamed("sum(total_played)", "total_matches_played") \
            .withColumnRenamed("Team1", "team")

        # Calculate total wins for each team
        total_wins = matches_df.groupBy("winner").agg(count("ID").alias("total_wins")) \
            .withColumnRenamed("winner", "team")

        # Combine total matches and wins to calculate win percentage
        win_percentage = total_matches.join(total_wins, "team", "left") \
            .withColumn("win_percentage", (col("total_wins") / col("total_matches_played")) * 100)

        win_percentage = win_percentage.na.fill(0) # Fill teams with 0 wins

        logging.info("Feature engineering complete.")

        # --- 4. Prepare Final Dataset ---
        # For our model, we want to join these features back to the original matches data
        # We'll create two instances of the win_percentage df to join for Team1 and Team2

        team1_stats = win_percentage.withColumnRenamed("team", "Team1") \
                                    .withColumnRenamed("win_percentage", "team1_win_percentage") \
                                    .select("Team1", "team1_win_percentage")

        team2_stats = win_percentage.withColumnRenamed("team", "Team2") \
                                    .withColumnRenamed("win_percentage", "team2_win_percentage") \
                                    .select("Team2", "team2_win_percentage")

        # Join the stats back to the main matches dataframe
        final_df = matches_df.join(team1_stats, "Team1", "left") \
                             .join(team2_stats, "Team2", "left")

        # Create the target variable: 1 if Team1 wins, 0 if Team2 wins
        # We exclude draws or no-result matches for this classification task
        final_df = final_df.filter(col("winner").isNotNull())
        final_df = final_df.withColumn("winner", when(col("winner") == col("Team1"), 1).otherwise(0))

        logging.info("Final dataset prepared.")

        # --- 5. Save Processed Data ---
        logging.info(f"Saving processed data to {output_path}")
        final_df.write.mode("overwrite").parquet(output_path)
        logging.info("Data processing complete and saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")
        raise

if __name__ == "__main__":
    # Define file paths based on our project structure
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_matches_path = os.path.join(project_root, "data/raw/matches.csv")
    raw_balls_path = os.path.join(project_root, "data/raw/deliveries.csv")
    processed_output_path = os.path.join(project_root, "data/processed/processed_ipl_data.parquet")

    # Create Spark Session
    spark = create_spark_session()

    # Run the processing function
    process_ipl_data(spark, raw_matches_path, raw_balls_path, processed_output_path)

    # Stop the Spark session
    spark.stop()
