from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Start Spark session
spark = SparkSession.builder \
    .appName("Healthcare Readmission Pipeline") \
    .getOrCreate()

# Load dataset
df = spark.read.csv("data/raw/healthcare_readmission.csv", header=True, inferSchema=True)

print("Initial Data:")
df.show(5)

# Basic cleaning (example transformations)
df_clean = df.withColumn(
    "high_risk_flag",
    when(
        (col("num_prior_visits") > 5) |
        (col("length_of_stay") > 7) |
        (col("has_diabetes") == 1),
        1
    ).otherwise(0)
)

# Feature engineering
df_features = df_clean.withColumn(
    "age_group",
    when(col("age") < 30, "Young")
    .when(col("age") < 60, "Adult")
    .otherwise("Senior")
)

print("Transformed Data:")
df_features.show(5)

# Save processed data
df_features.toPandas().to_csv("data/processed/healthcare_processed.csv", index=False)

print("ETL pipeline completed successfully.")