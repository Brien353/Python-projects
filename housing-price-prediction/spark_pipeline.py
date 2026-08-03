import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, max, min

# 1. Force PySpark to use the exact Python executable from your .venv
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 2. Set Java options for JDK 25 compatibility
java_opts = (
    '--add-opens=java.base/java.lang=ALL-UNNAMED '
    '--add-opens=java.base/java.util=ALL-UNNAMED '
    '--add-opens=java.base/java.util.concurrent=ALL-UNNAMED '
    '--add-opens=java.base/sun.nio.ch=ALL-UNNAMED'
)
os.environ['JAVA_OPTS'] = java_opts

# 3. Initialize PySpark Session
spark = (
    SparkSession.builder
    .appName("HousingPricePipeline")
    .master("local[1]")
    .config("spark.driver.extraJavaOptions", java_opts)
    .config("spark.executor.extraJavaOptions", java_opts)
    .getOrCreate()
)

# Suppress log noise
spark.sparkContext.setLogLevel("ERROR")

print("\n--- PySpark Session Successfully Created! ---\n")

csv_filename = "data/raw/Housing.csv"

try:
    # 4. Load Dataset
    df = spark.read.csv(
        csv_filename,
        header=True,
        inferSchema=True
    )

    print("Real Dataset Schema:")
    df.printSchema()

    print("\nFirst 5 Rows:")
    df.show(5)

    # 5. Data Transformations & Aggregations
    print("\n--- Average Price & House Count by Furnishing Status ---")
    df.groupBy("furnishingstatus") \
      .agg(
          count("*").alias("total_houses"),
          avg("price").alias("avg_price"),
          avg("area").alias("avg_area")
      ) \
      .orderBy(col("avg_price").desc()) \
      .show()

    print("\n--- Average Price by Bedrooms & Air Conditioning ---")
    df.groupBy("bedrooms", "airconditioning") \
      .agg(avg("price").alias("avg_price")) \
      .orderBy("bedrooms", "airconditioning") \
      .show()

finally:
    spark.stop()
    print("\n--- Spark Session Closed Cleanly ---")