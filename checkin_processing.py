from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import TimestampType

def clean_checkIn_data(df: DataFrame) -> DataFrame:
    """
        Clean the checkIn data by splitting timestamps and extracting relevant fields.
    """
    df = df.withColumn("timestamp", F.explode(F.split(F.col("date"), ", ")))
    
    df = df.withColumn("timestamp", F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss"))
    df = df.withColumn("check_in_date", F.to_date(F.col("timestamp")))
    df = df.withColumn("check_in_hour", F.hour(F.col("timestamp")))
    df = df.drop("date")
    
    return df

def transform_checkIn_data(df: DataFrame) -> DataFrame:
    """
        Transform the checkIn data to derive useful insights.
    """
    check_in_counts = df.groupBy("business_id").count().withColumnRenamed("count", "total_check_ins")

    monthly_trends = df.withColumn("check_in_month", F.month(F.col("check_in_date"))) \
                       .groupBy("business_id", "check_in_month").count() \
                       .withColumnRenamed("count", "monthly_check_ins")

    hourly_trends = df.groupBy("business_id", "check_in_hour").count() \
                      .withColumnRenamed("count", "hourly_check_ins")
    
    return check_in_counts, monthly_trends, hourly_trends

def save_checkIn_json(df: DataFrame, s3_path: str) -> None:
    """
        Save the processed review data as a JSON file to S3.
    """
    df.write.format("json").save(s3_path)
