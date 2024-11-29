from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType

def clean_user_data(df: DataFrame) -> DataFrame:
    """
    Clean the user data by adding a calculated friend count, 
    selecting relevant columns, and filtering rows based on criteria.
    """
    df = df.withColumn("friend_count", F.size(F.split(F.col("friends"), ", ")))

    cleaned_df = df.select(
        F.col("user_id"),
        F.col("name"),
        F.col("review_count").cast(IntegerType()),
        F.col("friend_count")
    ).dropna(subset=["user_id", "name", "review_count", "friend_count"]) \
     .filter((F.col("review_count") > 20) & (F.col("friend_count") >= 10))

    return cleaned_df

def transform_user_data(df: DataFrame) -> DataFrame:
    """
    Transform the user data by adding an activity score and classifying users.
    """
    df = df.withColumn(
        "activity_score", F.log(F.col("review_count") + 1) + F.log(F.col("friend_count") + 1)
    )
    
    df = df.withColumn(
        "user_tier", 
        F.when(F.col("activity_score") > 4, "High") \
         .when((F.col("activity_score") <= 4) & (F.col("activity_score") > 2), "Medium") \
         .otherwise("Low")
    )

    return df

def aggregate_user_data(df: DataFrame) -> DataFrame:
    """
    Aggregate user data to compute statistics by user tier.
    """
    aggregated_df = df.groupBy("user_tier").agg(
        F.count("*").alias("user_count"),
        F.avg("review_count").alias("avg_review_count"),
        F.avg("friend_count").alias("avg_friend_count"),
        F.avg("activity_score").alias("avg_activity_score")
    )
    return aggregated_df

def save_user_json(df: DataFrame, s3_path: str) -> None:
    """
    Save the processed user data as a JSON file to S3.
    """
    df.write.format("json").save(s3_path)
