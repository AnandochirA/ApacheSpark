from pyspark.sql import SparkSession, functions, DataFrame, Window
from pyspark.sql.types import FloatType, IntegerType

def clean_business_data(df: DataFrame) -> DataFrame:
    """
    Cleans business data by removing missing values, filtering, and extracting relevant information.

    Args:
        df: Input DataFrame containing business data.

    Returns:
        Cleaned DataFrame.
    """

    cleaned_df = (
        df.dropna(subset=["business_id", "name", "categories", "review_count", "latitude", "longitude", "hours"])
        .filter("is_open = 1 AND review_count > 100")
        .withColumn("categories", functions.split("categories", ","))
        .withColumn("main_category", functions.col("categories")[0])
        .withColumn("popularity_score", functions.col("review_count") * functions.col("stars"))
        .withColumn(
            "suggestion_probability",
            functions.col("stars") + functions.col("review_count") + functions.size("categories"),
        )
    )
    return cleaned_df

def transform_business_data(df: DataFrame) -> DataFrame:
    """
    Transform the business data by adding new calculated columns.
    """
    # Add a rank column for businesses by popularity within the same city using window functions
    window_spec = Window.partitionBy("city").orderBy(functions.desc("popularity_score"))
    df = df.withColumn("popularity_rank_city", functions.rank().over(window_spec))
    
    # Normalize popularity score globally
    # Avoid using a single partition; instead, calculate globally but efficiently
    global_stats = df.select(
        functions.min("popularity_score").alias("min_popularity"),
        functions.max("popularity_score").alias("max_popularity")
    ).collect()[0]
    
    min_popularity = global_stats["min_popularity"]
    max_popularity = global_stats["max_popularity"]
    
    # Add normalized popularity score using broadcasted global stats
    df = df.withColumn(
        "normalized_popularity",
        ((functions.col("popularity_score") - functions.lit(min_popularity)) /
         (functions.lit(max_popularity) - functions.lit(min_popularity))).cast(FloatType())
    )
    
    # Drop unwanted columns
    df = df.drop("address", 'attributes', "business_id", "hours", "is_open", "review_count")
    return df

def aggregate_business_data_city(df: DataFrame) -> DataFrame:
    """
    Aggregate business data with multiple metrics by city.
    """
    # Aggregate number of businesses, average rating, and total reviews by city
    return df.groupBy("city").agg(
        functions.count("business_id").alias("num_businesses"),
        functions.avg("stars").alias("average_rating"),
        functions.sum("review_count").alias("total_reviews"),
        functions.max("popularity_score").alias("max_popularity_score"),
        functions.avg("popularity_score").alias("avg_popularity_score")
    )

def aggregate_business_data_mainCategory(df: DataFrame) -> DataFrame:
    """
    Count the number of businesses by main category and calculate average rating.
    """
    # Group by main category and compute metrics
    return df.groupBy("main_category").agg(
        functions.count("business_id").alias("num_businesses"),
        functions.avg("stars").alias("average_rating"),
        functions.sum("review_count").alias("total_reviews"))

def clean_checkIn_data(df: DataFrame) -> DataFrame:
    """
    Clean the checkIn data by splitting timestamps and extracting relevant fields.
    """
    df = df.withColumn("timestamp", functions.explode(functions.split(functions.col("date"), ", ")))
    df = df.withColumn("timestamp", functions.to_timestamp(functions.col("timestamp"), "yyyy-MM-dd HH:mm:ss"))
    df = df.withColumn("check_in_date", functions.to_date(functions.col("timestamp")))
    df = df.withColumn("check_in_hour", functions.hour(functions.col("timestamp")))
    df = df.drop("date")
    return df

def transform_checkIn_data(df: DataFrame) -> DataFrame:
    """
    Transform the checkIn data to derive useful insights using advanced PySpark methods.
    """
    check_in_counts = df.groupBy("business_id").count().withColumnRenamed("count", "total_check_ins")
    # Monthly trends with percentage contribution of check-ins by month
    monthly_window = Window.partitionBy("business_id")
    monthly_trends = (
        df.withColumn("check_in_month", functions.month(functions.col("check_in_date")))
          .groupBy("business_id", "check_in_month").count()
          .withColumnRenamed("count", "monthly_check_ins")
          .withColumn("monthly_percentage", 
                      (functions.col("monthly_check_ins") / functions.sum("monthly_check_ins").over(monthly_window) * 100).cast(FloatType()))
    )
    # Hourly trends with peak hours identified for each business
    hourly_window = Window.partitionBy("business_id").orderBy(functions.desc("hourly_check_ins"))
    hourly_trends = (
        df.groupBy("business_id", "check_in_hour").count()
          .withColumnRenamed("count", "hourly_check_ins")
          .withColumn("rank", functions.rank().over(hourly_window))
          .filter(functions.col("rank") == 1) 
          .drop("rank")
    )
    # Peak day of the week for each business
    df = df.withColumn("day_of_week", functions.date_format(functions.col("check_in_date"), "EEEE"))
    day_trends = (
        df.groupBy("business_id", "day_of_week").count()
          .withColumnRenamed("count", "daily_check_ins")
          .withColumn("peak_day", 
                      functions.when(functions.row_number().over(Window.partitionBy("business_id").orderBy(functions.desc("daily_check_ins"))) == 1, 
                             functions.col("day_of_week")))
    )
    return check_in_counts, monthly_trends, hourly_trends, day_trends

def clean_user_data(df: DataFrame) -> DataFrame:
    """
    Clean the user data by adding a calculated friend count, 
    selecting relevant columns, and filtering rows based on criteria.
    """
    df = df.withColumn("friend_count", functions.size(functions.split(functions.col("friends"), ", ")))

    cleaned_df = df.select(
        functions.col("user_id"),
        functions.col("name"),
        functions.col("review_count").cast(IntegerType()),
        functions.col("friend_count")
    ).dropna(subset=["user_id", "name", "review_count", "friend_count"]) \
     .filter((functions.col("review_count") > 20) & (functions.col("friend_count") >= 10))

    return cleaned_df

def transform_user_data(df: DataFrame) -> DataFrame:
    """
    Transform the user data by adding an activity score and classifying users.
    """
    df = df.withColumn(
        "activity_score", functions.log(functions.col("review_count") + 1) + functions.log(functions.col("friend_count") + 1)
    )
    
    df = df.withColumn(
        "user_tier", 
        functions.when(functions.col("activity_score") > 4, "High") \
         .when((functions.col("activity_score") <= 4) & (functions.col("activity_score") > 2), "Medium") \
         .otherwise("Low")
    )

    return df

def aggregate_user_data(df: DataFrame) -> DataFrame:
    """
    Aggregate user data to compute statistics by user tier.
    """
    aggregated_df = df.groupBy("user_tier").agg(
        functions.count("*").alias("user_count"),
        functions.avg("review_count").alias("avg_review_count"),
        functions.avg("friend_count").alias("avg_friend_count"),
        functions.avg("activity_score").alias("avg_activity_score")
    )
    return aggregated_df

def main():
    spark = SparkSession.builder.appName("Yelp Data Processing").getOrCreate()

    #input_path = r"D:\yelp_dataset\yelp_academic_dataset_business.json"
    input_path = "s3://apache-spark-anand/yelp_academic_dataset_business.json"
    business_df = spark.read.json(input_path)

    cleaned_business_df = clean_business_data(business_df)
    cleaned_business_df.write.mode("overwrite").json("s3://processed-yelp-data/business.json")


    tranformed_business_df = transform_business_data(cleaned_business_df)
    tranformed_business_df.write.mode("append").json("s3://processed-yelp-data/business.json")

    aggregate_business_df = aggregate_business_data_city(cleaned_business_df)
    aggregate_business_df.write.mode("append").json("s3://processed-yelp-data/business.json")

    aggregate_business_df_main = aggregate_business_data_mainCategory(cleaned_business_df)
    aggregate_business_df_main.write.mode("append").json("s3://processed-yelp-data/business.json")
    
    #input_path_checkin = r"D:\yelp_dataset\yelp_academic_dataset_checkin.json"
    input_path_checkin = "s3://apache-spark-anand/yelp_academic_dataset_checkin.json"
    checkin_df = spark.read.json(input_path_checkin)

    cleaned_checkin_df = clean_checkIn_data(checkin_df)
    cleaned_checkin_df.write.mode("overwrite").json("s3://processed-yelp-data/checkin.json")

    check_in_counts, monthly_trends, hourly_trends, day_trends = transform_checkIn_data(cleaned_checkin_df)
    check_in_counts.write.mode("append").json("s3://processed-yelp-data/checkin.json")
    monthly_trends.write.mode("append").json("s3://processed-yelp-data/checkin.json")
    hourly_trends.write.mode("append").json("s3://processed-yelp-data/checkin.json")
    day_trends.write.mode("append").json("s3://processed-yelp-data/checkin.json")

    input_path_user = "s3://apache-spark-anand/yelp_academic_dataset_user.json"
    user_df = spark.read.json(input_path_user)

    cleaned_user_df = clean_user_data(user_df)
    cleaned_user_df.write.mode("overwrite").json("s3://processed-yelp-data/user.json")

    transformed_user_df = transform_user_data(cleaned_user_df)
    transformed_user_df.write.mode("append").json("s3://processed-yelp-data/user.json")

    aggregated_user_df = aggregate_user_data(transformed_user_df)
    aggregated_user_df.write.mode("append").json("s3://processed-yelp-data/user.json")

    spark.stop()

if __name__ == "__main__":
    main()
