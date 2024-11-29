from pyspark.sql import functions
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import FloatType

def clean_business_data(df: DataFrame) -> DataFrame:
    """
    Clean the business data by removing rows with missing values and transforming.
    """
    
    cleaned_df = df \
        .dropna("any", subset=["business_id", "name", "categories", "review_count", "latitude", "longitude", "hours"]) \
        .filter(functions.col("is_open") == 1) \
        .filter(functions.col("review_count") > 100) \
        .withColumn("categories", functions.split(functions.col("categories"), ", "))
    
    cleaned_df = cleaned_df.withColumn("main_category", functions.expr("categories[0]"))
    cleaned_df = cleaned_df.withColumn("popularity_score", functions.col("review_count") * functions.col("stars"))
    cleaned_df = cleaned_df.withColumn(
        "suggestion_probability",
        functions.col("stars") + functions.col("review_count") +
        functions.size(functions.split(functions.col("categories"), ", "))
    )
    return cleaned_df


def transform_business_data(df: DataFrame) -> DataFrame:
    """
    Transform the business data by adding new calculated columns.
    """

    # Add a rank column for businesses by popularity within the same city using window functions
    # To do that we can sort businesses in same city
    window_spec = Window.partitionBy("city").orderBy(functions.desc("popularity_score"))
    df = df.withColumn("popularity_rank_city", functions.rank().over(window_spec))
    
    # Normalize popularity score between 0 and 1 (optional, for ML purposes)
    # Which make things more easier for the machine learning project
    min_max_window = Window.partitionBy()
    df = df.withColumn("min_popularity", functions.min("popularity_score").over(min_max_window)) \
           .withColumn("max_popularity", functions.max("popularity_score").over(min_max_window)) \
           .withColumn(
               "normalized_popularity",
               ((functions.col("popularity_score") - functions.col("min_popularity")) /
                (functions.col("max_popularity") - functions.col("min_popularity"))).cast(FloatType())
           ).drop("min_popularity", "max_popularity")
    
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
        functions.sum("review_count").alias("total_reviews")
    )


def save_business_json(df: DataFrame, s3_path: str) -> None:
    """
    Save the processed business data as a JSON file to S3.
    """
    try:
        df.write.format("json").mode("overwrite").save(s3_path)
    except Exception as e:
        print(f"Failed to save data to {s3_path}: {str(e)}")
