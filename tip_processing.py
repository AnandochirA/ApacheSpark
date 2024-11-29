from pyspark.sql import functions
from pyspark.sql import DataFrame
from textblob import TextBlob
from pyspark.sql.types import IntegerType, ArrayType, StringType
from pyspark.sql.functions import udf

def clean_tip_data(df: DataFrame) -> DataFrame:
    """
    Clean the tip data by removing rows with missing values.
    """
    cleaned_df = df \
        .dropna(subset=["user_id", "business_id", "text", "date"]) \
        .withColumn("date", functions.to_date("date", "yyyy-MM-dd")) \
        .filter(functions.year("date") > 2016)
    return cleaned_df

def extract_vibe(text):
    words = []
    if 'thrilling' in text or 'daring' in text or 'unpredictable' in text or 'memorable' in text or 'transformative' in text or 'adventure' in text:
        words.append('adventure')
    if 'tranquil' in text or 'serene' in text or 'placid' in text or 'quiet' in text or 'calming' in text:
        words.append('peaceful')
    if 'attentive' in text or 'conscious' in text or 'present' in text or 'thoughtful' in text or 'intentional' in text:
        words.append('mindful')
    if 'fun' in text or 'silly' in text or 'goofy' in text or 'cheeky' in text or 'happy' in text:
        words.append('playful')
    if 'relaxed' in text or 'easygoing' in text or 'cool' in text or 'mellow' in text or 'chill' in text:
        words.append('chill')
    if 'lovely' in text or 'sweet' in text or 'passionate' in text or 'tender' in text or 'romantic' in text:
        words.append('romantic')
    if 'sentimental' in text or 'yearning' in text or 'old time' in text or 'dreamy' in text or 'nostalgic' in text:
        words.append('nostalgic')
    if len(words) == 0:
        words.append('undefined')
    return words

vibe_udf = udf(extract_vibe, ArrayType(StringType()))

def sentiment_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 1  # Positive sentiment
    elif polarity < -0.1:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

sentiment_udf = udf(sentiment_analysis, IntegerType())

def transform_tip_data(df: DataFrame) -> DataFrame:
    """
    Transform the tip data by adding columns for sentiment score, vibe keywords, and tip length.
    """
    df = df \
        .withColumn("vibe_keywords", vibe_udf(functions.lower(functions.col("text")))) \
        .withColumn("sentiment_score", sentiment_udf(functions.col("text"))) \
        .withColumn("tip_length", functions.length(functions.col("text")))
    return df  

def aggregate_tip_data(df: DataFrame) -> DataFrame:
    """Example aggregation: Count tips per business."""
    return df.groupBy("business_id").count()

def save_tip_json(df: DataFrame, s3_path: str) -> None:
    """Save the processed tip data as a JSON file to S3."""
    df.write.format("json").save(s3_path)
