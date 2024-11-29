from pyspark.sql import SparkSession
from pyspark import RDD

def get_spark_session(app_name="yelp_data_processing"):
    return SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

def load_data_from_s3(spark, s3_path):
    return spark.sparkContext.textFile(s3_path)

def save_data_to_s3(rdd, s3_path):
    dataframe = rdd.toDF()
    dataframe.write.format("parquet").save(s3_path)
    
def save_rdd_to_s3_text(rdd, s3_path):
    rdd.saveAsTextFile(s3_path)
