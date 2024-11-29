# Configuration settings (e.g., S3 paths, file names)

S3_BUCKET_NAME = "apache-spark-anand"
INPUT_FILES = {
    "business": "s3://apache-spark-anand/yelp_academic_dataset_business.json",
    "review": "s3://apache-spark-anand/yelp_academic_dataset_review.json",
    "tip": "s3://apache-spark-anand/yelp_academic_dataset_tip.json",
}
OUTPUT_DIR = "s3://yelp-dataset-output/"
