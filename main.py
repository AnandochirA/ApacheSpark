# Main script to execute the pipeline
from config import INPUT_FILES, OUTPUT_DIR
from utils import get_spark_session

def main():
    spark = get_spark_session()

    # Process business data
    process_business_data(
        spark,
        INPUT_FILES["business"],
        OUTPUT_DIR + "filtered_business/"
    )

    spark.stop()

if __name__ == "__main__":
    main()
