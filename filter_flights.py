from pyspark.sql import SparkSession
import sys

def filter_flights(carrier):
    # Initialize Spark session
    spark = SparkSession.builder.appName("Airline Data").getOrCreate()

    # Read CSV file
    file_path_pattern = './data/flights.csv'
    df = spark.read.csv(file_path_pattern, header=True, inferSchema=True)

    # Filter data for the specified airline
    df_filtered = df.filter(df.Carrier == carrier)

    # Save the filtered data to a new file
    output_path = f'./data/{carrier}_flights.csv'
    df_filtered.write.csv(output_path, header=True)

    spark.stop()

if __name__ == "__main__":
    carrier = sys.argv[1]
    filter_flights(carrier)
