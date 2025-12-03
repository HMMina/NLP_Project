import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.ml.feature import Word2Vec

def main():
    # 1. Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Spark Word2Vec Demo") \
        .master("local[*]") \
        .getOrCreate()

    # 2. Load dataset (C4 dataset JSON)
    # Each line is a JSON object with a 'text' field
    input_path = "../../c4-train.00000-of-01024-30K.json"
    df = spark.read.json(input_path)
    
    # Check schema
    print("Schema of loaded dataset:")
    df.printSchema()

    # 3. Basic text preprocessing
    # - Select text column
    # - Convert to lowercase
    # - Remove punctuation and special characters (keep letters, numbers, and spaces)
    # - Split into words
    cleaned_df = (
        df.select(lower(col("text")).alias("text"))
          .withColumn("text", regexp_replace(col("text"), r"[^\w\s]", " "))
          .withColumn("words", split(col("text"), r"\s+"))
          .filter(col("text").isNotNull())
    )

    print("Sample cleaned data:")
    cleaned_df.show(3, truncate=False)

    # 4. Train Word2Vec model
    word2vec = Word2Vec(
        vectorSize=100,   # 100-dimensional embeddings
        minCount=5,       # ignore rare words
        inputCol="words",
        outputCol="result"
    )

    model = word2vec.fit(cleaned_df)

    # 5. Demonstrate model usage
    synonyms = model.findSynonyms("computer", 5)
    print("Top 5 words similar to 'computer':")
    synonyms.show(truncate=False)

    # 6. Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()
