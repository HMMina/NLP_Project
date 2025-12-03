"""
PySpark Sentiment Analysis Pipeline
====================================
This script demonstrates text classification using Apache Spark ML Pipeline.

Pipeline stages:
1. Tokenizer: Split text into words
2. StopWordsRemover: Remove common stop words
3. HashingTF: Convert words to feature vectors
4. IDF: Apply inverse document frequency weighting
5. LogisticRegression: Train classification model
"""

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def create_spark_session():
    """Initialize Spark session."""
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_prepare_data(spark, data_path):
    """Load CSV data and prepare labels."""
    print(f"\n{'='*60}")
    print("Step 1: Loading Data")
    print('='*60)
    
    # Load data
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    initial_count = df.count()
    print(f"Initial rows: {initial_count}")
    
    # Show sample data
    print("\nSample data:")
    df.show(5, truncate=50)
    
    # Convert sentiment labels: -1/1 → 0/1
    # Original: -1 (negative), 1 (positive)
    # Converted: 0 (negative), 1 (positive)
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    
    # Drop rows with null sentiment values
    df = df.dropna(subset=["sentiment"])
    final_count = df.count()
    
    print(f"Rows after cleaning: {final_count}")
    print(f"Dropped {initial_count - final_count} rows with null values")
    
    # Show label distribution
    print("\nLabel distribution:")
    df.groupBy("label").count().show()
    
    return df


def build_pipeline():
    """Build ML pipeline with preprocessing and model."""
    print(f"\n{'='*60}")
    print("Step 2: Building Pipeline")
    print('='*60)
    
    # Stage 1: Tokenizer - Split text into words
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    print("Stage 1: Tokenizer (text → words)")
    
    # Stage 2: StopWordsRemover - Remove common stop words
    stopwordsRemover = StopWordsRemover(
        inputCol="words", 
        outputCol="filtered_words"
    )
    print("Stage 2: StopWordsRemover (words → filtered_words)")
    
    # Stage 3: HashingTF - Convert words to feature vectors
    # numFeatures: Size of feature vector (higher = more features but slower)
    hashingTF = HashingTF(
        inputCol="filtered_words", 
        outputCol="raw_features", 
        numFeatures=10000
    )
    print("Stage 3: HashingTF (filtered_words → raw_features, 10000 features)")
    
    # Stage 4: IDF - Apply inverse document frequency weighting
    # Downweights common terms, upweights rare terms
    idf = IDF(
        inputCol="raw_features", 
        outputCol="features"
    )
    print("Stage 4: IDF (raw_features → features)")
    
    # Stage 5: LogisticRegression - Classification model
    lr = LogisticRegression(
        maxIter=100,
        regParam=0.001,
        featuresCol="features",
        labelCol="label"
    )
    print("Stage 5: LogisticRegression (maxIter=100, regParam=0.001)")
    
    # Combine all stages into a pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
    print("\n Pipeline created with 5 stages")
    
    return pipeline


def train_model(pipeline, train_data):
    """Train the model using the pipeline."""
    print(f"\n{'='*60}")
    print("Step 3: Training Model")
    print('='*60)
    
    print(f"Training samples: {train_data.count()}")
    print("Training in progress...")
    
    # Fit the pipeline (trains all stages)
    model = pipeline.fit(train_data)
    
    print("Model training completed!")
    return model


def evaluate_model(model, test_data):
    """Evaluate model performance on test data."""
    print(f"\n{'='*60}")
    print("Step 4: Evaluating Model")
    print('='*60)
    
    print(f"Test samples: {test_data.count()}")
    
    # Make predictions
    predictions = model.transform(test_data)
    
    # Show sample predictions
    print("\n Sample predictions:")
    predictions.select("text", "label", "prediction", "probability").show(10, truncate=50)
    
    # Calculate metrics
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator_accuracy.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)
    
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    precision = evaluator_precision.evaluate(predictions)
    
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    recall = evaluator_recall.evaluate(predictions)
    
    # Print metrics
    print(f"\n{'='*60}")
    print("Evaluation Metrics")
    print('='*60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    return predictions, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def analyze_predictions(predictions):
    """Analyze prediction results."""
    print(f"\n{'='*60}")
    print("Prediction Analysis")
    print('='*60)
    
    # Count correct and incorrect predictions
    total = predictions.count()
    correct = predictions.filter(col("label") == col("prediction")).count()
    incorrect = total - correct
    
    print(f"Total predictions: {total}")
    print(f"Correct: {correct} ({correct/total*100:.2f}%)")
    print(f"Incorrect: {incorrect} ({incorrect/total*100:.2f}%)")
    
    # Show some misclassified examples
    print("\n Misclassified examples:")
    misclassified = predictions.filter(col("label") != col("prediction"))
    misclassified.select("text", "label", "prediction").show(5, truncate=80)


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("PySpark Sentiment Analysis Pipeline")
    print("="*60)
    
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Data path
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        data_path = os.path.join(workspace_root, "sentiments.csv")
        
        if not os.path.exists(data_path):
            print(f"\n Error: Data file not found at {data_path}")
            print("Please ensure sentiments.csv exists in the workspace root.")
            return
        
        # Load and prepare data
        df = load_and_prepare_data(spark, data_path)
        
        # Split data into train and test (80/20)
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        
        # Build pipeline
        pipeline = build_pipeline()
        
        # Train model
        model = train_model(pipeline, train_data)
        
        # Evaluate model
        predictions, metrics = evaluate_model(model, test_data)
        
        # Analyze predictions
        analyze_predictions(predictions)
        
        print(f"\n{'='*60}")
        print("Pipeline execution completed successfully!")
        print('='*60)
        
    except Exception as e:
        print(f"\n Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop Spark session
        spark.stop()
        print("\n Spark session stopped.")


if __name__ == "__main__":
    main()
