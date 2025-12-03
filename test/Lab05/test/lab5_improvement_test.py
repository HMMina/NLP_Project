"""
Task 4: Model Improvement Experiments
======================================
This script demonstrates various techniques to improve text classification performance:
1. Feature dimensionality reduction
2. Word2Vec embeddings
3. Alternative model architectures (Naive Bayes, GBT)
"""

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, size
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, 
    Word2Vec, VectorAssembler
)
from pyspark.ml.classification import (
    LogisticRegression, 
    NaiveBayes,
    GBTClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time


def create_spark_session():
    """Initialize Spark session with optimized settings."""
    spark = SparkSession.builder \
        .appName("ModelImprovement") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark, data_path):
    """Load and prepare dataset."""
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    df = df.dropna(subset=["sentiment"])
    return df


def experiment_1_dimensionality_reduction(train_data, test_data):
    """
    Experiment 1: Test different numFeatures values in HashingTF
    Goal: Find optimal feature dimensionality
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: Feature Dimensionality Reduction")
    print('='*70)
    print("Testing different numFeatures values: 1000, 5000, 10000, 20000")
    
    results = []
    feature_sizes = [1000, 5000, 10000, 20000]
    
    for num_features in feature_sizes:
        print(f"\n--- Testing with numFeatures = {num_features} ---")
        
        # Build pipeline with specific numFeatures
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        hashingTF = HashingTF(
            inputCol="filtered_words", 
            outputCol="raw_features", 
            numFeatures=num_features
        )
        idf = IDF(inputCol="raw_features", outputCol="features")
        lr = LogisticRegression(maxIter=100, regParam=0.001)
        
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
        
        # Train and evaluate
        start_time = time.time()
        model = pipeline.fit(train_data)
        train_time = time.time() - start_time
        
        predictions = model.transform(test_data)
        
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        
        evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")
        f1 = evaluator_f1.evaluate(predictions)
        
        results.append({
            'numFeatures': num_features,
            'accuracy': accuracy,
            'f1': f1,
            'train_time': train_time
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Training Time: {train_time:.2f}s")
    
    # Summary
    print(f"\n{'='*70}")
    print("Experiment 1 Summary")
    print('='*70)
    print(f"{'NumFeatures':<15} {'Accuracy':<12} {'F1 Score':<12} {'Train Time':<12}")
    print('-'*70)
    for r in results:
        print(f"{r['numFeatures']:<15} {r['accuracy']:<12.4f} {r['f1']:<12.4f} {r['train_time']:<12.2f}s")
    
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n Best: numFeatures={best['numFeatures']} with Accuracy={best['accuracy']:.4f}")
    
    return results


def experiment_2_word2vec_embeddings(train_data, test_data):
    """
    Experiment 2: Use Word2Vec embeddings instead of TF-IDF
    Goal: Capture semantic meaning better
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Word2Vec Embeddings")
    print('='*70)
    print("Using Word2Vec to generate dense word embeddings")
    
    # Build pipeline with Word2Vec
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # Word2Vec: generates dense vectors
    word2vec = Word2Vec(
        vectorSize=10000,  # Embedding dimension
        minCount=5,      # Minimum word frequency
        inputCol="filtered_words",
        outputCol="features"
    )
    
    lr = LogisticRegression(maxIter=100, regParam=0.001)
    
    pipeline = Pipeline(stages=[tokenizer, remover, word2vec, lr])
    
    # Train and evaluate
    print("\nTraining Word2Vec model...")
    start_time = time.time()
    model = pipeline.fit(train_data)
    train_time = time.time() - start_time
    
    predictions = model.transform(test_data)
    
    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator_acc.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")
    f1 = evaluator_f1.evaluate(predictions)
    
    evaluator_prec = MulticlassClassificationEvaluator(metricName="weightedPrecision")
    precision = evaluator_prec.evaluate(predictions)
    
    evaluator_rec = MulticlassClassificationEvaluator(metricName="weightedRecall")
    recall = evaluator_rec.evaluate(predictions)
    
    print(f"\n{'='*70}")
    print("Experiment 2 Results")
    print('='*70)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Training Time: {train_time:.2f}s")
    
    return {
        'method': 'Word2Vec',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time
    }


def experiment_3_model_comparison(train_data, test_data):
    """
    Experiment 3: Compare different model architectures
    Goal: Find best model for this task
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: Model Architecture Comparison")
    print('='*70)
    print("Comparing: Logistic Regression, Naive Bayes, GBT Classifier")
    
    results = []
    
    # Common preprocessing stages
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # Model 1: Logistic Regression
    print("\n--- Model 1: Logistic Regression ---")
    lr = LogisticRegression(maxIter=100, regParam=0.001)
    pipeline_lr = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
    
    start_time = time.time()
    model_lr = pipeline_lr.fit(train_data)
    train_time_lr = time.time() - start_time
    
    pred_lr = model_lr.transform(test_data)
    acc_lr = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pred_lr)
    f1_lr = MulticlassClassificationEvaluator(metricName="f1").evaluate(pred_lr)
    
    print(f"  Accuracy: {acc_lr:.4f}, F1: {f1_lr:.4f}, Time: {train_time_lr:.2f}s")
    results.append({'model': 'Logistic Regression', 'accuracy': acc_lr, 'f1': f1_lr, 'time': train_time_lr})
    
    # Model 2: Naive Bayes
    print("\n--- Model 2: Naive Bayes ---")
    nb = NaiveBayes(smoothing=1.0)
    pipeline_nb = Pipeline(stages=[tokenizer, remover, hashingTF, idf, nb])
    
    start_time = time.time()
    model_nb = pipeline_nb.fit(train_data)
    train_time_nb = time.time() - start_time
    
    pred_nb = model_nb.transform(test_data)
    acc_nb = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pred_nb)
    f1_nb = MulticlassClassificationEvaluator(metricName="f1").evaluate(pred_nb)
    
    print(f"  Accuracy: {acc_nb:.4f}, F1: {f1_nb:.4f}, Time: {train_time_nb:.2f}s")
    results.append({'model': 'Naive Bayes', 'accuracy': acc_nb, 'f1': f1_nb, 'time': train_time_nb})
    
    # Model 3: Gradient-Boosted Trees
    print("\n--- Model 3: Gradient-Boosted Trees ---")
    gbt = GBTClassifier(maxIter=20, maxDepth=7)
    pipeline_gbt = Pipeline(stages=[tokenizer, remover, hashingTF, idf, gbt])
    
    start_time = time.time()
    model_gbt = pipeline_gbt.fit(train_data)
    train_time_gbt = time.time() - start_time
    
    pred_gbt = model_gbt.transform(test_data)
    acc_gbt = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pred_gbt)
    f1_gbt = MulticlassClassificationEvaluator(metricName="f1").evaluate(pred_gbt)
    
    print(f"  Accuracy: {acc_gbt:.4f}, F1: {f1_gbt:.4f}, Time: {train_time_gbt:.2f}s")
    results.append({'model': 'Gradient-Boosted Trees', 'accuracy': acc_gbt, 'f1': f1_gbt, 'time': train_time_gbt})
    
    # Summary
    print(f"\n{'='*70}")
    print("Experiment 3 Summary")
    print('='*70)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'Train Time':<12}")
    print('-'*70)
    for r in results:
        print(f"{r['model']:<25} {r['accuracy']:<12.4f} {r['f1']:<12.4f} {r['time']:<12.2f}s")
    
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n Best Model: {best['model']} with Accuracy={best['accuracy']:.4f}")
    
    return results


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("TASK 4: Model Improvement Experiments")
    print("="*70)
    
    spark = create_spark_session()
    
    try:
        # Load data
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        data_path = os.path.join(workspace_root, "sentiments.csv")
        
        if not os.path.exists(data_path):
            print(f"\n Error: Data file not found at {data_path}")
            return
        
        print(f"\nLoading data from: {data_path}")
        df = load_data(spark, data_path)
        print(f"Total samples: {df.count()}")
        
        # Split data
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        print(f"Train: {train_data.count()}, Test: {test_data.count()}")
        
        # Run experiments
        exp1_results = experiment_1_dimensionality_reduction(train_data, test_data)
        exp2_results = experiment_2_word2vec_embeddings(train_data, test_data)
        exp3_results = experiment_3_model_comparison(train_data, test_data)
        
        # Final summary
        print(f"\n{'='*70}")
        print("FINAL SUMMARY - All Experiments")
        print('='*70)
        print("\n Key Findings:")
        print(f"1. Best numFeatures: {max(exp1_results, key=lambda x: x['accuracy'])['numFeatures']}")
        print(f"2. Word2Vec Accuracy: {exp2_results['accuracy']:.4f}")
        print(f"3. Best Model: {max(exp3_results, key=lambda x: x['accuracy'])['model']}")
        
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
