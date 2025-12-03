import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec, Normalizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import java.io.{File, PrintWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.math.sqrt

object TextPipelineExample {
  def main(args: Array[String]): Unit = {
    // 1. Customize document limit (Enhancement Request 1)
    val limitDocuments = 1000 // Easily configurable document limit
    
    val startTime = LocalDateTime.now()
    val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
    
    // Performance measurement variables (Enhancement Request 2)
    var stageStartTime = System.currentTimeMillis()
    
    println("=== Enhanced NLP Pipeline with Performance Measurement ===")
    println(s"Document limit set to: $limitDocuments")
    println(s"Pipeline started at: ${startTime.format(formatter)}")
    
    // Tạo thư mục log và results nếu chưa có
    new File("log").mkdirs()
    new File("results").mkdirs()
    
    // Khởi tạo log writer
    val logWriter = new PrintWriter(new File(s"log/pipeline_${startTime.format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))}.log"))
    
    try {
      logWriter.println(s"Pipeline started at: ${startTime.format(formatter)}")
      logWriter.flush()
    // 1. Tạo SparkSession với config tối ưu cho Windows
    val spark = SparkSession.builder()
      .appName("TF-IDF Pipeline Example")
      .master("local[*]") // chạy local
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .config("spark.storage.level", "MEMORY_ONLY")
      .config("spark.local.dir", System.getProperty("java.io.tmpdir"))
      .getOrCreate()

      // Giảm log level để ít error message hơn
      spark.sparkContext.setLogLevel("WARN")

      import spark.implicits._

      logWriter.println("SparkSession created successfully")
      logWriter.flush()

      // 2. Load dữ liệu JSON with performance measurement (Requirements 1)
      stageStartTime = System.currentTimeMillis()
      // Support compressed .json file as per requirements
      val dataPath = "../../c4-train.00000-of-01024-30K.json" // Load compressed file as specified
      logWriter.println(s"Loading data from: $dataPath")
      logWriter.println(s"Document limit: $limitDocuments")
      logWriter.flush()
      
      // Spark automatically handles decompression
      val df = spark.read.json(dataPath)
        .limit(limitDocuments) // Use configurable limit (Enhancement Request 1)

      // Select text column and remove null values (Requirements 1)
      val textDF = df.select("text").na.drop()
      
      val recordCount = textDF.count()
      val dataLoadTime = System.currentTimeMillis() - stageStartTime
      println(s"Data Loading Performance: ${dataLoadTime}ms")
      logWriter.println(s"Loaded $recordCount records")
      logWriter.println(s"Data loading time: ${dataLoadTime}ms")
      logWriter.flush()
      
      // 3. Pipeline Configuration with performance measurement
      stageStartTime = System.currentTimeMillis()
      
      // 3.1 Tokenization (RegexTokenizer)
      val tokenizer = new RegexTokenizer()
        .setInputCol("text")
        .setOutputCol("tokens")
        .setPattern("\\s+|[.,;!?()\"']") // tách theo ký tự không phải chữ

      // EXERCISE 1: Uncomment line below and comment RegexTokenizer above
      // val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")

      logWriter.println("Tokenizer configured")
      logWriter.flush()

      // 4. StopWordsRemover
      val remover = new StopWordsRemover()
        .setInputCol("tokens")
        .setOutputCol("filtered")

      logWriter.println("StopWordsRemover configured")
      logWriter.flush()

      // 5. HashingTF (Term Frequency) - sử dụng 20000 features như yêu cầu
      // EXERCISE 2: Change numFeatures from 20000 to 1000
      val hashingTF = new HashingTF()
        .setInputCol("filtered")
        .setOutputCol("rawFeatures")
        .setNumFeatures(20000) // số chiều vector

      logWriter.println("HashingTF configured with 20000 features")
      logWriter.flush()

      // 6. IDF (Inverse Document Frequency)
      val idf = new IDF()
        .setInputCol("rawFeatures")
        .setOutputCol("idfFeatures")

      logWriter.println("IDF configured")
      logWriter.flush()

      // 7. Vector Normalization (Enhancement Request 3)
      val normalizer = new Normalizer()
        .setInputCol("idfFeatures")
        .setOutputCol("features")
        .setP(2.0) // L2 normalization for better cosine similarity

      logWriter.println("Normalizer configured (L2 normalization)")
      logWriter.flush()

      // EXERCISE 4: Alternative Word2Vec implementation (comment out HashingTF + IDF above)
      /*
      val word2Vec = new Word2Vec()
        .setInputCol("filtered")
        .setOutputCol("features")
        .setVectorSize(100)     // Word2Vec vector size
        .setMinCount(0)         // Minimum word frequency
        
      logWriter.println("Word2Vec configured")
      logWriter.flush()
      */

      // 8. Xây dựng 2 Pipeline để so sánh (Enhancement Request 3)
      
      // Pipeline 1: Cơ bản (không có Normalizer)
      val basicPipeline = new Pipeline()
        .setStages(Array(tokenizer, remover, hashingTF, idf))
        
      // Pipeline 2: Enhanced (có Normalizer)  
      val enhancedPipeline = new Pipeline()
        .setStages(Array(tokenizer, remover, hashingTF, idf, normalizer))
        
      // EXERCISE 4: For Word2Vec, use this pipeline instead:
      // val word2vecPipeline = new Pipeline()
      //   .setStages(Array(tokenizer, remover, word2Vec))

      val pipelineConfigTime = System.currentTimeMillis() - stageStartTime
      println(s"Pipeline Configuration Performance: ${pipelineConfigTime}ms")
      logWriter.println("Two pipelines configured: Basic (TF-IDF only) and Enhanced (TF-IDF + Normalizer)")
      logWriter.println(s"Pipeline configuration time: ${pipelineConfigTime}ms")
      logWriter.flush()

      // 9. Train và compare 2 pipelines với performance measurement
      stageStartTime = System.currentTimeMillis()
      logWriter.println("Training both pipelines for comparison...")
      logWriter.flush()
      
      // Train Basic Pipeline (TF-IDF only)
      println("\nTraining Basic Pipeline (TF-IDF only)...")
      val basicStartTime = System.currentTimeMillis()
      val basicModel = basicPipeline.fit(textDF)
      val basicResult = basicModel.transform(textDF)
      basicResult.cache()
      val basicCount = basicResult.count()
      val basicTrainTime = System.currentTimeMillis() - basicStartTime
      println(s"Basic Pipeline Performance: ${basicTrainTime}ms")
      
      // Train Enhanced Pipeline (TF-IDF + Normalizer)  
      println("\nTraining Enhanced Pipeline (TF-IDF + Normalizer)...")
      val enhancedStartTime = System.currentTimeMillis()
      val enhancedModel = enhancedPipeline.fit(textDF)
      val enhancedResult = enhancedModel.transform(textDF)
      enhancedResult.cache()
      val enhancedCount = enhancedResult.count()
      val enhancedTrainTime = System.currentTimeMillis() - enhancedStartTime
      println(s"Enhanced Pipeline Performance: ${enhancedTrainTime}ms")
      
      val totalPipelineTime = System.currentTimeMillis() - stageStartTime
      println(f"\n Pipeline Comparison:")
      println(f"Basic Pipeline: ${basicTrainTime}ms")
      println(f"Enhanced Pipeline: ${enhancedTrainTime}ms") 
      println(f"Normalizer Overhead: ${enhancedTrainTime - basicTrainTime}ms")

      logWriter.println("Both pipelines trained successfully")
      logWriter.println(s"Basic pipeline time: ${basicTrainTime}ms")
      logWriter.println(s"Enhanced pipeline time: ${enhancedTrainTime}ms")
      logWriter.println(s"Total pipeline training time: ${totalPipelineTime}ms")
      logWriter.flush()

      // 10. Hiển thị kết quả comparison
      println("\n=== Pipeline Results Comparison ===")
      println("\n Basic Pipeline Results (TF-IDF only):")
      basicResult.select("text", "tokens", "filtered", "idfFeatures").show(3, truncate = false)
      
      println("\n Enhanced Pipeline Results (TF-IDF + Normalizer):")  
      enhancedResult.select("text", "tokens", "filtered", "features").show(3, truncate = false)
      
      // 11. Document Similarity Analysis using Enhanced Pipeline (Enhancement Request 4)
      println("\n=== Document Similarity Analysis (Using Normalized Vectors) ===")
      stageStartTime = System.currentTimeMillis()
      
      // Function to calculate cosine similarity between normalized vectors
      def cosineSimilarity(v1: Vector, v2: Vector): Double = {
        // Since vectors are already normalized, cosine similarity = dot product
        val dot = v1.toArray.zip(v2.toArray).map { case (a, b) => a * b }.sum
        dot
      }
      
      // Collect all documents with their normalized features from enhanced pipeline
      val documentsWithFeatures = enhancedResult.select("text", "features").collect()
      
      if (documentsWithFeatures.length > 1) {
        // Select the first document as reference (you can change this index)
        val referenceDocIndex = 0
        val referenceDoc = documentsWithFeatures(referenceDocIndex)
        val referenceText = referenceDoc.getAs[String]("text")
        val referenceFeatures = referenceDoc.getAs[Vector]("features")
        
        println(s"Reference Document: ${referenceText.take(100)}...")
        logWriter.println(s"Reference Document (index $referenceDocIndex): ${referenceText.take(100)}...")
        
        // Calculate similarities with all other documents using normalized vectors
        val similarities = documentsWithFeatures.zipWithIndex
          .filter(_._2 != referenceDocIndex) // Exclude reference document itself
          .map { case (doc, index) =>
            val docText = doc.getAs[String]("text")
            val docFeatures = doc.getAs[Vector]("features")
            val similarity = cosineSimilarity(referenceFeatures, docFeatures)
            (index, docText, similarity)
          }
          .sortBy(-_._3) // Sort by similarity descending
          .take(5) // Top 5 similar documents
        
        println("\n Top 5 Most Similar Documents (Using Normalized Vectors):")
        similarities.zipWithIndex.foreach { case ((docIndex, docText, similarity), rank) =>
          println(s"${rank + 1}. Document $docIndex (Cosine Similarity: ${f"$similarity%.4f"})")
          println(s"   Text: ${docText.take(150)}...")
          println()
        }
        
        // Log similarity results
        logWriter.println("\n=== Document Similarity Analysis ===")
        logWriter.println(s"Reference Document Index: $referenceDocIndex")
        logWriter.println(s"Reference Text: ${referenceText.take(100)}...")
        logWriter.println("\nTop 5 Similar Documents:")
        similarities.zipWithIndex.foreach { case ((docIndex, docText, similarity), rank) =>
          logWriter.println(s"${rank + 1}. Document $docIndex (Similarity: $similarity)")
          logWriter.println(s"   Text: ${docText.take(100)}...")
        }
      } else {
        println(" Need at least 2 documents for similarity analysis")
        logWriter.println("Need at least 2 documents for similarity analysis")
      }
      
      val similarityAnalysisTime = System.currentTimeMillis() - stageStartTime
      println(s"Similarity Analysis Performance: ${similarityAnalysisTime}ms")
      logWriter.println(s"Similarity analysis time: ${similarityAnalysisTime}ms")
      logWriter.flush()
      
      // 12. Lưu kết quả vào file theo yêu cầu
      stageStartTime = System.currentTimeMillis()
      logWriter.println("Saving pipeline comparison results to file...")
      logWriter.flush()
      
      // Lưu kết quả vào file với tên chính xác như yêu cầu
      val outputFileName = "results/lab17_pipeline_output.txt"
      
      // Tạo file writer cho kết quả
      val resultWriter = new PrintWriter(new File(outputFileName))
      
      try {
        // Thu thập kết quả từ cả 2 pipelines
        val basicResultData = basicResult.select("text", "idfFeatures").collect()
        val enhancedResultData = enhancedResult.select("text", "features").collect()
        
        resultWriter.println("=== Dual Pipeline Comparison Output ===")
        resultWriter.println(s"Pipeline executed at: ${startTime.format(formatter)}")
        resultWriter.println(s"Total documents processed: ${enhancedResultData.length}")
        resultWriter.println(s"Document limit setting: $limitDocuments")
        resultWriter.println(s"Feature vector size: ${hashingTF.getNumFeatures}")
        resultWriter.println(s"")
        resultWriter.println(s"Pipeline 1: Basic TF-IDF (no normalization)")
        resultWriter.println(s"Pipeline 2: Enhanced TF-IDF + L2 Normalization")
        resultWriter.println("=" * 50)
        
        // Save comparison data
        enhancedResultData.zipWithIndex.foreach { case (row, index) =>
          resultWriter.println(s"Document ${index + 1}:")
          resultWriter.println(s"Text: ${row.getAs[String]("text").take(100)}...")
          
          if (index < basicResultData.length) {
            resultWriter.println(s"Basic TF-IDF Features: ${basicResultData(index).getAs[org.apache.spark.ml.linalg.Vector]("idfFeatures")}")
          }
          resultWriter.println(s"Normalized TF-IDF Features: ${row.getAs[org.apache.spark.ml.linalg.Vector]("features")}")
          resultWriter.println("-" * 30)
        }
        
        // Add similarity analysis to output file if available
        if (documentsWithFeatures.length > 1) {
          resultWriter.println("\n=== Document Similarity Analysis ===")
          val referenceDoc = documentsWithFeatures(0)
          val referenceText = referenceDoc.getAs[String]("text")
          val referenceFeatures = referenceDoc.getAs[Vector]("features")
          
          val similarities = documentsWithFeatures.zipWithIndex
            .filter(_._2 != 0)
            .map { case (doc, index) =>
              val docText = doc.getAs[String]("text")
              val docFeatures = doc.getAs[Vector]("features")
              val similarity = referenceFeatures.toArray.zip(docFeatures.toArray).map { case (a, b) => a * b }.sum
              (index, docText, similarity)
            }
            .sortBy(-_._3)
            .take(5)
          
          resultWriter.println(s"Reference Document: ${referenceText.take(100)}...")
          resultWriter.println("Top 5 Similar Documents:")
          similarities.zipWithIndex.foreach { case ((docIndex, docText, similarity), rank) =>
            resultWriter.println(s"${rank + 1}. Document $docIndex (Similarity: $similarity)")
            resultWriter.println(s"   Text: ${docText.take(100)}...")
          }
        }
        
        resultWriter.println("=== End of Output ===")
        
      } finally {
        resultWriter.close()
      }

      val fileWriteTime = System.currentTimeMillis() - stageStartTime
      println(s"File Writing Performance: ${fileWriteTime}ms")
      logWriter.println(s"Results saved successfully to $outputFileName")
      logWriter.println(s"File writing time: ${fileWriteTime}ms")
      logWriter.flush()
      
      // 13. Enhanced Performance Summary and Statistics  
      val finalCount = enhancedCount
      val totalProcessingTime = dataLoadTime + pipelineConfigTime + totalPipelineTime + similarityAnalysisTime + fileWriteTime
      
      println("\n === Dual Pipeline Performance Summary ===")
      println(s"Total documents processed: $finalCount")
      println(s"Document limit setting: $limitDocuments")
      println(s"Vocabulary size (HashingTF): ${hashingTF.getNumFeatures}")
      println(s"Vector normalization: L2 normalization enabled")
      println(f"\n === Detailed Performance Breakdown ===")
      println(f"Data Loading: ${dataLoadTime}ms (${dataLoadTime.toDouble/totalProcessingTime*100}%.1f%%)")
      println(f"Pipeline Config: ${pipelineConfigTime}ms (${pipelineConfigTime.toDouble/totalProcessingTime*100}%.1f%%)")
      println(f"Basic Pipeline Training: ${basicTrainTime}ms (${basicTrainTime.toDouble/totalProcessingTime*100}%.1f%%)")
      println(f"Enhanced Pipeline Training: ${enhancedTrainTime}ms (${enhancedTrainTime.toDouble/totalProcessingTime*100}%.1f%%)")
      println(f"Similarity Analysis: ${similarityAnalysisTime}ms (${similarityAnalysisTime.toDouble/totalProcessingTime*100}%.1f%%)")
      println(f"File Writing: ${fileWriteTime}ms (${fileWriteTime.toDouble/totalProcessingTime*100}%.1f%%)")
      println(f"Total Processing Time: ${totalProcessingTime}ms")
      println(f"\n === Pipeline Comparison Analysis ===")
      println(f"Normalizer Overhead: ${enhancedTrainTime - basicTrainTime}ms")
      println(f"Overhead Percentage: ${((enhancedTrainTime - basicTrainTime).toDouble/basicTrainTime*100)}%.2f%%")
      
      logWriter.println(s"=== Dual Pipeline Performance Summary ===")
      logWriter.println(s"Total documents processed: $finalCount")
      logWriter.println(s"Document limit setting: $limitDocuments")
      logWriter.println(s"Vocabulary size (HashingTF): ${hashingTF.getNumFeatures}")
      logWriter.println(s"Performance breakdown:")
      logWriter.println(s"  Data loading: ${dataLoadTime}ms")
      logWriter.println(s"  Pipeline configuration: ${pipelineConfigTime}ms")
      logWriter.println(s"  Basic pipeline training: ${basicTrainTime}ms")
      logWriter.println(s"  Enhanced pipeline training: ${enhancedTrainTime}ms")
      logWriter.println(s"  Similarity analysis: ${similarityAnalysisTime}ms")
      logWriter.println(s"  File writing: ${fileWriteTime}ms")
      logWriter.println(s"  Total processing time: ${totalProcessingTime}ms")
      logWriter.println(s"  Normalizer overhead: ${enhancedTrainTime - basicTrainTime}ms")
      logWriter.println(s"  Total pipeline time: ${totalPipelineTime}ms")
      
      val endTime = LocalDateTime.now()
      val totalExecutionTime = java.time.Duration.between(startTime, endTime).toMillis
      
      println(s"\n=== Pipeline Completed Successfully ===")
      println(s"Started at: ${startTime.format(formatter)}")
      println(s"Ended at: ${endTime.format(formatter)}")
      println(s"Total Execution Time: ${totalExecutionTime}ms")
      
      logWriter.println(s"Enhanced pipeline completed successfully at: ${endTime.format(formatter)}")
      logWriter.println(s"Total execution time: ${totalExecutionTime}ms")
      logWriter.flush()
      
      // 14. Đóng Spark session một cách an toàn
      try {
        spark.stop()
        logWriter.println("Spark session closed successfully")
        println("Spark session closed successfully")
      } catch {
        case e: Exception => 
          println(s"Warning during Spark shutdown: ${e.getMessage}")
          logWriter.println(s"Warning during Spark shutdown: ${e.getMessage}")
      }
      
    } catch {
      case e: Exception =>
        println(s"Error occurred: ${e.getMessage}")
        logWriter.println(s"Error occurred: ${e.getMessage}")
        logWriter.println(s"Stack trace: ${e.getStackTrace.mkString("\n")}")
        throw e
    } finally {
      logWriter.close()
    }
  }
}
