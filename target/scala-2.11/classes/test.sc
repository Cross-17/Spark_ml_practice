import org.apache.log4j.{Logger, Level}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
val spark = SparkSession
  .builder()
  .config("spark.master", "local")
  .appName("Titanic")
  .getOrCreate()
import spark.implicits._

val df = spark.read
  .option("delimiter", "\t")
  .option("header", "true")
  .csv("D:\\Do\\train.csv")