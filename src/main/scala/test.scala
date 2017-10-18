import org.apache.log4j.{Logger, Level}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.feature.{IndexToString, VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorIndexer

object Test {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .config("spark.master", "local")
      .appName("Titanic")
      .getOrCreate()
    import spark.implicits._


    val (train, test) = loadData("D://Do/train.csv", "D://Do/test.csv", spark)

    val featureIndexer = new VectorAssembler()
      .setInputCols(test.schema.fieldNames)
      .setOutputCol("Feature")
    val numericFeatColNames = Seq("Age", "SibSp", "Parch", "Fare", "FamilySize")
    val f = test.columns.tail
    train.schema.fields.update(1,null)
    f.foreach(x => println(x))
    train.na.fill(0)
    train.printSchema()


//    val rf = new RandomForestRegressor()
//      .setLabelCol("SalePrice")
//      .setFeaturesCol("indexedFeatures")
//
//    val pipeline = new Pipeline()
//      .setStages(Array(featureIndexer, rf))
//
//    val model = pipeline.fit(train)
//
//    val predictions = model.transform(test)
//
//    predictions
//      .select("Id", "SalePrice")
//      .coalesce(1)
//      .write
//      .format("csv")
//      .mode("overwrite")
//      .option("header", "true")
//      .save("result")
  }

  def fillNAValues(trainDF: DataFrame, testDF: DataFrame): (DataFrame, DataFrame) = {
    val avgAge = trainDF.select("Age").union(testDF.select("Age"))
      .agg(avg("Age"))
      .collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }

    // fill empty values for the fare column
    val avgFare = trainDF.select("Fare").union(testDF.select("Fare"))
      .agg(avg("Fare"))
      .collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }

    // map to fill na values
    val fillNAMap = Map(
      "Fare"     -> avgFare,
      "Age"      -> avgAge,
      "Embarked" -> "S"
    )

    // udf to fill empty embarked string with S corresponding to Southampton
    val embarked: (String => String) = {
      case "" => "S"
      case a  => a
    }
    val embarkedUDF = udf(embarked)

    val newTrainDF = trainDF
      .na.fill(fillNAMap)
      .withColumn("Embarked", embarkedUDF(col("Embarked")))

    val newTestDF = testDF
      .na.fill(fillNAMap)
      .withColumn("Embarked", embarkedUDF(col("Embarked")))

    (newTrainDF, newTestDF)
  }

  def createExtraFeatures(trainDF: DataFrame, testDF: DataFrame): (DataFrame, DataFrame) = {
    // udf to create a FamilySize column as the sum of the SibSp and Parch columns + 1
    val familySize: ((Int, Int) => Int) = (sibSp: Int, parCh: Int) => sibSp + parCh + 1
    val familySizeUDF = udf(familySize)

    // udf to create a Title column extracting the title from the Name column
    val Pattern = ".*, (.*?)\\..*".r
    val titles = Map(
      "Mrs"    -> "Mrs",
      "Lady"   -> "Mrs",
      "Mme"    -> "Mrs",
      "Ms"     -> "Ms",
      "Miss"   -> "Miss",
      "Mlle"   -> "Miss",
      "Master" -> "Master",
      "Rev"    -> "Rev",
      "Don"    -> "Mr",
      "Sir"    -> "Sir",
      "Dr"     -> "Dr",
      "Col"    -> "Col",
      "Capt"   -> "Col",
      "Major"  -> "Col"
    )
    val title: ((String, String) => String) = {
      case (Pattern(t), sex) => titles.get(t) match {
        case Some(tt) => tt
        case None     =>
          if (sex == "male") "Mr"
          else "Mrs"
      }
      case _ => "Mr"
    }
    val titleUDF = udf(title)

    val newTrainDF = trainDF
      .withColumn("FamilySize", familySizeUDF(col("SibSp"), col("Parch")))
      .withColumn("Title", titleUDF(col("Name"), col("Sex")))
      .withColumn("SurvivedString", trainDF("Survived").cast(StringType))
    val newTestDF = testDF
      .withColumn("FamilySize", familySizeUDF(col("SibSp"), col("Parch")))
      .withColumn("Title", titleUDF(col("Name"), col("Sex")))
      .withColumn("SurvivedString", lit("0").cast(StringType))

    (newTrainDF, newTestDF)
  }

  def loadData(
                trainFile: String,
                testFile: String,
                spark: SparkSession
              ): (DataFrame, DataFrame) = {


    val trainDF = spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", true)
      .load(trainFile)

    val testDF = spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", true)
      .load(testFile)

    (trainDF, testDF)
  }

}