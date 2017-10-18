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

object House {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .config("spark.master", "local")
      .appName("Titanic")
      .getOrCreate()
    import spark.implicits._

    val (train, test) = loadData("D://Do/train.csv", "D://Do/test.csv", spark)

    val numericFeatColNames = Seq( "LotArea", "YearBuilt", "YearRemodAdd",
      "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea",
    "BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars",
    "GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","LotFrontage",
      "MasVnrArea","GarageYrBlt","TotalSF")

    val categoricalFeatColNames = Seq("MSSubClass", "MSZoning", "Street", "Alley","LotShape","LandContour","LotConfig",
    "LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","RoofStyle","RoofMatl",
    "Exterior1st","Exterior2nd","MasVnrType", "ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
    "BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish",
    "GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition","MoSold","YrSold")


    val idxdCategoricalFeatColName = categoricalFeatColNames.map(_ + "Indexed")
    val allIdxdFeatColNames = numericFeatColNames ++ idxdCategoricalFeatColName
    val allFeatColNames = numericFeatColNames ++ categoricalFeatColNames
    val labelColName = "SalePrice"
    val featColName = "Features"
    val idColName = "Id"

    val allPredictColNames = allFeatColNames ++ Seq(idColName)
    val newtest  = test.withColumn("SalePrice",lit("0").cast(DoubleType))
    val dataDFFiltered = train.select(labelColName, allPredictColNames: _*)
    val predictDFFiltered = newtest.select(labelColName, allPredictColNames: _*)

    val allData = dataDFFiltered.union(predictDFFiltered)
    allData.cache()

    val stringIndexers = categoricalFeatColNames.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
        .fit(allData)
    }

    val assembler = new VectorAssembler()
      .setInputCols(Array(allIdxdFeatColNames: _*))
      .setOutputCol(featColName)


    val rf = new RandomForestRegressor()
      .setLabelCol("SalePrice")
      .setFeaturesCol(featColName)

    val pipeline = new Pipeline().setStages(
      (stringIndexers :+  assembler :+ rf ).toArray)


    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxBins, Array(25, 28, 31))
      .addGrid(rf.maxDepth, Array(4, 6, 8))
      .build()


    val evaluator = new RegressionEvaluator()
      .setLabelCol("SalePrice")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)
    // train the model
    predictDFFiltered.show(5)
    val crossValidatorModel = cv.fit(train)

    val predictions = crossValidatorModel.transform(predictDFFiltered)
    predictions.show(5)
    predictions
      .withColumn("SalePrice", col("prediction"))
      .select("Id", "SalePrice")
      .coalesce(1)
      .write
      .format("csv")
      .mode("overwrite")
      .option("header", "true")
      .save("result")
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