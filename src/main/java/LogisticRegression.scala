import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object Diabetes {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val fileName = "diabets.csv"
    val sparkSession = SparkSession
      .builder()
      .appName("Diabetes logistic regression")
      .master("local[2]")
      .getOrCreate()

    val df = sparkSession.read
      .option("header", "true")
      .option("delimiter", ",")
      .option("nullValue", "")
      .option("treatEmptyValuesAsNulls", "true")
      .option("inferSchema", "true")
      .csv(fileName)

    df.show(10)

    df.printSchema()

    //Feature Extraction

    val DFAssembler = new VectorAssembler().

      setInputCols(Array(
        "pregnancy", "glucose", "arterial pressure",
        "thickness of TC", "insulin", "body mass index",
        "heredity", "age"))
      .setOutputCol("features")

    val features = DFAssembler.transform(df)
    features.show(10)

    val labeledTransformer = new StringIndexer().setInputCol("diabet").setOutputCol("label")

    val labeledFeatures = labeledTransformer.fit(features).transform(features)

    //labeledFeatures.show(10)

    // Split data into training (60%) and test (40%)

    val splits = labeledFeatures.randomSplit(Array(0.6, 0.4), seed = 11L)
    val trainingData = splits(0)
    val testData = splits(1)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    //val results = new mutable.HashMap[Double, Double]()
    var lr = new LogisticRegression()
        .setMaxIter(500)
        .setRegParam(0.01)
        .setElasticNetParam(0.7)

    var model = lr.fit(trainingData)

    //Make predictions on test data
    var predictions = model.transform(testData)

    //predictions.show(100)

    //Evaluate the precision and recall
    val countProve = predictions.where("label == prediction").count()
    val count = predictions.count()

    println(s"Count of true predictions: $countProve Total Count: $count")

    val accuracy = evaluator.evaluate(predictions)

    println(s"Accuracy = $accuracy")
  }
}