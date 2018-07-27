/////////////////////////////////////
// LOGISTIC REGRESSION PROJECT  ////
///////////////////////////////////

//  In this project i will be working with a advertising data set, indicating whether or not a particular internet user clicked on an Advertisement.
//  Try to create a model that will predict whether or not they will click on an ad based off the features of that user.
//  This data set contains the following features:
//    'Daily Time Spent on Site': consumer time on site in minutes
//    'Age': cutomer age in years
//    'Area Income': Avg. Income of geographical area of consumer
//    'Daily Internet Usage': Avg. minutes a day consumer is on the internet
//    'Ad Topic Line': Headline of the advertisement
//    'City': City of consumer
//    'Male': Whether or not consumer was male
//    'Country': Country of consumer
//    'Timestamp': Time at which consumer clicked on Ad or closed window
//    'Clicked on Ad': 0 or 1 indicated clicking on Ad


////////////////////////////
/// LOADING THE DATA //////
//////////////////////////

// Import SparkSession and Logisitic Regression
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

// code to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Creating a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Reading the Advertising csv file.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("advertising.csv")

// The Schema of the DataFrame
data.printSchema()

///////////////////////
/// Display Data /////
/////////////////////

// Print out a sample row of the data
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

//    - Renaming the Clicked on Ad column to "label"
//    - Grabing the following columns "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"
//    - Creating a new column called Hour from the Timestamp containing the Hour of the click

val timedata = data.withColumn("Hour",hour(data("Timestamp")))

val logregdata = (timedata.select(data("Clicked on Ad").as("label"),
                    $"Daily Time Spent on Site", $"Age", $"Area Income",
                    $"Daily Internet Usage",$"Hour",$"Male"))


// Importing VectorAssembler and Vectors
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// Creating a new VectorAssembler object called assembler for the feature
// columns as the input Set the output column to be called features
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income",
                  "Daily Internet Usage","Hour"))
                  .setOutputCol("features") )


// randomSplit to create a train test split of 70/30
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)


///////////////////////////////////
// Setting Up the Pipeline ///////
/////////////////////////////////

// Importing Pipeline
import org.apache.spark.ml.Pipeline

// Creating a new LogisticRegression object called lr
val lr = new LogisticRegression()

// Creating a new pipeline with the stages: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler, lr))

// Fitting the pipeline to training set.
val model = pipeline.fit(training)

// Results on Test Set with transform
val results = model.transform(test)

////////////////////////////////////
//// MODEL EVALUATION /////////////
//////////////////////////////////

// For Metrics and Evaluation importing MulticlassMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Converting the test results to an RDD using .as and .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiating a new MulticlassMetrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Printing out the Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
