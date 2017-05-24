import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.feature.{ StringIndexer, IndexToString, VectorIndexer }
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import scala.util.MurmurHash


val dataRaw = spark.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load("/home/administrator/Training_set_values.csv")

val dataAll=dataRaw.na.fill("unknown")

val columnsLabels = dataAll.columns

val filteredCols = columnsLabels.filter(!_.contains("id")).filter(!_.contains("date_recorded")).filter(!_.contains("wpt_name")).filter(!_.contains("subvillage")).filter(!_.contains("funder")).filter(!_.contains("installer")).filter(!_.contains("scheme_name")).filter(!_.contains("ward")).filter(!_.contains("quantity_group")).filter(!_.contains("quality_group")).filter(!_.contains("management")).filter(!_.contains("management_group")).filter(!_.contains("payment")).filter(!_.contains("payment_type")).filter(!_.contains("extraction_type")).filter(!_.contains("extraction_type_group")).filter(!_.contains("recorded_by")).filter(!_.contains("scheme_name")).filter(!_.contains("scheme_management"))

val data = dataAll.select(filteredCols.head, filteredCols.tail: _*)

val modifiedData = data.rdd.map(line=>line.toString().replace("[","").replace("]",""))

val desiredData = modifiedData.map(line=>(MurmurHash.stringHash(line.split(",")(0)).toDouble,MurmurHash.stringHash(line.split(",")(1)).toDouble,MurmurHash.stringHash(line.split(",")(2)).toDouble,MurmurHash.stringHash(line.split(",")(3)).toDouble,MurmurHash.stringHash(line.split(",")(4)).toDouble,MurmurHash.stringHash(line.split(",")(5)).toDouble,MurmurHash.stringHash(line.split(",")(6)).toDouble,MurmurHash.stringHash(line.split(",")(7)).toDouble,MurmurHash.stringHash(line.split(",")(8)).toDouble,MurmurHash.stringHash(line.split(",")(9)).toDouble,MurmurHash.stringHash(line.split(",")(10)).toDouble,MurmurHash.stringHash(line.split(",")(11)).toDouble,MurmurHash.stringHash(line.split(",")(12)).toDouble,MurmurHash.stringHash(line.split(",")(13)).toDouble,MurmurHash.stringHash(line.split(",")(14)).toDouble,MurmurHash.stringHash(line.split(",")(15)).toDouble,MurmurHash.stringHash(line.split(",")(16)).toDouble,MurmurHash.stringHash(line.split(",")(17)).toDouble,MurmurHash.stringHash(line.split(",")(18)).toDouble,MurmurHash.stringHash(line.split(",")(19)).toDouble,MurmurHash.stringHash(line.split(",")(20)).toDouble,MurmurHash.stringHash(line.split(",")(21)).toDouble))

val vecData = desiredData.map(line=>(line._1,line._2,line._3,line._4,line._5,line._6,line._7,line._8,line._9,line._10,line._11,line._12,line._13,line._14,line._15,line._16,line._17,line._18,line._19,line._20,line._21,line._22))

val vectDataDF = vecData.toDF("amount_tsh","gps_height","longitude","latitude","num_private","basin","region","region_code","district_code","lga","population","public_meeting","permit","construction_year","water_quality","quantity","source","source_type","source_class","waterpoint_type","waterpoint_type_group","status_group")

val assembler = new VectorAssembler().setInputCols(vectDataDF.columns).setOutputCol("features") 

val output = assembler.transform(vectDataDF)

val labelIndexer = new StringIndexer().setInputCol("status_group").setOutputCol("indexedLabel").fit(output)

val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(output)

val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3))

val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(100)

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)

println("Test Error = " + (1.0 - accuracy))
