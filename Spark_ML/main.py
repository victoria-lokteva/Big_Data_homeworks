from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes, RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler,  MinMaxScaler, VectorAssembler
from pyspark.sql.functions import mean as _mean, col
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf

name = '/Users/victorialokteva/Downloads/mushrooms.csv'
df = spark.read.csv(name, header=True)

# переведем таргет в целые числа

def class_to_num(cl):
    if cl == 'e':
        return 0
    return 1

class_map = udf(class_to_num, IntegerType())
df = df.withColumn("class", class_map("class"))

# Сделаем one-hot encoding

features = ["cap-shape", "cap-surface", "cap-color", "bruises", "odor",
       "gill-attachment", "gill-spacing", "gill-size", "gill-color",
       "stalk-shape", "stalk-root", "stalk-surface-above-ring",
       "stalk-surface-below-ring", "stalk-color-above-ring",
       "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
       "ring-type", "spore-print-color", "population", "habitat"]

for f in features:
    Indexer = StringIndexer(inputCol=f,
                                outputCol=f+'Index',
                                handleInvalid="keep")  
    df = Indexer.fit(df.select(f)).transform(df.select("class", *features)).drop(f) 
    df = df.withColumnRenamed(f+'Index', f)
    
    



vectorAssembler = VectorAssembler(inputCols = features, outputCol = 'features')
df = vectorAssembler.transform(df)
df = df.select(['features', 'class'])

    
# разобъем на тренировочную и тестовую выборки
train, test = df.randomSplit([0.6, 0.4], seed=2)

# Линейная решрессия

lr = LogisticRegression(labelCol="class", featuresCol="features")
lr_model = lr.fit(train)

rawPredictions = lr_model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction",
                                              metricName="accuracy")

accuracy = evaluator.evaluate(rawPredictions)
print("Test accuracy ", accuracy)

# SVM

lsvc = LinearSVC(labelCol="class", featuresCol="features", maxIter=10, regParam=0.1)
lsvc = lsvc.fit(train)

rawPredictions = lsvc.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction",
                                              metricName="accuracy")

accuracy = evaluator.evaluate(rawPredictions)
print("Test accuracy ", accuracy)
    
    
# Random Forest

rf = RandomForestClassifier(labelCol="class", featuresCol="features", seed=2)
grid = ParamGridBuilder().addGrid(rf.maxDepth, [1, 4]).addGrid(rf.numTrees, [3, 15]).build()
evaluator = MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction",
                                              metricName="accuracy")
cv = CrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
    parallelism=2)

cv = cv.fit(train)
print(cv.bestModel)
print(cv.avgMetrics)
rawPredictions = cv.transform(test)


accuracy = evaluator.evaluate(rawPredictions)
print("Test accuracy ", accuracy)
