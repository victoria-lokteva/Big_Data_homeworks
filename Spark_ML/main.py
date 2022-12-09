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
df = spark.read.csv(name, header=True, schema=schema)

# Сделаем one-hot encoding



    
    
features = ['pclass','sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'boat', 'body', 'homedest']
vectorAssembler = VectorAssembler(inputCols = features, outputCol = 'features')
df = vectorAssembler.transform(df)
df = df.select(['features', 'survived'])

    
# разобъем на тренировочную и тестовую выборки
train, test = df.randomSplit([0.6, 0.4], seed=2)

# Линейная решрессия

lr = LogisticRegression(labelCol="survived", featuresCol="features")
lr_model = lr.fit(train)

rawPredictions = lr_model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="survived", predictionCol="prediction",
                                              metricName="accuracy")

accuracy = evaluator.evaluate(rawPredictions)
print("Test accuracy ", accuracy)

# SVM

lsvc = LinearSVC(labelCol="survived", featuresCol="features", maxIter=10, regParam=0.1)
lsvc = lsvc.fit(train)

rawPredictions = lsvc.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="survived", predictionCol="prediction",
                                              metricName="accuracy")

accuracy = evaluator.evaluate(rawPredictions)
print("Test accuracy ", accuracy)
    
    
# Random Forest

rf = RandomForestClassifier(labelCol="survived", featuresCol="features", seed=2)
grid = ParamGridBuilder().addGrid(rf.maxDepth, [1, 4]).addGrid(rf.numTrees, [3, 15]).build()
evaluator = MulticlassClassificationEvaluator(labelCol="survived", predictionCol="prediction",
                                              metricName="accuracy")
cv = CrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
    parallelism=2)

cv = cv.fit(train)
print(cv.bestModel)
print(cv.avgMetrics)
rawPredictions = cv.transform(test)


accuracy = evaluator.evaluate(rawPredictions)
print("Test accuracy ", accuracy)
