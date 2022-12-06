from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes, RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

name = '/Users/victorialokteva/Downloads/titanic.csv'
schema = StructType([
    StructField("pclass", IntegerType(), True),
    StructField("survived", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("sibsp", IntegerType(), True),
    StructField("parch", IntegerType(), True),
    StructField("ticket", IntegerType(), True),
    StructField("fare", IntegerType(), True),
    StructField("cabin", StringType(), True),
    StructField("embarked", StringType(), True),
    StructField("boat", IntegerType(), True),
    StructField("body", IntegerType(), True),
    StructField("homedest", StringType(), True)])

df = spark.read.csv(name, sep =';', header=True, schema=schema)
train, test = df.randomSplit([0.6, 0.4], seed=2)


lr = LinearRegression()
lr_model = lr.fit(train)


lsvc = LinearSVC(maxIter=10, regParam=0.1)
lsvc = lsvc.fit(train)


nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
nb = nb.fit(train)


rf = RandomForestClassifier(seed=42)
grid = ParamGridBuilder().addGrid(rf.maxDepth, [1, 4]).addGrid(rf.numTrees, [3, 15]).build()
evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
    parallelism=2)
