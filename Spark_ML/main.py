from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes, RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler,  MinMaxScaler, VectorAssembler
from pyspark.sql.functions import mean as _mean, col
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf

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

# заполним пропуски (body заполним средним, порт посадки - самым распространенным значением, а для возраств введем большое значение)
df_stats = df.select(_mean(col('body')).alias('mean')).collect()
mean_body = df_stats[0]['mean']
df = df.na.fill({'age': 150, 'cabin': 'unknown', 'embarked':'S', 'homedest':'unknown',
                'body': mean_body, 'boat': -1, "sex": 'unknown', 'fare': -1, 
                 'parch': -1, 'sibsp':-1, 'ticket': 'unknown'})

# Сделаем one-hot encoding и нормализацию

unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
for i in ["age", "body"]:
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")
    pipeline = Pipeline(stages=[assembler, scaler])
    df = pipeline.fit(df).transform(df).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i+"_Vect")

    
# разобъем на тренировочную и тестовую выборки
train, test = df.randomSplit([0.6, 0.4], seed=2)

# Линейная решрессия

lr = LinearRegression()
lr_model = lr.fit(train)

# SVM

lsvc = LinearSVC(maxIter=10, regParam=0.1)
lsvc = lsvc.fit(train)

# NAive Bayes

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
nb = nb.fit(train)

# Random Forest

rf = RandomForestClassifier(seed=42)
grid = ParamGridBuilder().addGrid(rf.maxDepth, [1, 4]).addGrid(rf.numTrees, [3, 15]).build()
evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
    parallelism=2)
