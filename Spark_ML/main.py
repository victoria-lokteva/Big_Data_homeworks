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
df = df.na.fill({'age': 100, 'cabin': 'unknown', 'embarked':'S', 'homedest':'unknown',
                'body': mean_body, 'boat': -1, "sex": 'unknown', 'fare': -1, 
                 'parch': -1, 'sibsp':-1, 'ticket': 'unknown'})

# Сделаем one-hot encoding и нормализацию

def emb_map(s):
    if s == "S":
        return 0
    elif s == "C":
        return 1
    elif s == "Q":
        return 2
    return -1

def sex_map(s):
    if s == "F":
        return 0
    elif s == "M":
        return 1
    return -1

def dest_map(s):
    if s == "New York, NY":
        return 0
    elif s == "London":
        return 1
    elif s == "Montreal, PQ":
        return 2
    elif s == "Paris, France":
        return 3
    elif s == "Cornwall / Akron, OH":
        return 4
    return 5

dest_map = udf(dest_map, IntegerType())
df = df.withColumn("homedest", dest_map("homedest"))

emb_map = udf(emb_map, IntegerType())
df = df.withColumn("embarked", emb_map("embarked"))

sex_map = udf(sex_map, IntegerType())
df = df.withColumn("sex", sex_map("sex"))

unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
for i in ["age", "body"]:
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")
    pipeline = Pipeline(stages=[assembler, scaler])
    df = pipeline.fit(df).transform(df).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i+"_Vect")

df = df.drop('age')
df = df.drop('body')
df = df.withColumnRenamed("body_Scaled", "body").withColumnRenamed("age_Scaled", "age")
    
    
features = [['pclass','sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 
            'cabin', 'embarked', 'boat', 'body', 'homedest']]
vectorAssembler = VectorAssembler(inputCols = features, outputCol = 'features')
df = vectorAssembler.transform(df)
df = df.select(['features', 'survived'])

    
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
