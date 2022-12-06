from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml.classification import LogisticRegression

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

df = spark.read.csv(name, sep =';', header=False,schema=schema)
train, test = df.randomSplit([0.6, 0.4], seed=2)


lr = LogisticRegression()
