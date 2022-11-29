from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType
from pyspark.sql.functions import udf
from pyspark.sql import functions as F

def isCanadaNeighbour(stateName):
    if stateName in ['AK', 'ME', 'NY', 'ID', 'VT', 'MN', 'MI', 'OH',
                     'ND', 'MO', 'WA', 'NH', 'PA']:
        return True
    else:
        return False


udf_isCanadaNeighbour = udf(isCanadaNeighbour, BooleanType()) 
    
name = '/Users/victorialokteva/Downloads/StateNames.csv'
schema = StructType([
    StructField("Id", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Year", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("State", StringType(), True),
    StructField("Count", IntegerType(), True)])

df = spark.read.csv(name,header=False,schema=schema)
df = df.withColumn("isCanadaNeighbour", udf_isCanadaNeighbour("State"))
df = df.filter(F.col("Gender")=='F')
df = df.filter(F.col("Year")>=1914)
df = df.filter(F.col("Year")<=1918)
df = df.filter(F.col("isCanadaNeighbour") == True)

names = df.select('Name').distinct().rdd.map(lambda r: r[0]).collect()


names.write.parquet("result.parquet")
