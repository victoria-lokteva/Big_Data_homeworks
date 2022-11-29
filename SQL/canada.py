from pyspark.sql.types import StructType, StructField, IntegerType, StringType

def isCanadaNeighbour(stateName):
    if stateName in ['AK', 'ME', 'NY', 'ID', 'VT', 'MN', 'MI', 'OH',
                     'ND', 'MO', 'WA', 'NH', 'PA']:
        return True
    else:
        return False


name = '/Users/victorialokteva/Downloads/StateNames.csv'
schema = StructType([
    StructField("Id", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Year", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("State", StringType(), True),
    StructField("Count", IntegerType(), True)])

df = spark.read.csv(name,header=False,schema=schema)
df = df.filter(col("Gender")=='F')
df = df.filter(col("Year")>=1914)
df = df.filter(col("Year")<=1918)
df = df.filter(col("State").map(isCanadaNeighbour))
