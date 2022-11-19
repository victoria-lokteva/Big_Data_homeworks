
def avg_salaries(salaries: list, json_name: str = "salaries.json"):
    salaries = [x.split(' ') for x in salaries]
    salaries = [(x[0], int(x[1]), x[2]) for x in salaries]
    columns = ["Name", "Salary", "Month"]
    dataframe = spark.createDataFrame(salaries, columns)
    mean_salaries = dataframe.groupBy("Name").avg("Salary")
    mean_salaries.write.json(json_name)
    return mean_salaries
