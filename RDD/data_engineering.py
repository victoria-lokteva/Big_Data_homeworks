import numpy as np

def split(x):
    x = x.split(" ")
    return (x[0], int(x[1]))

def avg(x):
    return mean

def avg_salary(salaries):
    rdd = sc.parallelize(salaries, 3)
    # Разобъем строку на имя зарплату и месяц
    rdd = rdd.map(split)
    rdd = rdd.groupByKey().mapValues(list)
    return rdd.mapValues(np.mean).collect()
  
if __name__ == "__main__":

    salaries = [
    "John 1900 January",
    "Mary 2000 January",
    "John 1800 February",
    "John 1000 March",
    "Mary 1500 February",
    "Mary 2900 March"
    "Mary 1600 April",
    "John 2800 April"
    ]

    avg_salary(salaries)
