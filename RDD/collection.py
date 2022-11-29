import numpy as np


def log(x):
    return np.log(x)
    
def f(x):
    return(x%100, log(x))

def even_lst(lst):
    cnt = 0
    for x in lst:
        digit = (int(x*10)) % 10
        if (digit)%2==0:
            cnt+=1
    return cnt
  
  
rdd = sc.parallelize(np.arange(1, 100000), 10)
name = "/Users/victorialokteva/Downloads/rdd_as_text.txt"
rdd.saveAsTextFile(name)
rdd = sc.textFile(name) 
rdd = rdd.map(float)

rdd2 = rdd.map(f)
rdd2 = rdd2.groupByKey().mapValues(list)
rdd_counted = rdd2.mapValues(even_lst)

rdd_counted.collect()
