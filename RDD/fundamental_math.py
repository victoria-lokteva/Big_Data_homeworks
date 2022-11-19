values = [-1, 1, 3, 2, 2, 150, 1, 2, 3, 2, 2, 1, 1, 1, -100, 2, 2, 3, 4, 1, 1, 3, 4]

def filter_and_check(values):
    rdd = sc.parallelize(values, 3)
    mean = rdd.mean()
    std = rdd.stdev()
    # Величины, отклоняющийся не более чем на 3 стандартных отклонения
    rdd_filtered= rdd.filter(lambda x: x - mean <= 3*std)
    # проверим нормальность
    perc = 100 * rdd_filtered.count()/rdd.count()
    if perc >= 99.7:
        print("Данные распределены нормально")
    else:
        print("Данные не распределены нормально")
    return  rdd_filtered.collect()



