import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.master("local[2]").appName("concept").getOrCreate()

# print list of full configuration
for property, value in spark.sparkContext.getConf().getAll():
    print(property, value)

# use rdd
simple_rdd = spark.sparkContext.parallelize(["김보섭", "김성현", "이종원", "정두해", "김다인", "조원익"]) # transformation
simple_rdd.collect() # action
simple_rdd.glom().collect()
simple_rdd.repartition(1).glom().collect()
simple_rdd.repartition(2).glom().collect()
simple_rdd.repartition(3).glom().collect()
simple_rdd.repartition(4).glom().collect()
simple_rdd.repartition(10).glom().collect()

# use key-value rdd
key_value_rdd = spark.sparkContext.parallelize([("카카오브레인", "김보섭"), ("신한은행", "김성현"), ("삼성전자", "이종원"), ("카카오브레인", "정두해"), ("카카오브레인", "김다인"), ("삼성전자", "조원익")])
key_value_rdd.glom().collect()
key_value_rdd.groupByKey().mapValues(list).collect()
key_value_rdd.groupByKey().mapValues(list).glom().collect()
key_value_rdd = key_value_rdd.map(lambda item: (item[0], item[-1], len(item[-1]))) # transformation
key_value_rdd.collect() # action

dir(key_value_rdd)

# use dataframe
df = spark.createDataFrame(
    data=[("카카오브레인", "김보섭"), ("신한은행", "김성현"), ("삼성전자", "이종원"), ("카카오브레인", "정두해"), ("카카오브레인", "김다인"), ("삼성전자", "조원익")],
    schema=["company", "name"]
)
df.rdd.glom().collect()
df.repartition(4).rdd.glom().collect()

# use pandas_udf
@pandas_udf(IntegerType())
def slen(s: pd.Series) -> pd.Series:
    return s.str.len()

df = df.withColumn("length", slen(col("name"))) # transformatoin
df.collect() # action
df.show()

df.write.option("compression", "gzip").mode("errorifexists").format("json").save("tmp_output")
