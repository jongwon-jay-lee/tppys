import pandas as pd
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType
from transformers import AutoTokenizer
from lbox_test_tokenizer import tokenize

exec(open("./lbox_test_tokenizer.py").read())

"""
def tokenize(record):
    return record[1], len(tokenizer(record[3])["input_ids"])
"""


def main():
    num_workers = 4
    spark = SparkSession.builder.master(f"local[{num_workers}]").appName("lbox_cnt").getOrCreate()
    
    # TODO: load_dataset 으로 연것은 메모리에 파일을 올린것임
    #  즉, spark가 지향하는 바와 맞지 않음
    #  수정 필요!!
    train_data_cn = load_dataset("lbox/lbox_open", "casename_classification")["train"]
    total_len = len(train_data_cn)

    num_partitions = total_len // num_workers
    
    df = spark.createDataFrame(
        data=train_data_cn.to_pandas(),
        schema=train_data_cn.column_names,
    )
    
    # df = df.rdd.glom().collect()    
    # records = df.repartition(4).rdd.glom()
    records = df.repartition(num_workers).rdd

    tokenizer = AutoTokenizer.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )

    # records = records.map(lambda x: (x[1], len(x[3].split()))).groupByKey()
    # result = records.map(lambda x: (x[1], len(tokenizer(x[3])["input_ids"]))).reduceByKey(lambda x, y: x+y).collect()
    result = records.map(tokenize).reduceByKey(lambda x, y: x+y).collect()
    print(result)


@pandas_udf(IntegerType())
def slen(s: pd.Series) -> pd.Series:
    tmp = s
    return s.str.len()


if __name__ == "__main__":
    main()
