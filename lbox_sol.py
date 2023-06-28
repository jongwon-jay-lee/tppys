import os
import pandas as pd
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from transformers import AutoTokenizer
from pyspark.sql.types import IntegerType


num_workers = 4
spark = SparkSession.builder.master(f"local[{num_workers}]").appName("lbox_cnt").getOrCreate()

input_filepath = "./lbox_dataset.json"
if not os.path.exists(input_filepath):
    train_data_cn = load_dataset("lbox/lbox_open", "casename_classification")["train"]
    train_data_cn.to_json(f"{input_filepath}")
df = spark.read.json(input_filepath)


tokenizer = AutoTokenizer.from_pretrained(
    "kakaobrain/kogpt",
    revision="KoGPT6B-ryan1.5b-float16",
    bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]", mask_token="[MASK]"
)


def count_tokens(text, tokenizer):
    length = tokenizer(text, return_length=True, return_attention_mask=False, return_token_type_ids=False)["length"][0]
    return length


@pandas_udf(IntegerType())
def count_tokens_udf(text: pd.Series) -> pd.Series:
    num_tokens = text.apply(lambda string: count_tokens(string, tokenizer))
    return num_tokens

num_rows = df.count()
df = df.withColumn("num_tokens", count_tokens_udf(col("facts")))     # transformation

casetype_num_tokens = df.groupBy("casetype").sum("num_tokens")

collected = casetype_num_tokens.collect()
print(collected)

"""
necessary_fields = ["casetype", "num_tokens"]
df = df[necessary_fields]

df.agg({"num_tokens": "sum"}).show()

df.createOrReplaceTempView("df")
spark.sql("SELECT COUNT(*) FROM df").show()
"""