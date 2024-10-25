# import pyarrow as pa
# import pyarrow.parquet as pq
# import pandas as pd
# import numpy as np
# import json

# # 读取原始 Parquet 文件
# table = pq.read_table(r'O:\Applications\Installations\AI\AImodels\huggingface\datasets\DPO-ShareGPT-computer-zh-reject-en\data\train-00000-of-00001.parquet')

# # 将表格转换为 Pandas DataFrame
# df = table.to_pandas()

# print(df.head(2))

# # 修改列名
# # df = df.rename(columns={'system': 'input', 'prompt': 'instruction', 'answer': 'output'})

# # 使用 DataFrame.apply 将所有的 ndarray 转换为列表
# df = df.apply(lambda x: x.apply(lambda y: y.tolist() if isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)) else y))

# # 将 DataFrame 转换为字典列表
# data = df.to_dict(orient='records')

# # 保存为新的 JSON 文件
# with open(r'O:\Code\GithubProject\LLaMA-Factory\data\DPO-ShareGPT-computer-zh.json', 'w', encoding='utf-8') as json_file:
#     json.dump(data, json_file, ensure_ascii=False, indent=4)

# print("数据已保存到 modified_data.json")

#-------------------------------------------------------------------------------------------
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json

# 读取原始 Parquet 文件
table = pq.read_table(r'O:\Applications\Installations\AI\AImodels\huggingface\datasets\DPO-ShareGPT-computer-zh-reject-en\data\train-00000-of-00001.parquet')

# 将表格转换为 Pandas DataFrame
df = table.to_pandas()

print(f"原始数据集{df.head(2)}")  # 打印前两行数据，检查格式

# 使用 DataFrame.apply 将所有的 ndarray 转换为列表
df = df.apply(lambda x: x.apply(lambda y: y.tolist() if isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)) else y))

# 将 DataFrame 转换为字典列表
data = df.to_dict(orient='records')

# 创建转换函数，将原始格式转换为 sharegpt 格式
def convert_to_sharegpt(data):
    sharegpt_data = []

    for entry in data:
        # 提取数据
        system_value = entry.get("system", "").strip()
        prompt_value = entry.get("prompt", "").strip()
        chosen_value = entry["answer"][0] if "answer" in entry and len(entry["answer"]) > 0 else ""
        rejected_value = entry["answer"][1] if "answer" in entry and len(entry["answer"]) > 1 else ""

        # 如果 system 有内容，将其添加到 prompt 中，并在后面换行
        if system_value:
            instruction_value = f"{system_value}\n{prompt_value}"
        else:
            instruction_value = prompt_value
            
        # 构建 sharegpt 格式的数据
        sharegpt_entry = {
            "instruction": [
                {
                    "from": "human",
                    "value": instruction_value
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_value
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_value
            }
        }

        # 添加到列表中
        sharegpt_data.append(sharegpt_entry)

    return sharegpt_data

# 进行转换
sharegpt_data = convert_to_sharegpt(data)

# 保存为新的 JSON 文件
output_file = r'O:\Code\GithubProject\LLaMA-Factory\data\DPO-ShareGPT-computer-zh-sharegpt.json'
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(sharegpt_data, json_file, ensure_ascii=False, indent=4)

print(f"数据已保存到 {output_file}")
#--------------------------------------------------------------------------