import json
from datasets import load_dataset, DatasetDict, concatenate_datasets, DownloadConfig

# 通用字段映射表，定义不同源字段到目标字段的映射关系
FIELD_MAPPING = {
    "instruction": ["instruction", "abstract", "content", "title", "act","chinese"],
    "input": ["input", "title", "path", "url"],
    "output": ["output", "translation", "markdown", "text","english","content"],
    "system": ["system", "prompt", "repo_name"],
    "history": [
        []
    ]
}

def load_and_concatenate_datasets_train(dataset_train_configs):
    """
    Load and concatenate multiple datasets based on provided configurations.

    Args:
        dataset_train_configs (list of dict): List of dataset configurations, each containing
                                         'name' and optional 'split' and 'download_config'.

    Returns:
        Dataset: Concatenated dataset.
    """
    datasets = []
    for config in dataset_train_configs:
        dataset = load_dataset(
            config['name'],
            split=config.get('split', 'train'),  # Default to 'train' if split is not provided
            download_config=config.get('download_config')
        )
        datasets.append(dataset)
    
    return concatenate_datasets(datasets)

def load_and_concatenate_datasets_validation(dataset_validation_configs):
    """
    Load and concatenate multiple datasets based on provided configurations.

    Args:
        dataset_validation_configs (list of dict): List of dataset configurations, each containing
                                         'name' and optional 'split' and 'download_config'.

    Returns:
        Dataset: Concatenated dataset.
    """
    datasets = []
    for config in dataset_validation_configs:
        dataset = load_dataset(
            config['name'],
            split=config.get('split', 'validation'),  # Default to 'test' if split is not provided
            download_config=config.get('download_config')
        )
        datasets.append(dataset)
    
    return concatenate_datasets(datasets)

def map_features(example):
    """
    Map dataset features to the target format for training.

    Args:
        example (dict): A single example from the dataset.

    Returns:
        dict: Example mapped to the target format.
    """
    # 获取第一个非空的字段值，根据优先级顺序
    def get_first_non_empty(field_list):
        for field in field_list:
            if field in example and example[field]:
                value = example[field]
                # 如果字段值是字典或列表，转换为 JSON 字符串
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                return value
        return ''  # 如果所有字段为空，返回空字符串

    # 生成 history 列表
    def generate_history():
        history = []
        for history_pair in FIELD_MAPPING["history"]:
            # 确保 history_pair 至少有 2 个元素
            if len(history_pair) >= 2:
                human_instruction = example.get(history_pair[0], '')
                model_response = example.get(history_pair[1], '')
                if human_instruction or model_response:  # 如果任意一个非空，则添加到历史记录中
                    # 如果是字典或列表，转换为 JSON 字符串
                    if isinstance(human_instruction, (dict, list)):
                        human_instruction = json.dumps(human_instruction)
                    if isinstance(model_response, (dict, list)):
                        model_response = json.dumps(model_response)
                    
                    history.append([human_instruction, model_response])
        return history if history else None

    # 使用 FIELD_MAPPING 提取每个目标字段的值
    mapped_data = {
        "instruction": get_first_non_empty(FIELD_MAPPING["instruction"]),
        "input": get_first_non_empty(FIELD_MAPPING["input"]),
        "output": get_first_non_empty(FIELD_MAPPING["output"]),
        "system": get_first_non_empty(FIELD_MAPPING["system"]),
        "history": generate_history()
    }

    # 移除 history 为 None 的情况
    if not mapped_data["history"]:
        del mapped_data["history"]
    
    # 仅保留需要的字段
    target_fields = ["instruction", "input", "output", "system", "history"]
    return {k: v for k, v in mapped_data.items() if k in target_fields}

def dataset_train_prepare():
    """
    Load, concatenate, and prepare training and validation datasets.

    Returns:
        DatasetDict: A dictionary containing train and valid datasets in target format.
    """
    # Define the configurations for the datasets to be loaded
    dataset_train_configs = [
        # {
        #     "name": "huggingface-course/codeparrot-ds-train",
        #     "split": "train",
        #     "download_config": DownloadConfig(resume_download=True)
        # },
        # {
        #     "name": "neuralwork/arxiver",
        #     "split": "train",
        #     "download_config": DownloadConfig(resume_download=True)
        # },
        {
            "name": "fka/awesome-chatgpt-prompts",
            "split": "train"
        },
        {
            "name": "DDDSSS/en-zh-dataset",
            "split": "train"
        },
        {
            "name": "nickmuchi/trade-the-event-finance",
            "split": "train"
        },
        {
            "name": "sujet-ai/Sujet-Finance-Instruct-177k",
            "split": "train"
        },
        {
            "name": "Aye10032/zh-en-translate-20k",
            "split": "train"
        },
        {
            "name": "suolyer/cnki_summary",
            "split": "test"
        },
        {
            "name": "xiaodongguaAIGC/alpaca_en_zh_ruozhiba",
            "split": "train"
        }
    ]

    dataset_validation_configs = [
        {
            "name": "huggingface-course/codeparrot-ds-valid",
            "split": "validation",
            "download_config": DownloadConfig(resume_download=True)
        },
        {
            "name": "causal-lm/finance",
            "split": "validation"
        },
        {
            "name": "nickmuchi/trade-the-event-finance",
            "split": "test"
        },
        {
            "name": "Duxiaoman-DI/FinanceIQ",
            "splie": "validation"
        },
        {
            "name": "Aye10032/zh-en-translate-20k",
            "split": "validation"
        },
        {
            "name": "suolyer/cnki_summary",
            "split": "validation"
        }
    ]


    # Load and concatenate training datasets
    ds_train = load_and_concatenate_datasets_train(dataset_train_configs)

    # Load validation dataset
    # ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
    ds_valid = load_and_concatenate_datasets_validation(dataset_validation_configs)
    
    # Map features to the target format
    ds_train = ds_train.map(map_features, remove_columns=ds_train.column_names)
    ds_valid = ds_valid.map(map_features, remove_columns=ds_valid.column_names)

    # Create a DatasetDict to hold the training and validation datasets
    raw_datasets = DatasetDict({
        "train": ds_train,
        "valid": ds_valid,
    })
    
    return raw_datasets

def main():
    raw_datasets = dataset_train_prepare()
    print(raw_datasets)

if __name__ == "__main__":
    main()
