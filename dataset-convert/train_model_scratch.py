import json
from datasets import load_dataset, DatasetDict, concatenate_datasets, DownloadConfig, Value
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 通用字段映射表，定义不同源字段到目标字段的映射关系
FIELD_MAPPING = {
    "instruction": ["instruction", "title", "act","chinese","message"],
    "input": ["input", "title", "path", "url","labels"],
    "output": ["output", "article", "translation", "markdown", "text","english","content", "tools"],
    "system": ["system", "abstract", "prompt", "repo_name"],
    "history": [
        "history"
    ]
}

def standardize_features(dataset):
    if "id" in dataset.features:
        return dataset.cast_column("id", Value(dtype="string"))
    return dataset

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
        print(f"Features of {config['name']}: {dataset.features}")
        dataset = standardize_features(dataset)     # Standardize the 'id' column
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
        print(f"Features of {config['name']}: {dataset.features}")
        dataset = standardize_features(dataset)      # Standardize the 'id' column
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
        # "history": generate_history()
        "history": get_first_non_empty(FIELD_MAPPING["history"])
    }

    # 移除 history 为 None 的情况
    if not mapped_data["history"]:
        del mapped_data["history"]

    # print(mapped_data)
    
    # 仅保留需要的字段
    target_fields = ["instruction", "input", "output", "system", "history"]

    if not mapped_data["output"]:
        mapped_data["output"] = "Sorry, I may need more information."  # or some meaningful default value    

    return {k: v for k, v in mapped_data.items() if k in target_fields}

def dataset_train_prepare():
    """
    Load, concatenate, and prepare training and validation datasets.

    Returns:
        DatasetDict: A dictionary containing train and valid datasets in target format.
    """
    # Define the configurations for the datasets to be loaded
    dataset_train_configs = [
        {
            "name": "huggingface-course/codeparrot-ds-train",
            "split": "train",
            "download_config": DownloadConfig(resume_download=True)
        },
        {
            "name": "neuralwork/arxiver",
            "split": "train",
            "download_config": DownloadConfig(resume_download=True)
        },
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
            "name": "ZixuanKe/sujet-finance-instruct-177k-clean",
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
        },
        {
            "name": "yuyuans/English-Chinese",
            "split": "train"
        },
        {
            "name": "wlhb/Transaltion-Chinese-2-English"
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
            "name": "suolyer/cnki_summary",
            "split": "validation"
        },
        {
            "name": "Aye10032/zh-en-translate-20k",
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

    # print(ds_train[0])
    
    return raw_datasets

def tokenize(tokenizer, element):
    context_length = 128
    # tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    if "output" not in element or not element["output"]:
        return {"input_ids": []} 

    outputs = tokenizer(
        element["output"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    # input_batch = []
    # for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
    #     if length == context_length:
    #         input_batch.append(input_ids)
    input_batch = [input_ids for length, input_ids in zip(outputs["length"], outputs["input_ids"]) if length == context_length]

    # print(f"Input IDs length: {len(outputs['input_ids'])}")
    # print(f"Input chunk lengths: {(outputs['length'])}")
    # print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")  
    return {"input_ids": input_batch}

def model_pretrain(tokenizer, tokenized_datasets):
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="gpt-pretrain",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    # Start training
    trainer.train()


def main():
    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    raw_datasets = dataset_train_prepare()
    # print(raw_datasets)
    # for key in raw_datasets["train"][0]:
    #     print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

    # tokenized_datasets = raw_datasets.map(
    # tokenize(tokenizer,raw_datasets), batched=True, remove_columns=raw_datasets["train"].column_names
    # )
    tokenized_datasets = raw_datasets.map(
        lambda element: tokenize(tokenizer, element),  # Use a lambda to pass each element
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    print(tokenized_datasets)

    # model_pretrain(tokenizer, tokenized_datasets)
    # print("Pretrain completed")

    # tokenizer.pad_token = tokenizer.eos_token
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
    # for key in out:
    #     print(f"{key} shape: {out[key].shape}")


        
if __name__ == "__main__":
    main()
