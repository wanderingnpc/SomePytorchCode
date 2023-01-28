import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator
from torch.utils.data import DataLoader


class Dataset:
    train_dataloader = None
    eval_dataloader = None
    test_dataloader = None


def load_dataset_for_mlm(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize(examples):
        result = tokenizer(examples['text'])
        return result

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // config.chunk_size) * config.chunk_size
        result = {
            k: [t[i: i + config.chunk_size] for i in range(0, total_length, config.chunk_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    all_datasets = []

    for script in config.dataset_scripts:
        raw_dataset = load_dataset(script)
        if config.mini_test:
            raw_dataset['train'] = raw_dataset['train'].shard(1000, 0)
            raw_dataset['validation'] = raw_dataset['validation'].shard(1000, 0)
            raw_dataset['test'] = raw_dataset['test'].shard(1000, 0)
        tokenized_dataset = raw_dataset.map(
            tokenize, batched=True, remove_columns=["text"]
        )
        lm_dataset = tokenized_dataset.map(group_texts, batched=True)
        print(lm_dataset)
        all_datasets.append(lm_dataset)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=config.mlm_probability)

    train_dataloader = DataLoader(
        concatenate_datasets([ds['train'] for ds in all_datasets]),
        shuffle=True,
        batch_size=config.train_batch_size,
        collate_fn=data_collator,
    )

    eval_dataloader = DataLoader(
        concatenate_datasets([ds['validation'] for ds in all_datasets]),
        batch_size=config.eval_batch_size,
        collate_fn=data_collator
    )

    ds = Dataset()
    ds.train_dataloader = train_dataloader
    ds.eval_dataloader = eval_dataloader

    return ds


def load_dataset_for_mlm_cl(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize_function(example):
        len1 = len(tokenizer(example['sentence1'])[0])
        len2 = len(tokenizer(example['sentence2'])[0])
        max_len = max(len1, len2)
        result1 = tokenizer(example['sentence1'], padding='max_length', max_length=max_len)
        result2 = tokenizer(example['sentence2'], padding='max_length', max_length=max_len)
        result = {
            'input_ids1': result1.input_ids,
            'token_type_ids1': result1.token_type_ids,
            'attention_mask1': result1.attention_mask,
            'input_ids2': result2.input_ids,
            'token_type_ids2': result2.token_type_ids,
            'attention_mask2': result2.attention_mask,
        }
        return result

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // config.chunk_size) * config.chunk_size
        result = {
            k: [t[i: i + config.chunk_size] for i in range(0, total_length, config.chunk_size)]
            for k, t in concatenated_examples.items()
        }
        result['input_ids'] = [result['input_ids1'][i] + result['input_ids2'][i] for i in
                               range(len(result['input_ids1']))]
        return result

    all_datasets = []

    for script in config.dataset_scripts:
        raw_dataset = load_dataset(script, name="Parallel")
        if config.mini_test:
            raw_dataset['train'] = raw_dataset['train'].shard(1000, 0)
            raw_dataset['validation'] = raw_dataset['validation'].shard(1000, 0)
            raw_dataset['test'] = raw_dataset['test'].shard(1000, 0)
        tokenized_dataset = raw_dataset.map(tokenize_function, remove_columns=["sentence1", "sentence2"])
        lm_dataset = tokenized_dataset.map(group_texts, batched=True, remove_columns=["input_ids1", "input_ids2"])
        print(lm_dataset)
        all_datasets.append(lm_dataset)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=config.mlm_probability)

    def pair_sentence_data_collator(inp):
        res = data_collator(inp)
        res.pop("attention_mask")
        input_ids = torch.chunk(res.pop('input_ids'), 2, dim=1)
        labels = torch.chunk(res.pop('labels'), 2, dim=1)
        res['input_ids1'] = input_ids[0]
        res['input_ids2'] = input_ids[1]
        res['labels1'] = labels[0]
        res['labels2'] = labels[1]
        return res

    train_dataloader = DataLoader(
        concatenate_datasets([ds['train'] for ds in all_datasets]),
        shuffle=True,
        batch_size=config.train_batch_size,
        collate_fn=pair_sentence_data_collator,
    )

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = pair_sentence_data_collator(features)
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    eval_dataset = concatenate_datasets([ds['validation'] for ds in all_datasets])
    eval_dataset = eval_dataset.map(insert_random_mask,
                                    batched=True,
                                    remove_columns=all_datasets[0]["validation"].column_names)
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids1": "input_ids1",
            "masked_token_type_ids1": "token_type_ids1",
            "masked_attention_mask1": "attention_mask1",
            "masked_labels1": "labels1",
            "masked_input_ids2": "input_ids2",
            "masked_token_type_ids2": "token_type_ids2",
            "masked_attention_mask2": "attention_mask2",
            "masked_labels2": "labels2",
        }
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=default_data_collator
    )

    ds = Dataset()
    ds.train_dataloader = train_dataloader
    ds.eval_dataloader = eval_dataloader

    return ds
