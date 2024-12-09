
import os

import pickle
import re
import csv
import json
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
import argparse

def baseline_create_dictionary_of_eval_datasets(
        params: argparse.Namespace) -> list:
    """
    We create dictionary with the description of each validation set we want
    """

    eval_data_list = []

    eval_data_list = _append_anli_dict(eval_data_list)

    return eval_data_list

def _append_anli_dict(eval_data_list: list) -> list:
    """
    Appending ANLI dictionary
    """

    for split in ['dev_r1', 'dev_r2', 'dev_r3', 'test_r1', 'test_r2', 'test_r3']:
        eval_data_list.append({'description': 'anli', 'split_name': split})

    return eval_data_list


def baseline_prepare_all_dataloaders(
        eval_data_list,
        params,
        tokenizer):
    """
    Creates our dataset_config dictionary, with parameters for our datasets
    
    We then create and return the train, dev, test and evaluation dataloaders
    """
    # Set the columns required for our tokenized data
    dataset_config = {}

    dataset_config['data_cols'] = [
            'input_ids', 
            'token_type_ids', 
            'attention_mask', 
            'label']

    train_dataloaders = {}
 
    assert params.train_data == "anli"

    train_dataloaders.update({'anli_train': _prepare_huggingface_data_train(
        params,
        tokenizer,
        {'description': 'anli',
            'splits': params.train_splits})})

    eval_dataloaders = {}

    # Creating evaluation dataloaders (from Huggingface)
    for dataset_dict in eval_data_list:
        eval_dataloaders.update(
            {dataset_dict['description'] + "_" + dataset_dict['split_name']: \
                _prepare_huggingface_data(
                    params,
                    tokenizer,
                    dataset_dict)})

    return train_dataloaders, eval_dataloaders

def _tokenize_data(loaded_data, batch_size, tokenizer):
    """
    We tokenize the data
    """

    # Information to store with dataset

    loaded_data = loaded_data.map(lambda x: tokenizer(
            x['premise'],
            x['hypothesis'], 
            truncation=True, 
            padding=True),
        batched=True, 
        batch_size=batch_size)

    return loaded_data

def _load_hf_dataset(dataset_dict):

    # Load dataset
    print("Description:", dataset_dict['description'])
    print("Split name:", dataset_dict['split_name'])
    loaded_data = load_dataset(
            dataset_dict['description'], 
            split=dataset_dict['split_name'])
    
    return loaded_data

def _filter_labels(dataset):
    # Remove examples with no gold label

    dataset = dataset.filter(
            lambda example: example['label'] in [0, 1, 2])

    return dataset

def _create_dataloader(dataset, batch_size):

    dataset.set_format(
            type='torch',
            columns=[
            'input_ids',
            'token_type_ids',
            'attention_mask',
            'label'])

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size)

    return dataloader

def _prepare_huggingface_data_train(
        params,
        tokenizer,
        dataset_dict):
    """
    From dictionary with dataset description and split name, returns dataloader for huggingface dataset
    """

    all_train = []

    train_splits = "_".join(params.train_splits)

    for split_name in dataset_dict['splits']:
            all_train.append(_load_hf_dataset(
                {'description': dataset_dict['description'], 'split_name': split_name}))

    train_data = concatenate_datasets(all_train)
    train_data = _filter_labels(train_data)
    train_data = train_data.shuffle(seed=params.random_seed)
    train_data = _tokenize_data(train_data, params.batch_size, tokenizer)
    dataloader = _create_dataloader(train_data, params.batch_size)

    return dataloader

def _prepare_huggingface_data(
        params,
        tokenizer,
        dataset_dict):
    """
    From dictionary with dataset description and split name, returns dataloader for huggingface dataset
    """


    huggingface_dataset = _load_hf_dataset(dataset_dict)
    huggingface_dataset = _filter_labels(huggingface_dataset)
    huggingface_dataset = _tokenize_data(huggingface_dataset, params.batch_size, tokenizer)
    dataloader = _create_dataloader(huggingface_dataset, params.batch_size)

    return dataloader


def _load_data(name):
    data = []
    a_file = open(name, "r")
    data = json.load(a_file)
    return data

def read_jsonl(filename):
    data=[]
    with open(filename, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


