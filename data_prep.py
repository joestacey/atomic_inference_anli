

import os

import string
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
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from joblib import Parallel, delayed

filtered_indices = []

HP_DICT_ALL = {
        'anli': 'get_expl/processed_explanations_default_seed_hypconditioned.pkl',
        'control': 'get_expl/processed_explanations_ctrl_hcond_seed_hypconditioned.pkl',
        'rte': 'get_expl/processed_explanations_rte_hcond_seed_hypconditioned.pkl',
        'wnli': 'get_expl/processed_explanations_hcond_wnli_seed_hypconditioned.pkl'
        }

FILE_LOOKUP = {
        'default': 'get_expl/processed_explanations_train_expl_anli.pkl'
        }

def create_dictionary_of_eval_datasets(
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


def prepare_all_dataloaders(
        eval_data_list: list,
        params: argparse.Namespace,
        tokenizer: BertTokenizerFast) -> (dict, dict):
    """
    Creates our dataset_config dictionary, with parameters for our datasets
    
    We then create and return the train, dev, test and evaluation dataloaders
    """


    # Loading explanations for training
    assert params.fact_version in FILE_LOOKUP.keys()

    with open(FILE_LOOKUP[params.fact_version], 'rb') as handle:
        expl_dict = pickle.load(handle)

    # Set the columns required for our tokenized data
    dataset_config = {}

    dataset_config['data_cols'] = [
            'input_ids', 
            'token_type_ids', 
            'attention_mask', 
            'label']

    # Creating our train and eval dataloaders
    train_dataloaders = {}

    if params.train_data == "anli":

        train_dataloaders.update({'anli_train': _prepare_huggingface_data_train(
            expl_dict,
            params,
            tokenizer,
            {'description': 'anli',
                'splits': params.train_splits})})

    eval_dataloaders = {}

    for dataset_dict in eval_data_list:
        eval_dataloaders.update(
            {'test_default_' \
                    + dataset_dict['description'] \
                    + "_" + dataset_dict['split_name']: \
                _prepare_huggingface_data(
                    expl_dict,
                    params,
                    tokenizer,
                    dataset_dict)})

    # Load OOD tests/dev sets
    with open('get_expl/processed_explanations_ctrl_test.pkl', 'rb') as handle:
            expl_control = pickle.load(handle)

    with open('get_expl/processed_explanations_ctrl_dev.pkl', 'rb') as handle:
            expl_control_dev = pickle.load(handle)

    with open('get_expl/processed_explanations_wnli_dev.pkl', 'rb') as handle:
            expl_wnli_dev_ood = pickle.load(handle)

    with open('get_expl/processed_explanations_rte_dev.pkl', 'rb') as handle:
            expl_rte_dev_ood = pickle.load(handle)

    # To evaluate on OOD settings, you need to downlnoad the ctrl, rte and wnli datasets and uncomment below
    #eval_dataloaders.update({'control_test': _get_control(params, expl_control, tokenizer, "ctrl_test.jsonl")})
    #eval_dataloaders.update({'control_dev': _get_control(params, expl_control_dev, tokenizer, "ctrl_dev.jsonl")})
    #eval_dataloaders.update({'rte_dev': _get_rte(params, expl_rte_dev_ood, tokenizer, "rte_dev.tsv")})
    #eval_dataloaders.update({'wnli_dev': _get_wnli(params, expl_wnli_dev_ood, tokenizer, "wnli_dev.tsv")})

    return train_dataloaders, eval_dataloaders

def read_jsonl(filename):
    data=[]
    with open(filename, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


def _reduce_control(control_test, expl_control):

    labels = []
    prems = []
    hyps = []

    for obs in control_test:

        if not expl_control:
            if len(obs['premise']) < 2000:

                if obs['label'] == 'e':
                    labels.append(0)
                elif obs['label'] == 'n':
                    labels.append(1)
                elif obs['label'] == 'c':
                    labels.append(2)

                prems.append(obs['premise'])
                hyps.append(obs['hypothesis'])

        elif obs['premise'] in expl_control.keys():

            if expl_control[obs['premise']] != ['Premise too large'] and len(obs['premise']) < 2000:
                
                if obs['label'] == 'e':
                    labels.append(0)
                elif obs['label'] == 'n':
                    labels.append(1)
                elif obs['label'] == 'c':
                    labels.append(2)

                prems.append(obs['premise'])
                hyps.append(obs['hypothesis'])
    
    return {'premise': prems, 'hypothesis': hyps, 'label': labels}

def _format_wnli(wnli, expl):

    labels = []
    prems = []
    hyps = []

    for obs_no in range(len(wnli)):

        assert wnli['sentence1'][obs_no] in expl.keys()

        if wnli['label'][obs_no] == 1:
            assigned_label = 0
        elif wnli['label'][obs_no] == 0:
            assigned_label = 1

        prems.append(wnli['sentence1'][obs_no])
        hyps.append(wnli['sentence2'][obs_no])
        labels.append(assigned_label)

    return {'premise': prems, 'hypothesis': hyps, 'label': labels}


def _format_rte(rte, expl):

    labels = []
    prems = []
    hyps = []

    for obs_no in range(len(rte)):

        assert rte['sentence1'][obs_no] in expl.keys()

        # Exclude 'nan' label
        if isinstance(rte['label'][obs_no], str):

            prems.append(rte['sentence1'][obs_no])
            hyps.append(rte['sentence2'][obs_no])
            if rte['label'][obs_no] == "not_entailment":
                labels.append(1)
            elif rte['label'][obs_no] == "entailment":
                labels.append(0)

    return {'premise': prems, 'hypothesis': hyps, 'label': labels}


def _get_rte(params, expl, tokenizer, rte_file):

    rte = pd.read_csv(rte_file, sep = '\t')
    rte = _format_rte(rte, expl)
    rte = Dataset.from_dict(rte)
    rte = _filter_labels(rte)
    rte, batch_sampler = _insert_facts("rte", params, expl, rte)

    rte = _tokenize_data(
            rte,
            tokenizer)

    dataloader = _create_dataloader(rte, batch_sampler)

    return dataloader


def _get_wnli(params, expl, tokenizer, wnli_file):

    wnli = pd.read_csv(wnli_file, sep = '\t')
    wnli = _format_wnli(wnli, expl)
    wnli = Dataset.from_dict(wnli)
    wnli = _filter_labels(wnli)
    wnli, batch_sampler = _insert_facts("wnli", params, expl, wnli)

    wnli = _tokenize_data(
            wnli,
            tokenizer)

    dataloader = _create_dataloader(wnli, batch_sampler)

    return dataloader


def _get_control(params, expl_control, tokenizer, ctrl_file):

    control_test = read_jsonl(filename=ctrl_file)
    control_test = _reduce_control(control_test, expl_control)
    control_test = Dataset.from_dict(control_test)
    control_test = _filter_labels(control_test)
    control_test, batch_sampler = _insert_facts("control", params, expl_control, control_test)

    control_test = _tokenize_data(
            control_test,
            tokenizer)

    dataloader = _create_dataloader(control_test, batch_sampler)

    return dataloader

def _insert_facts(name, params, expl_dict, loaded_data, excl_h_cond=False):

    prem_facts = [[fact for fact in expl_dict[str(premise)]] \
            for premise in loaded_data['premise']]
    
    if params.h_facts and not excl_h_cond:

        hp_dict = HP_DICT_ALL[name]

        with open(hp_dict, "rb") as f:
            hp_dict_expl = pickle.load(f)

        for idx, observation in enumerate(loaded_data):
        
            obs_premise = str(observation['premise'])
            obs_hypothesis = str(observation['hypothesis'])

            if obs_premise in hp_dict_expl.keys():
                if obs_hypothesis in hp_dict_expl[obs_premise].keys():
                    extra_facts = hp_dict_expl[obs_premise][obs_hypothesis]
                    prem_facts[idx] = prem_facts[idx] + extra_facts

    ## Arranging the facts into batches with the corresponding hypotheses
    fact_lengths = [len(prem_facts[obs_no]) \
            for obs_no in range(len(loaded_data["hypothesis"]))]

    label = [[loaded_data["label"][obs_no]] * fact_lengths[obs_no] \
            for obs_no in range(len(loaded_data["label"]))]

    hyp_repeated = [[loaded_data["hypothesis"][obs_no]] * fact_lengths[obs_no] \
            for obs_no in range(len(loaded_data["hypothesis"]))]
      
    # Flattens the lists
    hyp_repeated = sum(hyp_repeated, [])
    prem_facts = sum(prem_facts, [])
    label = sum(label, [])
    expanded_data = Dataset.from_dict(
            {
                'premise': prem_facts, 
                'hypothesis': hyp_repeated, 
                'label': label
                }
            )

    # Creating sampler
    obs_no = 0
    sampler = []
    for fact_len in fact_lengths:
        obs_idx = []
        for i in range(fact_len):
            obs_idx.append(obs_no)
            obs_no += 1
        sampler.append(obs_idx)

    return expanded_data, sampler


def _tokenize_data(
        loaded_data: Dataset, 
        tokenizer: BertTokenizerFast,
        _ascii=False) -> Dataset:
    """
    We tokenize the data
    """

    if _ascii:
        loaded_data = loaded_data.map(lambda x: tokenizer(
                x['premise'].encode('ascii',errors='ignore').decode().strip(" "),
                x['hypothesis'].encode('ascii',errors='ignore').decode().strip(" "),
                truncation=False,
                padding=False))

    else:

        loaded_data = loaded_data.map(lambda x: tokenizer(
                x['premise'],
                x['hypothesis'],
                truncation=False,
                padding=False))

    return loaded_data

def find_all_nones(dataset):
    
    all_indices = []
    for i, obs in enumerate(dataset):
        if obs['premise'] is None:
            all_indices.append(i)

    return all_indices
    
def replace_all_none(dataset):
    
    filtered_indices = find_all_nones(dataset)
    dataset = dataset.filter(
            lambda example, 
            idx: idx not in filtered_indices, with_indices=True)

    return dataset, filtered_indices

def _load_hf_dataset(dataset_dict: dict) -> Dataset:

    # Load dataset
    print("Description:", dataset_dict['description'])
    print("Split name:", dataset_dict['split_name'])
    loaded_data = load_dataset(
            dataset_dict['description'], 
            split=dataset_dict['split_name'])
 
    return loaded_data

def _filter_labels(dataset: Dataset) -> Dataset:
    # Remove examples with no gold label
    
    dataset = dataset.filter(
            lambda example: example['label'] in [0, 1, 2])

    return dataset

def _create_dataloader(hf_dataset: Dataset, batch_sampler_list: list) \
        -> torch.utils.data.dataloader.DataLoader:
    
    hf_dataset.set_format(
            type='torch', 
            columns=[
            'input_ids',
            'token_type_ids',
            'attention_mask',
            'label'])

    dataloader = torch.utils.data.DataLoader(
            hf_dataset, 
            batch_sampler=batch_sampler_list,
            collate_fn=_padding)

    return dataloader

def _padding(batch):

    max_len = 0
    total_obs = 0
    for obs in batch:
        total_obs += 1
        if obs['input_ids'].shape[0] > max_len:
            max_len = obs['input_ids'].shape[0]

    all_input_ids = torch.zeros(total_obs, max_len).long()
    all_attention_mask = torch.zeros(total_obs, max_len).long()
    all_token_type_ids = torch.zeros(total_obs, max_len).long()
    all_labels = torch.zeros(total_obs).long()

    for obs_no, obs in enumerate(batch):
        len_example = obs['input_ids'].shape[0]
        padding = max_len - len_example
        if padding > 0:
            new_input_ids = torch.cat(
                    [obs['input_ids'], torch.zeros(padding).long()])
            new_att_mask = torch.cat(
                    [obs['attention_mask'], torch.zeros(padding).long()])
            new_token_type_ids = torch.cat(
                    [obs['token_type_ids'], torch.zeros(padding).long()])
        else:
            new_input_ids = obs['input_ids']
            new_att_mask = obs['attention_mask']
            new_token_type_ids = obs['token_type_ids']
        
        all_input_ids[obs_no,:] = new_input_ids
        all_attention_mask[obs_no,:] = new_att_mask
        all_token_type_ids[obs_no,:] = new_token_type_ids
        all_labels[obs_no] = obs['label']

        all_obs = {
                'input_ids': all_input_ids, 
                'attention_mask': all_attention_mask, 
                'token_type_ids': all_token_type_ids, 
                'label':  all_labels}

    return all_obs

def _prepare_huggingface_data_train(
        expl_dict :dict,
        params: argparse.Namespace,
        tokenizer: BertTokenizerFast,
        dataset_dict: dict):

    all_train = []

    train_splits = "_".join(params.train_splits)

    if params.load_train_data:

        assert params.fact_version in FILE_LOOKUP.keys()

        train_data = Dataset.from_csv("saved_train_data/train_data" + \
                train_splits + params.fact_version + ".csv")

        with open("saved_train_data/train_data_sampler" \
                + train_splits + params.fact_version + ".pkl", "rb") as fp:
            batch_sampler = pickle.load(fp)

    elif not params.load_train_data:

        for split_name in dataset_dict['splits']:
            all_train.append(_load_hf_dataset(
                {'description': dataset_dict['description'], 'split_name': split_name}))
        
        train_data = concatenate_datasets(all_train)
        train_data = _filter_labels(train_data)
        train_data = train_data.shuffle(seed=params.random_seed)
        train_data, batch_sampler = _insert_facts("anli", params, expl_dict, train_data, True)

        assert params.fact_version in FILE_LOOKUP.keys()

        train_data, indices_removed = replace_all_none(train_data)
            
        if len(indices_removed) > 0:
            batch_sampler = remove_obs(batch_sampler, indices_removed)

        assert batch_sampler[-1][-1] == len(train_data)-1,\
                "batch sampler matches length of training data"

        if params.save_train_data:

            print("Saving training data from CSV")
            train_data.to_csv("saved_train_data/train_data" \
                    + train_splits \
                    + params.fact_version + ".csv")

            with open("saved_train_data/train_data_sampler" \
                    + train_splits \
                    + params.fact_version \
                    + ".pkl", "wb") as fp:
                pickle.dump(batch_sampler, fp)

    print("Len train:", len(train_data))

    assert batch_sampler[-1][-1] == len(train_data)-1,\
            "batch sampler matches length of training data: "  \
            + str(len(train_data)-1) \
            + " - " + str(batch_sampler[-1][-1])

    train_data = _tokenize_data(
            train_data,
            tokenizer)

    train_data = _create_dataloader(train_data, batch_sampler)

    return train_data

def remove_obs(batch_sampler, indices_removed):
    
    new_batch_sampler = []
    prev_number = -1

    for obs_no in range(len(batch_sampler)):
        new_batch_sampler.append([])
        for fact_no in batch_sampler[obs_no]:
            if fact_no not in indices_removed:
                new_batch_sampler[obs_no].append(prev_number+1)
                prev_number += 1

    return new_batch_sampler

def _prepare_huggingface_data(
        expl_dict :dict,
        params: argparse.Namespace,
        tokenizer: BertTokenizerFast,
        dataset_dict: dict) -> (torch.utils.data.dataloader.DataLoader, dict):
    """
    From dictionary for dataset, returns dataloader for huggingface dataset
    """
    huggingface_dataset = _load_hf_dataset(dataset_dict)
    
    huggingface_dataset = _filter_labels(huggingface_dataset)

    huggingface_dataset, batch_sampler = _insert_facts(
            "anli",
            params,
            expl_dict, 
            huggingface_dataset)

    huggingface_dataset = _tokenize_data(
            huggingface_dataset, 
            tokenizer)

    dataloader = _create_dataloader(huggingface_dataset, batch_sampler)

    return dataloader

def _load_data(name):
    data = []
    a_file = open(name, "r")
    data = json.load(a_file)
    return data


