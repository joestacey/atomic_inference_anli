
import os
from torch import nn
import numpy as np
from collections import OrderedDict
import transformers
import torch
import random
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from transformers import AdamW
from datasets import load_dataset
import sys
import argparse
from typing import Tuple
from torch.utils.data.dataloader import DataLoader

import csv
import pickle
import pandas as pd

from utils_log import Log

from data_prep import create_dictionary_of_eval_datasets, prepare_all_dataloaders
from models import LogicModel

print("\n \n PID is",os.getpid(), "\n \n")

def get_args():

  parser = argparse.ArgumentParser(description="Training model parameters")

  # Arguments for model training
  parser.add_argument("--model_type", type=str, default="microsoft/deberta-v3-base",
          help="Model to be used")
  parser.add_argument("--random_seed", type=int, default=42,
          help="Choose the random seed")
  parser.add_argument("--epochs", type=int, default=2, 
          help="Number of epochs for training")
  parser.add_argument("--learning_rate", type=float, default=7e-6, 
          help="Choose learning rate")
  parser.add_argument("--linear_schedule", type=int, default=1,
          help="To use linear schedule with warm up or not")
  parser.add_argument("--warmup_epochs", type=int, default=1,
          help="Warm up period")
  parser.add_argument("--warmdown_epochs", type=int, default=1,
          help="Warm down period")
  parser.add_argument("--weight_decay", type=float, default=0.01,
          help="Weight decay for AdamW")
  parser.add_argument("--grad_accum", type=int, default=0,
          help="1 if we want gradient accumulation for the encoder, 0 otherwise")
  parser.add_argument("--grad_accum_batch_size", type=int, default=16,
          help="Batch size for gradient accumulation")

  # Evaluate-only options
  parser.add_argument("--eval_only", type=int, default=0,
          help="Only evaluate without training")

  # Log name
  parser.add_argument("--name_id", type=str, default="default", 
          help="Name used for saved model and log file")

  # Dropout arguments (not used in paper)
  parser.add_argument("--dropout_type", type=str, default='overlap',
          help="'standard' or 'overlap'. Only 'overlap' dropout was applied")
  parser.add_argument("--atom_dropout", type=float, default=0.0,
          help="Dropout acting on spans in additional attention layer")

  # Train data arguments
  parser.add_argument("--train_data", type=str, default="anli",
          help="Code set to run with 'anli'")
  parser.add_argument("--train_splits", type=str, default='all',
          help="Code expects 'all'")
  parser.add_argument("--save_train_data", type=int, default=0,
          help="If we save a CSV of the training data")
  parser.add_argument("--load_train_data", type=int, default=1,
          help="If we load a CSV of the training data")
  parser.add_argument("--fact_version", type=str, default='default',
          help="Currently code runs or 'default'")

  # Model settings
  parser.add_argument("--load_model", type=int, default=0,
          help="Load a baseline model")
  parser.add_argument("--load_logic_model", type=int, default=0,
          help="Load a baseline model")
  parser.add_argument("--load_id", type=str, default="",
          help="Location of baseline model to load")
  parser.add_argument("--h_facts", type=int, default=0,
          help="If we also use hypothesis conditioned facts")
  parser.add_argument("--instance_loss_mult", type=float, default=0.9,
          help="Multiplier for instance level loss")
  parser.add_argument("--save_at_end", type=int, default=1,
          help="Do we save the model at the end or not")

  params, _ = parser.parse_known_args()

  return params

def set_seed(seed_value: int) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed_value: chosen random seed
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def find_logic_predictions(
        att_unnorm_cont: torch.tensor, 
        att_unnorm_ent: torch.tensor, 
        dataset_name: str) -> int:

    """
    Make predictions based atom-level predictions

    Args:
        att_unnorm_cont: output from contradiction detection attention layer
        att_unnorm_ent: output from entailment detection attention layer
        dataset_name: dataset name

    Returns:
        pred: logic prediction for NLI instance pair
    """
    # First check if we have contradiction or not
    if torch.max(att_unnorm_cont) > 0.5:
        pred = 2
    elif torch.max(att_unnorm_ent) > 0.5:
        pred = 0
    else:
        pred = 1
    
    if pred == 2 and dataset_name in two_class_datasets:
        pred = 1

    return pred


@torch.no_grad()
def evaluate(
        epoch: int,
        dataset_name: str, 
        dataloader_eval: Tuple[DataLoader, dict, dict, dict]) -> None:
    """
    Evaluates NLI logic model

    Args:
        dataset_name: description of the evaluation dataset
        dataloader_eval: dataloaders for evaluation
    """

    logic_model.encoder.eval()
    logic_model.attention_ent.eval_()
    logic_model.attention_cont.eval_()

    correct_logic_att, total =  0, 0

    # Stats for evaluating span performance
    dataloader = dataloader_eval
    
    model_log.msg(["Output for dataset: " + dataset_name])

    for i, batch in enumerate(dataloader):

        if i == 0:
            print("Dataset name:", dataset_name)

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = logic_model(batch)

        # Find model predictions
        pred_logic_att = find_logic_predictions(
                outputs['cont']['att_unnorm'], 
                outputs['ent']['att_unnorm'],
                dataset_name)

        total += 1
        assert batch['label'][0] == batch['label'][-1]

        if pred_logic_att == batch['label'][0].item():
            correct_logic_att = correct_logic_att + 1

    model_log.msg(["Total accuracy (logic att):" + \
            str(round(correct_logic_att/total, 4))])

    model_log.msg(["Total correct & total:" + \
            str(round(correct_logic_att, 4)) + " & " + str(round(total, 4))])

    return correct_logic_att
    
def get_att_loss(
        outputs: dict, 
        class_str: str, 
        desired_label: int,
        instance_loss: torch.tensor, 
        atom_loss_term: torch.tensor) \
                -> (torch.tensor, torch.tensor, torch.tensor):
    """
    Finds the instance loss, and additional loss term
    
    Args:
        outputs: dictionary of model outputs
        class_str: 'ent' or 'cont' for different attention layers
        instance_loss: instance loss for observation so far
        atom_loss_term: additional loss term for observation so far

    Returns:
        instance_loss: updated instance loss for observation
        atom_loss_term: updated atom loss for observation
    """

    # Instance loss
    instance_loss += (outputs[class_str]['instance_output'] \
                            - desired_label)**2

    # Additional loss term
    atom_loss_term += torch.square(
            torch.max(outputs[class_str]['att_unnorm']) \
                    - desired_label)

    return instance_loss, atom_loss_term

def get_saved_predictions(batch):

    if params.model_type[0:4] == 'bert':
        sep_toks = torch.where(batch['input_ids'][0] == 102)[0]
    elif params.model_type[0:9] == 'microsoft':
        sep_toks = torch.where(batch['input_ids'][0] == 2)[0]

    hyp = batch['input_ids'][0][sep_toks[0]+1:sep_toks[1]]
    
    hyp_prediction = preds_dict[str(
                list(np.array(hyp.cpu())))]
    
    return hyp_prediction

def train() -> None:

    # Our train dataloader
    dataloader_train = train_dataloader['anli_train']

    best_dev = 0
    best_epoch = 0
    # Training loop

    for epoch in range(params.epochs):


        # Set model in training mode
        logic_model.encoder.train()
        logic_model.attention_ent.train()
        logic_model.attention_cont.train()

        len_b = len(dataloader_train)

        for i, batch in enumerate(dataloader_train):

            if i % 1000 == 0:
                model_log.msg(["Training with minibatch: " + str(i)])
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = logic_model(batch)

            e_sup_value = outputs['ent']['label'].item()
            c_sup_value = outputs['cont']['label'].item()
        
            # We calculate losses
            instance_loss = torch.tensor([0.0]).to(device)
            atom_loss_term = torch.tensor([0.0]).to(device)

            # Update the loss from the entailment and cont attention layers
            if outputs['true_label'] == 1 or outputs['true_label'] == 0:
                instance_loss, atom_loss_term = get_att_loss(
                            outputs,
                            'ent',
                            e_sup_value,
                            instance_loss,
                            atom_loss_term)

            instance_loss, atom_loss_term = get_att_loss(
                        outputs,
                        'cont',
                        c_sup_value,
                        instance_loss,
                        atom_loss_term)

            loss = params.instance_loss_mult * instance_loss + atom_loss_term

            loss.backward(retain_graph=True)
            
            if outputs['true_label'] == 1 or outputs['true_label'] == 0:
                optimizer_ent.step()

            optimizer_contradiction.step()

            if i % params.grad_accum_batch_size == 0 or not params.grad_accum:
                optimizer_encoder.step()

            if params.linear_schedule:
                if i % params.grad_accum_batch_size == 0 or not params.grad_accum:
                    schedule_encoder.step()

                schedule_ent.step()
                schedule_cont.step()

            if i % params.grad_accum_batch_size == 0 or not params.grad_accum:
                optimizer_encoder.zero_grad()
            optimizer_ent.zero_grad()
            optimizer_contradiction.zero_grad()

        evaluate_each_epoch(epoch)
    
def evaluate_each_epoch(epoch: int, final_eval=False) -> None:
    """
    Evaluates model on each dataset after each epoch

    Args:
        epoch: epoch number (starting at 0)
    """
    
    if not final_eval:
        model_log.msg(["Epoch:" + str(epoch+1)])

        # We evaluate on other huggingface evaluation datasets
        for dataset_name, dataset in eval_dataloaders.items():
            model_log.msg(["Dataset: " + dataset_name])
            _ = evaluate(
                    epoch, dataset_name, dataset)

    else:
        model_log.msg(["Final evaluation"])

        for dataset_name, dataset in eval_dataloaders.items():
            model_log.msg(["Dataset: " + dataset_name])
            _ = evaluate(epoch, dataset_name, dataset)

    return

def create_lr_schedules():

    if params.linear_schedule:
        schedule_encoder = LambdaLR(optimizer_encoder, lr_lambda_enc)
        schedule_ent = LambdaLR(optimizer_ent, lr_lambda)
        schedule_cont = LambdaLR(optimizer_contradiction, lr_lambda)

        return schedule_encoder, schedule_ent, schedule_cont

    return None, None, None

def lr_lambda_enc(current_step: int) -> float:

    if params.grad_accum:
        num_warmup_steps_opt = num_warmup_steps / params.grad_accum_batch_size
        num_training_steps_opt = num_training_steps / params.grad_accum_batch_size

    else:

        num_warmup_steps_opt = num_warmup_steps
        num_training_steps_opt = num_training_steps

    if current_step < num_warmup_steps_opt:
        return float(current_step) / float(max(1, num_warmup_steps_opt))

    return max(
            0.0, float(num_training_steps_opt - current_step) / float(
                max(1, num_training_steps_opt - num_warmup_steps_opt)))


def lr_lambda(current_step: int) -> float:

    num_warmup_steps_opt = num_warmup_steps
    num_training_steps_opt = num_training_steps

    if current_step < num_warmup_steps_opt:
        return float(current_step) / float(max(1, num_warmup_steps_opt))

    return max(
            0.0, float(num_training_steps_opt - current_step) / float(
                max(1, num_training_steps_opt - num_warmup_steps_opt)))


def get_loaded_state():
    
    loaded_state_dict = torch.load(
            os.getcwd() + "/savedmodel/" + params.load_id + '.pt')

    if params.model_type[0:9] == 'microsoft':
        
        keys_for_encoder_params = {}
        keys_to_remove = []

        for key, value in loaded_state_dict.items():
            if key[:8] == 'deberta.':
                keys_for_encoder_params[key] = key[8:]
            elif  key[:11] == 'classifier.':
                keys_to_remove.append(key)
            elif  key[:7] == 'pooler.':
                  keys_to_remove.append(key)

        for _ in range(len(loaded_state_dict)):
            k, v = loaded_state_dict.popitem(False)
            loaded_state_dict[keys_for_encoder_params[k] if k in keys_for_encoder_params.keys() else k] = v

        for key in keys_to_remove:
            del loaded_state_dict[key]

    elif params.model_type[0:4] == 'bert':
        keys_for_encoder_params = {}
        keys_to_remove = []

        for key, value in loaded_state_dict.items():
            if key[:5] == 'bert.':
                keys_for_encoder_params[key] = key[5:]
            elif key[:11] == 'classifier.':
                keys_to_remove.append(key)

        for _ in range(len(loaded_state_dict)):
            k, v = loaded_state_dict.popitem(False)
            loaded_state_dict[keys_for_encoder_params[k] if k in keys_for_encoder_params.keys() else k] = v

        for key in keys_to_remove:
            del loaded_state_dict[key]
    
    return loaded_state_dict


if __name__ == '__main__':

    params = get_args()

    params.linear_schedule = bool(params.linear_schedule)
    params.eval_only = bool(params.eval_only)
    params.save_train_data = bool(params.save_train_data)
    params.load_train_data = bool(params.load_train_data)
    params.grad_accum = bool(params.grad_accum)
    params.load_model = bool(params.load_model)
    params.load_logic_model = bool(params.load_logic_model)
    params.h_facts = bool(params.h_facts)
    params.baseline = False
    params.save_at_end = bool(params.save_at_end)

    if params.eval_only:
        assert params.load_logic_model

    if params.train_splits == 'all':
        params.train_splits = ['train_r1', 'train_r2', 'train_r3']

    print(params)

    if params.name_id == 'default':
        name_id =  str(os.getpid())
    else:
        name_id = params.name_id
    
    # Logging file
    log_file_name = 'log_logic_model_' + name_id + '.txt'
    model_log = Log(log_file_name, params)

    # Set CUDAS to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(params.random_seed)

    # Create folder for saving models
    if not os.path.exists('savedmodel'):
        os.makedirs('savedmodel')

    # We create a dictionary of the Huggingface datasets to be OOD datasets
    eval_data_list = create_dictionary_of_eval_datasets(
                params)

    two_class_datasets = ['rte', 'wnli', 'wnli_dev', 'rte_dev']

    tokenizer = AutoTokenizer.from_pretrained(
            params.model_type,
            truncation=False)

    # We create dataloaders
    train_dataloader, eval_dataloaders = prepare_all_dataloaders(
                    eval_data_list,
                    params,
                    tokenizer)

    if params.model_type == 'microsoft/deberta-large' or \
            params.model_type == 'microsoft/deberta-xlarge' or \
            params.model_type == 'microsoft/deberta-v3-large':
        dim = 1024
    else:
        dim = 768

    logic_model = LogicModel(
            dim, 
            params.model_type, 
            params.atom_dropout, 
            params.dropout_type)

    logic_model.to(device)

    # Check if we load a previous model
    if params.load_model:

        loaded_state_dict = get_loaded_state()

        if params.model_type[0:4] == 'bert':

            logic_model.encoder.bert.load_state_dict(
                loaded_state_dict)

        elif params.model_type[0:9] == 'microsoft':

            logic_model.encoder.deberta.load_state_dict(
                    loaded_state_dict)


    if params.load_logic_model:

         loaded_state_dict = torch.load(
            os.getcwd() + "/savedmodel/" + params.load_id + '.pt')

         logic_model.load_state_dict(
                loaded_state_dict)

    # We create our optimizers
    optimizer_encoder = AdamW(
            list(logic_model.encoder.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay)

    optimizer_ent = AdamW(
            list(logic_model.attention_ent.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay)

    optimizer_contradiction = AdamW(
            list(logic_model.attention_cont.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay)

    # We create our learning schedules
    assert params.train_data == "anli"
    dataloader_name = 'anli_train'

    num_warmup_steps = len(
            train_dataloader[dataloader_name])*params.warmup_epochs
    warm_down_steps = len(
            train_dataloader[dataloader_name])*params.warmdown_epochs
    num_training_steps = num_warmup_steps + warm_down_steps

    schedule_encoder, schedule_ent, schedule_cont = create_lr_schedules()

    if params.eval_only:

        logic_model.load_state_dict(torch.load(
        os.getcwd() + "/savedmodel/" + params.load_id + '.pt'))
        evaluate_each_epoch(0, True)

    else:

        train()
        print("All done")
        evaluate_each_epoch(0, True)


        if params.save_at_end:
            torch.save(logic_model.state_dict(),
                                os.getcwd() + "/savedmodel/saved_model_" \
                                        + name_id + '.pt')
            print("All models saved")
