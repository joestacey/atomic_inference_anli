
import os

# Imports
from torch import nn
import numpy as np
import transformers
import torch
import random
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from transformers import AdamW
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
import sys
import argparse
from typing import Tuple
from torch.utils.data.dataloader import DataLoader
import itertools
import pickle

from utils_log import Log
from data_baseline import baseline_create_dictionary_of_eval_datasets, baseline_prepare_all_dataloaders
from models import LogicModel

print("\n \n PID is",os.getpid(), "\n \n")

def get_args():

  parser = argparse.ArgumentParser(description="Training model parameters")

  # Arguments for modelling different scenarios
  parser.add_argument("--model_type", type=str, default="microsoft/deberta-v3-base", 
          help="Model to be used")
  parser.add_argument("--random_seed", type=int, default=42, 
          help="Choose the random seed")

  # Arguments for model training
  parser.add_argument("--epochs", type=int, default=2, 
          help="Number of epochs for training")
  parser.add_argument("--batch_size", type=int, default=4, 
          help="batch_size")
  parser.add_argument("--learning_rate", type=float, default=4e-5, 
          help="Choose learning rate")

  # Evaluate-only options
  parser.add_argument("--eval_only", type=int, default=0,
          help="Only evaluate without training")
  parser.add_argument("--model_file", type=str, default='saved_model.pt',
          help="File to load model from if eval_only mode")

  # Learning schedule arguments
  parser.add_argument("--linear_schedule", type=int, default=1, 
          help="To use linear schedule with warm up or not")
  parser.add_argument("--warmup_epochs", type=int, default=1, 
          help="Warm up period")
  parser.add_argument("--warmdown_epochs", type=int, default=1, 
          help="Warm down period")
  parser.add_argument("--weight_decay", type=float, default=0.01,
          help="Weight decay for AdamW")
  parser.add_argument("--schedule", type=str, default='up_down',
          help="'up_down' or 'up'")
  parser.add_argument("--warmup", type=float, default=0.1,
          help="Warmup as a proportion of total training")

  # Span arguments
  parser.add_argument("--name_id", type=str, default="default", 
          help="Name used for saved model and log file")

  # Train data arguments
  parser.add_argument("--train_data", type=str, default="anli",
          help="Code setup to run for 'anli'")

  parser.add_argument("--train_splits", type=str, default='all',
          help="model set to run for 'all'")

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

@torch.no_grad()
def evaluate(dataset_name: str, 
        dataloader_eval: DataLoader) -> int:
    """
    Evaluates NLI logic model

    Args:
        dataset_name: description of the evaluation dataset
        dataloader_eval: evaluation dataloader

    Returns:
        correct: number of correct predictions
    """

    baseline_model.eval()
    correct, total =  0, 0

    dataloader = dataloader_eval
    
    model_log.msg(["Output for dataset: " + dataset_name])

    for i, batch in enumerate(dataloader):

        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = baseline_model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['token_type_ids'])

        logit_outputs = outputs['logits']

        predictions = torch.max(logit_outputs,1)[1]

        correct += int(batch['label'].eq(predictions).sum())
        total += batch['label'].shape[0]
        
    model_log.msg(["Total accuracy:" + \
            str(round(correct/total, 4))])

    model_log.msg(["Total correct & total:" + \
            str(round(correct, 4)) + " & " + str(round(total, 4))])

    return correct

def train() -> None:
    """
    Model training
    """

    # Our train dataloader
    dataloader_train = train_dataloader['anli_train']

    print(len(dataloader_train))

    best_dev = 0
    best_epoch = 0

    # Training loop
    for epoch in range(params.epochs):

        # Set model in training mode
        baseline_model.train()

        for i, batch in enumerate(dataloader_train):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = baseline_model(
                    batch['input_ids'], 
                    batch['attention_mask'],
                    batch['token_type_ids'])

            logit_outputs = outputs['logits']

            loss = nli_loss = F.cross_entropy(logit_outputs, batch['label'])

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if params.linear_schedule:
                schedule.step()

        dev_acc = evaluate_each_epoch(epoch)

def evaluate_each_epoch(epoch: int, final_eval=False) -> int:
    """
    Evaluates model on each dataset after each epoch

    Args:
        epoch: epoch number (starting at 0)
        final_eval: if performing the final model evaluation after training

    Returns:
        dev_scores: dev set performance
    """

    if not final_eval:
        model_log.msg(["Epoch:" + str(epoch+1)])
        dev_scores = 0

        # We evaluate on other huggingface evaluation datasets
        for dataset_name, dataset in eval_dataloaders.items():
            if 'anli_dev' in dataset_name:
                model_log.msg(["Dataset: " + dataset_name])
                dev_score = evaluate(dataset_name, dataset)
                dev_scores = dev_scores + dev_score

    else:
        model_log.msg(["Final evaluation"])

        for dataset_name, dataset in eval_dataloaders.items():
            model_log.msg(["Dataset: " + dataset_name])
            _ = evaluate(dataset_name, dataset)
            dev_scores = 0

    return dev_scores

def create_lr_schedules():

    if params.linear_schedule:
        if params.schedule == 'up_down':
            schedule = LambdaLR(optimizer, lr_lambda)
        elif params.schedule == 'up':
            schedule = LambdaLR(optimizer, lr_lambda_warmup)

        return schedule

    return None


def lr_lambda_warmup(current_step: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0

def lr_lambda(current_step: int) -> float:

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    return max(
            0.0, float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)))


if __name__ == '__main__':

    params = get_args()

    params.linear_schedule = bool(params.linear_schedule)
    params.eval_only = bool(params.eval_only)

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
    eval_data_list = baseline_create_dictionary_of_eval_datasets(
                params)

    tokenizer = AutoTokenizer.from_pretrained(
            params.model_type,
            truncation=False)

    # We create dataloaders for eSNLI and HuggingFace datasets
    train_dataloader, eval_dataloaders = baseline_prepare_all_dataloaders(
                    eval_data_list,
                    params,
                    tokenizer)

    if params.model_type == 'microsoft/deberta-large' or \
            params.model_type == 'microsoft/deberta-xlarge':
        dim = 1024
    else:
        dim = 768

    baseline_model = AutoModelForSequenceClassification.from_pretrained(
            params.model_type,
            output_attentions=True,
            output_hidden_states=True,
            num_labels=3)

    baseline_model.to(device)

    # We create our optimizer
    optimizer = AdamW(
            list(baseline_model.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay)

    # We create our learning schedules
    assert params.train_data == 'anli'
    dataloader_name = 'anli_train'

    if params.schedule == 'up_down':

        num_warmup_steps = len(
                train_dataloader[dataloader_name])*params.warmup_epochs
        warm_down_steps = len(
                train_dataloader[dataloader_name])*params.warmdown_epochs
        num_training_steps = num_warmup_steps + warm_down_steps

    elif params.schedule == 'up':

        num_warmup_steps = len(
                train_dataloader[dataloader_name])*params.epochs*params.warmup

    schedule = create_lr_schedules()

    if params.eval_only:

        baseline_model.load_state_dict(torch.load(
        os.getcwd() + "/savedmodel/" + params.model_file + '.pt'))
        _ = evaluate_each_epoch(0, True)

    else:

        train()
        print("All done")
        _ = evaluate_each_epoch(0, True)

        torch.save(baseline_model.state_dict(),
                            os.getcwd() + "/savedmodel/saved_baseline_" \
                                    + name_id + '.pt')

        print("All models saved")
