
from torch import nn
import transformers
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
import random

class AttentionClassifier(nn.Module):
    """
    Entailment/Contradiction detection attention layer
    """
    def __init__(self, dimensionality, dropout, dropout_type):
        super(AttentionClassifier,self).__init__()

        self.linear1 = torch.nn.Linear(dimensionality, dimensionality)
        self.linear2 = torch.nn.Linear(dimensionality, 1)
        self.tanh = torch.tanh
        self.linear3 = torch.nn.Linear(1, 1)
        self.sig = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.eval = True
        self.dropout = dropout
        self.dropout_type = dropout_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def eval_(self):
        self.eval = True

    def train(self):
        self.eval = False

    def forward(
              self, 
              rep: torch.tensor, 
              logit: torch.tensor) -> dict:
        """
        Forward pass for ent/contradiction detection attention layers

        Args:
            rep: CLS representations of all atoms (atom no x CLS dimensions)
            logit: logit for all atoms (atom no x 1)

        Returns:
            output_val: dictionary, containing:
                'instance_output': probability of ent/cont. for instance
                'att_weights': normalized attention weights
                'att_unnorm': unnormalzed attention weights
                'dropout': dropout mask applied 
        """

        # batch_size x seq_len x dim:
        val = self.linear1(rep)
        val = self.tanh(val)

        # batch_size x seq_len x 1:
        val = self.linear2(val)
        val = self.sig(val)

        sum_val = torch.sum(val)
        att_unnorm = val
        inv_sum_val = 1/sum_val

        # batch_size x seq_len x 1:
        att_weights = val*inv_sum_val

        dropout_mask = torch.ones(att_weights.shape).to(self.device)
        
        if not self.eval:
            
            if self.dropout_type == 'standard':
                dropout_mask = self.standard_dropout(dropout_mask) 

            att_weights = att_weights * dropout_mask
            att_unnorm = att_unnorm * dropout_mask

            new_sum = torch.sum(att_weights)
            att_weights = att_weights / new_sum

        #batch_size x dimensions
        updated_rep = torch.einsum('jk, jm -> k', [logit, att_weights])
        output_val = self.linear3(updated_rep)
        output_val = self.sig(output_val)

        # Preparing dictionary of outputs
        output_dict = {}
        output_dict['instance_output'] = output_val
        output_dict['att_weights'] = att_weights
        output_dict['att_unnorm'] = att_unnorm
        output_dict['dropout'] = dropout_mask

        return output_dict

    def standard_dropout(
            self, 
            dropout_mask: torch.tensor):
        """
        Random applies dropout to atoms

        Args:
            dropout_mask: mask before dropout (all 1s)

        Returns:
            dropout_mask: updated mask when applying dropout
        """

        number_atoms = dropout_mask.shape[0]
        atoms_dropped_out = []

        # We allow dropout on the atoms in the attention layer
        for atom_no in range(number_atoms):
            if random.choices([True, False],
                    [self.dropout, 1-self.dropout])[0] == True:
                atoms_dropped_out.append(atom_no)

        # We can't apply drop-out to every atom
        if number_atoms == len(atoms_dropped_out):
            atom_remove_dropout = random.randint(0,len(atoms_dropped_out)-1)
            atoms_dropped_out.remove(atom_remove_dropout)

        for atom_no in atoms_dropped_out:
            dropout_mask[atom_no] = 0

        return dropout_mask


class LogicModel(nn.Module):
    """
    Our Logic NLI model
    """
    def __init__(self, dimensionality, model_type, atom_dropout, dropout_type):
        super(LogicModel,self).__init__()

        self.attention_cont = AttentionClassifier(
                dimensionality, 
                atom_dropout,
                dropout_type)
        self.attention_ent = AttentionClassifier(
                dimensionality, 
                atom_dropout,
                dropout_type)

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_type,
            output_attentions=True,
            output_hidden_states=True,
            num_labels=2)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_type = model_type

    def forward(
            self, 
            batch: dict) -> dict:
        """
        We create batch input from different atom masks, and pass through model
        
        Args:
            batch: batch input into model (input ids, mask and token_type_ids)
        
        Returns:
            outputs: instance outputs for each class, with unnormalized 
                attention, which atoms to supervise, and the labels
        """

        # We pass each atom through BERT (with atoms created from masking)
        all_atoms_cls, all_atoms_logits = None, None

        atom_outputs_dict = self.encoder(
                batch['input_ids'],
                batch['attention_mask'],
                batch['token_type_ids'],
                return_dict=True)

        all_atoms_cls = atom_outputs_dict['hidden_states'][-1][:,0,:]
        all_atoms_logits = atom_outputs_dict['logits']
        
        # We set the labels for our entailment and condiction detection layers
        if batch['label'][0] == 0:
            ent_label=1
            cont_label=0
        elif batch['label'][0] == 1:
            ent_label=0
            cont_label=0
        else:
            ent_label=0
            cont_label=1

        # We find the outputs from the entailment/contradiction detection layers
        outputs = self.ent_and_cont_detection(
                    batch['label'][0],
                    all_atoms_logits,
                    all_atoms_cls,
                    ent_label=ent_label,
                    cont_label=cont_label)

        return outputs


    def ent_and_cont_detection(
        self,
        true_label,
        all_atoms_logits: torch.tensor,
        all_atoms_cls: torch.tensor,
        ent_label: int,
        cont_label: int,
        ) -> dict:
        """
        Apply the entailment and contradiction detection attention layers

        Args:
            all_atoms_logits: logits for each atom (for both classes)
            all_atoms_cls: cls representation for each atom
            ent_label: entailment detection label
            cont_label: cont detection label

        Returns:
            output_dict: dict of outputs for NLI instance pairs
        """

        ent_output = self.attention_ent(
                all_atoms_cls, 
                all_atoms_logits[:,0].unsqueeze(1))

        ent_output['label'] = torch.tensor(
                [ent_label]).to(self.device)

        cont_output = self.attention_cont(
                all_atoms_cls, 
                all_atoms_logits[:,1].unsqueeze(1))

        cont_output['label'] = torch.tensor(
                [cont_label]).to(self.device)


        output_dict = {
                'true_label': true_label,
                'ent': ent_output,
                'cont': cont_output}

        return output_dict
