# This is the project code for our paper _**Atomic Inference for NLI with Generated Facts as Atoms**_

Contact email address: j.stacey20@imperial.ac.uk

## How to run the project code:

### Step 1) 
- We need to create and save the training data: python run.py --load_train_data 0 --save_train_data 1
- This process is very slow, but only needs to be run once

### Step 2) 
- To run our model (FGLR), use: python run.py --h_facts 1 --name_id name_of_exp
- Please note, in the paper experiments, the FGLR model encoder is initialised from the fine-tuned baseline. To do this, folow steps 3 and 4 below:

### Step 3) 
- Run the baseline: python baseline_run.py --name_id name_of_exp

### Step 4) 
- Then load the saved baseline model: python run.py --name_id w_loading_baseline --load_model 1 --load_id saved_baseline_filename --h_facts 1

