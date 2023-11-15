# -*- coding: utf-8 -*-
"""Coursera's_reviews_sentiments_analysis_.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10WOVaIUDX0f5_1y79cW77iGGDl4W6DSv

# Student's reviews sentiment analysis

## 1. Install transformers and datasets
"""

# !pip install transformers[torch] datasets
# !pip install torchinfo
# !pip install accelerate -U
# !pip install peft

"""## 2. Import neccessay libraries"""

import requests
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import  DistilBertTokenizer, \
                         Trainer, TrainingArguments, pipeline, DistilBertForSequenceClassification
from torchinfo import summary
from peft import PeftModel, PeftConfig, get_peft_model,LoraConfig

"""## Download files from Google Drive and save it in the dataset directory"""

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

file_id = "1HsYBeLV0EJ5rNX6RIrV6GaBiVJcm8Q6Z"
destination = './reviews.csv'
download_file_from_google_drive(file_id, destination)

"""## Read the dataset and convert it to a format suitable for huggingface"""

def transform_csv_for_huggingface(dataset_path):
  review_df = pd.read_csv(dataset_path)
  # it is necessary to convert the Label to label
  review_df = review_df.rename(columns={'Label':'label'})
  review_df['label'] = review_df['label'] - 1
  review_df = review_df.drop('Id',axis=1)
  review_df.to_csv('reviews_hf.csv',index=None)

dataset_path = 'reviews.csv'
transform_csv_for_huggingface(dataset_path)

review_data_raw = load_dataset('csv',data_files='reviews_hf.csv')

review_data_raw

"""## Split dataset into train and test"""

review_data_split = review_data_raw['train'].train_test_split(test_size=0.2)

"""## Checkpoint"""

checkpoint = 'distilbert-base-cased'

"""## Tokenize the dataset"""

tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)

def tokenize_fn(batch):
  return tokenizer(batch['Review'],truncation=True, padding='max_length',max_length=512)

review_data_tokenized = review_data_split.map(tokenize_fn,batched=True)

"""## Define model"""



model = DistilBertForSequenceClassification.from_pretrained(checkpoint,num_labels=5)

peft_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=['q_lin'])
#peft_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# for name, module in model.named_modules():
#     print(name)

summary(model)

"""## Define training arguments"""

training_args = TrainingArguments(
    output_dir = 'training_dir',
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    num_train_epochs = 5,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 64,
)

"""## Define evaluation metric"""

def compute_metrics(logits_and_labels):
  logits, labels = logits_and_labels
  predictions = np.argmax(logits,axis=-1)
  acc = np.mean(predictions == labels)
  f1 = f1_score(labels, predictions, average='macro')
  return {'accuracy':acc, 'f1':f1}

# Test metrics
# logits = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])
# labels = np.array([2, 1, 1])

# # Call the function with the example data
# metrics = compute_metrics((logits, labels))

# # Print the results
# print(metrics)



"""## Define trainer"""

trainer = Trainer(model,
                args = training_args,
                train_dataset =review_data_tokenized['train'],
                eval_dataset= review_data_tokenized['test'],
                tokenizer = tokenizer,
                compute_metrics = compute_metrics)

trainer.train()

# import os
# import shutil
# num_epochs = 10
# for epoch in range(num_epochs):
#     # Train for one epoch
#     trainer.train()

#     # Delete the contents of the tmp_trainer folder
#     folder = './tmp_trainer'
#     for filename in os.listdir(folder):
#         file_path = os.path.join(folder, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print('Failed to delete %s. Reason: %s' % (file_path, e))



"""## Load the trained model and inference on test data"""

saved_model = pipeline("text-classification",model='training_dir/checkpointxxxx',device=0)
