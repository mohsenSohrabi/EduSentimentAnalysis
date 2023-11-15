from  transformers import (
                            DistilBertForSequenceClassification,
                            DistilBertTokenizer,
                            Trainer,
                            TrainingArguments,
                            )
import pandas as pd
from peft import (
                  PeftModel, 
                  PeftConfig,
                  get_peft_model,
                  LoraConfig)

from utilities import (
                        download_file_from_google_drive,
                        transform_csv_for_huggingface
                      )

from datasets import load_dataset
from config import *
from llm_util import tokenize_fn, compute_metrics

'''
Download the Dataset
The dataset of interest is originally hosted on Kaggle and can be accessed via the following link:
https://www.kaggle.com/datasets/septa97/100k-courseras-course-reviews-dataset

For convenience, the dataset has also been uploaded to Google Colab. The code snippet below is designed to download the file directly from Google Colab into your local environment.
'''
download_file_from_google_drive(id=FILE_ID, destination=DATASET_SAVE_PATH)

# Convert the data in a format suitable for Huggingface
transform_csv_for_huggingface(dataset_path=DATASET_SAVE_PATH, 
                              destination= DATSET_SUITABLE_FOR_HF_NAME) 

# load dataset using datasets library from huggingface
review_data_raw = load_dataset('csv',data_files=DATSET_SUITABLE_FOR_HF_NAME)

# This line of code splits the 'train' subset of the 'review_data_raw' dataset into a training set and a test set. 
# The 'test_size' parameter is set to 0.2, meaning that 20% of the data will be used for the test set, 
review_data_split = review_data_raw['train'].train_test_split(test_size=0.2)

# Define tokenizer based on our checkpoint 'distilbert-base-cased' 
tokenizer = DistilBertTokenizer.from_pretrained(BASE_CHECKPOINT)

# tokenize the dataset
review_data_tokenized = review_data_split.map(lambda batch: 
                                              tokenize_fn(batch, tokenizer), batched=True)
# Load the pre-trained DistilBert model for sequence classification from Hugging Face's model hub.
# The 'num_labels' parameter is set to 5, indicating that the model should be configured to output 5 different classes.
model = DistilBertForSequenceClassification.from_pretrained(BASE_CHECKPOINT,num_labels=5)

# Create a configuration for the PEFT (Progressive Embedding Fine-Tuning) method.
# The 'task_type' is set to "SEQ_CLS" for sequence classification tasks.
# The 'r' parameter is set to 4, which is the rank of the low-rank approximation in PEFT.
# The 'lora_alpha' parameter is set to 32, which is the scaling factor for the low-rank approximation in PEFT.
# The 'lora_dropout' parameter is set to 0.01, which is the dropout rate for the low-rank approximation in PEFT.
# The 'target_modules' parameter is set to ['q_lin'], which means that the PEFT method will be applied to the 'q_lin' module of the model.
peft_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=['q_lin'])

# Apply the PEFT method to the model using the specified configuration.
# The 'get_peft_model' function returns a new model that has been modified according to the PEFT configuration.
model = get_peft_model(model, peft_config)

# Print the names and shapes of the trainable parameters of the model.
# This is useful for understanding the structure of the model and for debugging.
model.print_trainable_parameters()

# Define training arguments such as output directory, evaluation strategy, number of epochs, and batch sizes.
training_args = TrainingArguments(
    output_dir = 'training_dir',
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    num_train_epochs = 5,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 64,
)

# Initialize the Trainer with the model, training arguments, datasets, tokenizer, and the compute metrics function.
# compute_metrics is a function defined in llm_util
trainer = Trainer(model,
                args = training_args,
                train_dataset =review_data_tokenized['train'],
                eval_dataset= review_data_tokenized['test'],
                tokenizer = tokenizer,
                compute_metrics = compute_metrics)

# Start the training process.
trainer.train()
