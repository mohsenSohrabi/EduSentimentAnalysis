from transformers import pipeline, AutoConfig
import torch
from utilities import download_unizip_checkpoint

# Google Drive file id
file_id = '1aGuijP8LZljj2xvPE9-OVwoBhhvTjJNB'

# Destination path
dest_path = 'checkpoint-2676.zip'
device = "cuda" if torch.cuda.is_available() else "cpu"


# define model checkpoint - can be the same model that you already have on the hub
model_ckpt = 'content/training_dir/checkpoint-2676'  # model checkpoin

download_unizip_checkpoint(file_id, dest_path)



# Create the pipeline
trained_model = pipeline("text-classification", model=model_ckpt, device=device)

# Change the text below
sample_review = "It is a great course"

# Get the prediction
prediction = trained_model(sample_review)[0]

# label range which shows the stars is between 0 to 4 and we add one to change the range between 1 to 5
label_id = int(prediction['label'].split('_')[-1]) + 1

print('⭐' * label_id)
