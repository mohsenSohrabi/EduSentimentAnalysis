# DistilBERT-based Sentiment Analysis for Educational Reviews

This project involves fine-tuning a DistilBERT model on student reviews provided by Coursera. The model is trained to perform sentiment analysis on the reviews, classifying them based on the sentiment expressed.

## Getting Started

### Inference with the Trained Model

You can directly use the fine-tuned model for inference using the `demo.py` script. This script loads the trained model and uses it to perform sentiment analysis on a sample review. You can change the `sample_review` variable in the script to test the model on different reviews.

The `demo.py` script automatically downloads the trained model checkpoint from Google Drive, so you can use the model without having to fine-tune it yourself. To use the model, simply run the `demo.py` script.

### Fine-Tuning the Model

If you wish to fine-tune the model yourself, you can do so by running the `train_model.py` script. This script fine-tunes a DistilBERT model on the review data, and saves the trained model. The fine-tuning process involves the following steps:

1. Downloading the dataset from Google Drive.
2. Transforming the data into a format suitable for Hugging Face's `datasets` library.
3. Tokenizing the data using a DistilBERT tokenizer.
4. Loading a pre-trained DistilBERT model for sequence classification from Hugging Face's model hub.
5. Fine-tuning the model on the review data.

You can start the fine-tuning process by running the following command:

- On Windows: `py train_model.py`
- On other platforms: `python train_model.py`

## Model Performance

The accuracy of the model after fine-tuning was about 79%. This means that the model correctly predicts the sentiment of the reviews 79% of the time. Happy coding! 😊
