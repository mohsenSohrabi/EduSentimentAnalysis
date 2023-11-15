from transformers import DistilBertTokenizer
import numpy as np
from sklearn.metrics import f1_score 

def tokenize_fn(batch, tokenizer):
    return tokenizer(batch['Review'], truncation=True, padding='max_length', max_length=512)

def compute_metrics(logits_and_labels):
  logits, labels = logits_and_labels
  predictions = np.argmax(logits,axis=-1)
  acc = np.mean(predictions == labels)
  f1 = f1_score(labels, predictions, average='macro')
  return {'accuracy':acc, 'f1':f1}