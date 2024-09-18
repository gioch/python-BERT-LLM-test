from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset
import torch


print('Load Dataset ')
dataset = load_dataset("yelp_review_full", split="train[:1%]")

print('initialize Models and Tokenizers')
# Load the pre-trained BERT model for sequence classification
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

device = 0 if torch.cuda.is_available() else -1

def preprocess_data(examples):
  return tokenizer(examples['text'], padding="max_length", truncation=True)

print('Encode Datasets')
# Preprocess the data for BERT
encoded_dataset = dataset.map(preprocess_data, batched=True)

print('Prepare the Trainer')
# Define training arguments

# training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8, save_steps=10_000, save_total_limit=2)
# trainer = Trainer(model=model, args=training_args, train_dataset=encoded_dataset)
# trainer.train()

print('Prepare Pipeline Before Training')
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

# Test with an example sentence
print('Result Before Training')

print(nlp("The stock market is bullish today."))

# print('Prepare Pipeline After Training')
# trainer.save_model("./fine_tuned_model")
# fine_tuned_nlp = pipeline("sentiment-analysis", model="./fine_tuned_model", tokenizer=tokenizer)

# print('Result After Training')
# print(fine_tuned_nlp("The stock market is bullish today."))
