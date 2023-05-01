from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
import torch

# test the available of GPU
print("CUDA is available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))

# import the training dataset
df = pd.read_csv("train.csv")
data = np.array(df)
# print(df.head())

# select the model we based on
model_name = "bert-base-uncased"


train_texts = data[:, 1].tolist()
# print(len(train_texts))
train_labels = data[:, 2:].tolist()
# print(len(train_labels[0]))


# build tre dataset
class dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


tokenizer = BertTokenizerFast.from_pretrained(model_name)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_dataset = dataset(train_encodings, train_labels)

# set the arguments
training_args = TrainingArguments(output_dir='./result',
                                  num_train_epochs=2,
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=64,
                                  warmup_steps=500,
                                  learning_rate=5e-5,
                                  weight_decay=0.01,
                                  logging_dir='./logs',
                                  logging_steps=10,)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,)

trainer.train()


# ========================================


# from torch.utils.data import DataLoader
# from transformers import AdamW
#
# device = torch.device('cuda')
# model = BertForSequenceClassification.from_pretrained(model_name)
# model.to(device)
# model.train()
#
# train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
#
# optim = AdamW(model.parameters(), lr=5e-5)
#
# num_train_epochs = 2
# for epoch in range(num_train_epochs):
#     for batch in train_loader:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention'].to(device)
#         labels = batch['labels'].to(device)
#
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#
#         loss = outputs[0]
#         loss.backward()

model.eval()
