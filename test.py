from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch

from transformers import BertTokenizerFast, BertForSequenceClassification

model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained("./saved/")
tokenizer = BertTokenizerFast.from_pretrained(model_name)

X_val = "I love you"
input_val = tokenizer(X_val, padding=True, truncation=True, max_length=512, return_tensors="pt")
output_val = model(**input_val)
probabilities = torch.sigmoid(output_val.logits)
print(output_val)
print(probabilities)

# ============================================================

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#
# res = classifier("I'm happy")
#
# print(res)
#
# tokens = tokenizer.tokenize("I'm happy")
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# input_ids = tokenizer("I'm happy")
#
# print(tokens)
# print(token_ids)
# print(input_ids)
#
# X_train = ["I'm happy",
#            "I'm so sad"]
#
# batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
# print(batch)
#
# with torch.no_grad():
#     # add labels to calculate the loss
#     outputs = model(**batch, labels=torch.tensor([1, 0]))
#     print(outputs)
#     predictions = F.softmax(outputs.logits, dim=1)
#     print(predictions)
#     labels = torch.argmax(predictions, dim=1)
#     print(labels)
#     labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
#     print(labels)
#
# save_directory = "saved"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)
#
# tokenizer = AutoTokenizer.from_pretrained(save_directory)
# model = AutoModelForSequenceClassification.from_pretrained(save_directory)

# ============================================================

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# texts = ["I'm happy",
#          "I'm so sad"]
#
# batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
# print(batch)
#
# with torch.no_grad():
#     # add labels to calculate the loss
#     outputs = model(**batch, labels=torch.tensor([1, 0]))
#     label_ids = torch.argmax(outputs.logits, dim=1)
#     print(label_ids)
#     labels = [model.config.id2label[label_id] for label_id in label_ids.tolist()]
#     print(labels)