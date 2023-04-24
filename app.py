import streamlit as st
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import torch


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


st.title("Sentiment Analysis App")

user_input = st.text_input("Input a sentence to analyze", "I'm happy")

option = st.selectbox("Select Model", ("pre-trained", "fine-tuning"))

if option == "pre-trained":
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
else:
    MODEL = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained("./saved/")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL)


if st.button("Analyze"):
    if option == "pre-trained":
        text = preprocess(user_input)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            st.write(f"{i + 1}) {l} {np.round(float(s), 4)}")
    else:
        input_val = tokenizer(user_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
        output_val = model(**input_val)
        probabilities = torch.sigmoid(output_val.logits)
        print(output_val)
        print(probabilities)
        st.write(probabilities)