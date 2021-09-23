from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
from scipy.special import softmax
import csv
import urllib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def sentiment(text):


    tokenizer = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")

    model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/beto-sentiment-analysis")
    config = AutoConfig.from_pretrained("finiteautomata/beto-sentiment-analysis")

    tokenizer.save_pretrained("finiteautomata/beto-sentiment-analysis")
    model.save_pretrained("finiteautomata/beto-sentiment-analysis")
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    response = {"Sentiment": [], "Probability": []}
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        response["Sentiment"].append(l)
        response["Probability"].append(np.round(float(s), 4))
    df = pd.DataFrame(response)
    return df
