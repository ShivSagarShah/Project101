from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline
import shap

import numpy as np
import pandas as pd
from scipy.special import softmax
import csv
import urllib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lime
from lime import lime_tabular


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def sentiment(text):
    use_cuda = False


    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

    model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base")
    config = AutoConfig.from_pretrained("vinai/bertweet-base")


    tokenizer.save_pretrained("vinai/bertweet-base")
    model.save_pretrained("vinai/bertweet-base")
    pred = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1, return_all_scores=True)
    explainer = shap.Explainer(pred)
    shap_values = explainer([text])
    # print(shap_values)
    shap.plots.text(shap_values)
    # shap.plots.text(shap_values[:, :, "anger"])
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # # TF
    # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
    # model.save_pretrained(MODEL)

    # text = "Good night 😊"
    # encoded_input = tokenizer(text, return_tensors='tf')
    # output = model(encoded_input)
    # scores = output[0][0].numpy()
    # scores = softmax(scores)

    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    response = {"Sentiment": [], "Probability": []}
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        response["Sentiment"].append(l)
        response["Probability"].append(np.round(float(s), 4))
    df = pd.DataFrame(response)
    return df, shap_values, explainer, pred

# # from textblob import TextBlob
# # import streamlit as st
# # y = st.text_area("Type your sentence: ")
# # res = st.button("Submit")
# # if res:
# #     edu = TextBlob(y)
# #     x = edu.sentiment.polarity
# #     if x<0:
# #         st.write("Negative")
# #     elif x == 0:
# #         st.write("Neutral")    
# #     elif x>0 and x<=1:
# #         st.write("Positive")
