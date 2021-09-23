from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import streamlit as st
import pandas as pd
def similar(sentences):
    model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    sentences = sentences.split("\n")
    embeddings = model.encode(sentences)
    data = []

    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            dictionary = {"Sentence 1": sentences[i], "Sentence 2": sentences[j], 
            "Similarity Probability": cosine_similarity(embeddings[i].reshape(1, -1),embeddings[j].reshape(1, -1))[0][0]}
            data.append(dictionary)
    df = pd.DataFrame(data)
    return df
