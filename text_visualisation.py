from itertools import count
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import sys
import operator
import argparse
from nltk import corpus
import cufflinks as cf
from sklearn.feature_extraction.text import CountVectorizer

text = st.text_area("Text Box")
option = st.multiselect("Select the option", ("Uni-Gram","Bi-Grams", "Tri-Grams"))
res = st.button("Submit")

def get_num_words_per_sample(text):
    if text == "":
        return
    num_words = [len(s.split()) for s in text]
    return np.median(num_words)

def plot_sample_length_distribution(text):
    plt.hist([len(s) for s in text], 50)
    st.pyplot()
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

#Uni gram
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


#Bi-Gram
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]



#Tri-Gram
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def options(option):
        if "Uni-Gram" in option:
            common_words = get_top_n_words([text], 50)
            for word, freq in common_words:
                print(word, freq)
            df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
            fig = df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(asFigure=True, kind='bar', yTitle='Count', linecolor='black', title='Top 50 words in review')
            st.plotly_chart(fig)
        if "Bi-Gram" in option:
            common_words = get_top_n_bigram([text], 50)
            for word, freq in common_words:
                print(word, freq)
            df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
            st.write(df1.head())
            fig = df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(asFigure=True, kind='bar', yTitle='Count', linecolor='black', title='Top 50 words in review')
            st.plotly_chart(fig)

        if "Tri-Gram" in option:
            common_words = get_top_n_trigram([text], 50)
            for word, freq in common_words:
                print(word, freq)
            df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
            fig = df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(asFigure=True, kind='bar', yTitle='Count', linecolor='black', title='Top 50 words in review')
            st.plotly_chart(fig)    
    # st.write(get_num_words_per_sample(text))

if res:
# Create and generate a word cloud image:
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    options(option)