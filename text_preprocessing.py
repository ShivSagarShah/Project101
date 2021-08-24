from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from os import sep
import streamlit as st
import pandas as pd
import io
import re
from spellchecker import SpellChecker
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

data = ""

status = st.radio("What do you want", ("Write", "Upload"))
if status == "Write":
    text = st.text_area("Text Box")
else:
    uploaded_file = st.file_uploader("Upload a File")
    if uploaded_file:
        for line in uploaded_file:
            data += line.decode("UTF-8") + '\n'

preprocess = st.multiselect("What all do you want?", ("Lowercasing", "Remove Extra Whitespaces", "Tokenization", "Spelling Correction",
                                                      "Removing Stopwords", "Removing Punctuations", "Removing Frequent Words", "Lemmatization", "Stemming", "Removal of Tags", "Removal of URLs", ""))

def remove_whitespace(text):
    return " ".join(text.split())


def spell_check(text):
    result = []
    spell = SpellChecker()
    for word in text:
        correct_word = spell.correction(word)
        result.append(correct_word)

    return result


def remove_stopwords(text):
    result = []
    en_stopwords = stopwords.words('english')
    for token in text:
        if token not in en_stopwords:
            result.append(token)

    return result


def remove_punct(text):

    tokenizer = RegexpTokenizer(r"\w+")
    lst = tokenizer.tokenize(' '.join(text))
    return lst


def frequent_words(df):

    lst = []
    for text in df.values:
        lst.append(text[0])

    fdist = FreqDist(lst)
    return fdist.most_common(2)


def remove_freq_words(text, lst):
    result = []
    for item in text:
        if item not in lst:
            result.append(item)

    return result


def lemmatization(text):

    result = []
    wordnet = WordNetLemmatizer()
    for token, tag in pos_tag(text):
        pos = tag[0].lower()

        if pos not in ['a', 'r', 'n', 'v']:
            pos = 'n'

        result.append(wordnet.lemmatize(token, pos))

    return result


def stemming(text):
    porter = PorterStemmer()

    result = []
    for word in text:
        result.append(porter.stem(word))
    return result


def remove_tag(text):

    text = ' '.join(text)
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    print(text)
    return url_pattern.sub(r'', text)


def get_dataset(preprocess):
    if "Lowercasing" in preprocess:
        if status == "Write":
            df = pd.DataFrame([text])
            st.write(df[0].str.lower())
        else:
            df = pd.DataFrame([data])
            st.write(df[0].str.lower())



    if "Remove Extra Whitespaces" in preprocess:
        if status == "Write":
            st.write(remove_whitespace(text))
        else:
            st.write(remove_whitespace(data))

    if "Tokenization" in preprocess:
        if status == "Write":
            df = pd.DataFrame([text])
            st.write(df[0].apply(lambda X: word_tokenize(X)))
        else:
            df = pd.DataFrame([data])
            st.write(df[0].apply(lambda X: word_tokenize(X)))


    if "Spelling Correction" in preprocess: 
        if status == "Write":
            st.write(spell_check(text.split()))
        else:
             st.write(spell_check(data.split()))

   
    if "Removing Stopwords" in preprocess:
        if status == "Write":
            st.write(remove_stopwords(text.split()))
        else:
            st.write(remove_stopwords(data.split()))

    
    if "Removing Punctuations" in preprocess:
        if status == "Write":
            st.write(remove_punct(text.split()))
        else:
            st.write(remove_punct(data.split()))
    
    if "Removing Frequent Words" in preprocess:
        if status == "Write":
            df = pd.DataFrame(text.split())
            freq_words = frequent_words(df)
            lst = []
            for a,b in freq_words:
                lst.append(a)
            st.write(remove_freq_words(text.split(), lst))
        else:
            df = pd.DataFrame(data.split())
            freq_words = frequent_words(df)
            lst = []
            for a,b in freq_words:
                lst.append(a)
            st.write(remove_freq_words(data.split(), lst))



    if "Lemmatization" in preprocess:
        if status == "Write":
            st.write(lemmatization(text.split()))
        else:
            st.write(lemmatization(data.split()))


    if "Stemming" in preprocess:
        if status == "Write":
            st.write(stemming(text.split()))
        else:
            st.write(stemming(data.split()))

    if "Removal of Tags" in preprocess:
        if status == "Write":
            st.write(remove_tag(text.split()))
        else:
            st.write(remove_tag(data.split()))

    if "Removal of URLs" in preprocess:
        if status == "Write":
            st.write(remove_urls(text))
        else:
            st.write(remove_urls(data))

get_dataset(preprocess)
