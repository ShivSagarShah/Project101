import streamlit as st
from sentiment_analysis import sentiment
from summarisation import summary
from sentence_similarity import similar
text = st.text_area("Text Box")
option = st.selectbox("Select the option", ("Text Summarisation", "Sentiment Analysis", "Sentence Similarity"))
res = st.button("Submit")
def run(option):
    if "Text Summarisation" in option:
        st.write(summary(text))
    if "Sentiment Analysis" in option:
        st.dataframe(sentiment(text))
    if "Sentence Similarity" in option:
        st.dataframe(similar(text))
      
if res:
    run(option)
    
