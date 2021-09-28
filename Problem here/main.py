from shap import explainers
import streamlit as st
import lime
import shap
import numpy as np
import streamlit.components.v1 as components
from sentiment_analysis import sentiment
from summarisation import summary
from sentence_similarity import similar
text = st.text_area("Text Box")
option = st.selectbox("Select the option", ("Text Summarisation", "Sentiment Analysis", "Sentence Similarity"))
res = st.button("Submit")
def st_shap(plot, height=700, width = 1000):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, width=width)
def run(option):
    if "Text Summarisation" in option:
        st.write(summary(text))
    if "Sentiment Analysis" in option:
        # st.dataframe(sentiment(text))
        data, shap_values, explainer, model= sentiment(text)
        print(data)
        st.dataframe(data)

        # print(type([shap_values]))
        # print(explainer.expected_value)
        lst = np.unique(shap_values[0,:].data)
        dst = np.array([-1, 0 ,1])
        feature = np.array(["Negative", "Neutral", "Positive"])

        # print(shap_values[0,:])
        pmodel = shap.models.TransformersPipeline(model, rescale_to_logits=False)
        explainer2 = shap.Explainer(pmodel)
        shap_values = explainer2([text])

        shap.plots.text(shap_values[:,:,1])
        # print(plt)
        # rawhtml = plt._repr_html_()
        # components.html(rawhtml)
        # st_shap(shap.force_plot(shap_values[0,:].values, feature))

        # st.dataframe(shap_values)
        # st.dataframe(explainer)


    if "Sentence Similarity" in option:
        st.dataframe(similar(text))
      
if res:
    run(option)
    
