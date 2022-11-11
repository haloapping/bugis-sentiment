import utils
import options
import streamlit as st 

st.set_page_config(
	    page_title="Buginese Sentiment",
	    page_icon="💚",
	    layout="centered"
)

st.write(
    "<h3 style='margin-bottom: 30px; text-align: center;'>Buginese Languange Sentiment Analysis</h3>",
    unsafe_allow_html=True
)

st.write(
    "<p style='margin-bottom: 30px; text-align: center;'>\
        This app analyze sentiment for Buginese languange (negative, neutral, or positive). \
        Use random forest machine learning algorithm and count, one hot, and tf-idf for feature extraction. \
        Inspired by <a href='https://github.com/IndoNLP/nusax'>IndoNLP - nusax</a>. \
    </p>",
    unsafe_allow_html=True
)

# Select text
texts = options.texts

st.write(
    "<h4 style='margin-top: 20px; text-align: left;'>Text</h4>",
    unsafe_allow_html=True
)

selected_text = st.selectbox(
    '',
    options=list(texts.keys()),
    format_func=lambda option: texts[option],
    label_visibility="hidden"
)

text = st.text_area(
    label="",
    value=texts[selected_text]
)

# Select model
models = options.models

st.write(
    "<h4 style='margin-top: 20px; text-align: left;'>Model</h4>",
    unsafe_allow_html=True
)

selected_model = st.selectbox(
    '',
    options=list(models.keys()),
    format_func=lambda option: models[option],
    label_visibility="hidden"
)

# load model
if selected_model == "model_1":
    model = utils.load_model("count")
elif selected_model == "model_2":
    model = utils.load_model("one_hot")
else:
    model = utils.load_model("tf_idf")

# prediction
st.write(
    "<h4 style='margin-top: 20px; margin-bottom: 30px; text-align: left;'>Sentiment Analysis</h4>",
    unsafe_allow_html=True
)

sentiment = model.predict([text]).item()
sentiment_proba = model.predict_proba([text]).max() * 100
st.info(f"Sentiment is {sentiment} with probability {sentiment_proba:.1f}%.", icon="ℹ️")

st.write(
    "<p style='margin-top: 50px; text-align: center;'>\
        Made with 💚 by <a href='https://haloapping.github.io/'>haloapping</a>\
    </p>",
    unsafe_allow_html=True
)
