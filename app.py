import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pickle
from PIL import Image




model = load_model('model2.h5')


negation_exceptions = {'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
                         'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                         'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                         'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                         'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}


stop_words = set(stopwords.words('english'))
stop_words.update(['br', 'one', 'character', 'film', 'movie'])
stop_words.remove('not')
stop_words.remove('no')
stop_words = stop_words - negation_exceptions


def predict_sentiment(sample):


    sample = sample.lower()

    sample = re.sub("[^a-z\s\']", "", sample)

    pattern = r'\b\w*(\w)\1{2, }\w*\\b|\b\w{1}\b'

    sample = re.sub(pattern, '', sample)

    token_sample = word_tokenize(sample)

    filtered_sample = [word for word in token_sample if word not in stop_words]

    ps = PorterStemmer()
    stemm_text = [ps.stem(word) for word in filtered_sample]

    sample = ' '.join(stemm_text)

    sample = [sample]

    text = sample

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen = 200)

    prediction = model.predict(padded_sequence)


    return prediction[0][0]

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon=":smile:",  # You can use an emoji or path to an image file
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .stApp {
        background: url('https://miro.medium.com/v2/resize:fit:1400/1*TKUne_iAVLScrDUGHqgG1g.jpeg');
        background-size: cover;
       }
       
    .stButton button {
        display: block;
        margin: 0 auto;
        color: silver;
    }
    .stTextInput > div > div > input, .stTextArea textarea, .stText, .stTitle, .stMarkdown {
        color: silver;
    }
    .stTextInput > div > div > input, .stTextArea textarea {
        background-color: #0C1844;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Sentiment Analysis Movie comments App')

user_input = st.text_area("Enter your text here:")


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 200

if st.button('Predict'):
    if user_input:
        prediction = predict_sentiment(user_input)
        if prediction >= 0.5:
            st.write(f'Positive sentiment')
            img = Image.open('positive.png')
            st.image(img)

        else:
            st.write(f'Negative sentiment')
            img = Image.open('negative.png')
            st.image(img)

    else:
        st.write("Please enter some text to analyze.")