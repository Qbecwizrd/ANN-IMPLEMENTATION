import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

model=load_model('simple_rrn.h5')

def decode_review(sample_review):
    decoded_review=' '.join([reverse_word_index.get(i-3,'?') for i in sample_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)

    sentiment= 'Positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment,prediction[0][0] 

st.title("IMDB CLASSIFICATION DATASET")
st.write('enter a movie review to classify is as positive or negative')
 
#user input

user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)



    ##make prediction
    prediction=model.predict(preprocessed_input)

    st.write(f'the prediction was {prediction[0][0]}')