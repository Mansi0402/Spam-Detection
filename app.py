import streamlit as st;
import pickle as pk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string


def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)

tfidf = pk.load(open('vectorizer.pkl','rb'))
model = pk.load(open('model.pkl','rb'))

st.title("Spam Message Classifer")

imput_sms = st.text_area("Enter message")

if st.button("Predict"):

    # 1. Preprocess
    transformed_text = transform_text(imput_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_text])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")