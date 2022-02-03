#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:37:40 2022

@author: vinis
"""

# -*- coding: utf-8 -*-
import numpy as np
import pickle
import nltk
import re
import streamlit as st
from PIL import Image

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

vec_file = pickle.load(open(r'C:\Users\shiva\ExcelR\Project2\deploy\vectorizer.pkl', 'rb'))
model_file = pickle.load(open(r'C:\Users\shiva\ExcelR\Project2\modelll.pkl', 'rb'))


def predict_sentiment(sample_review):
    sample_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)
    temp = vec_file.transform([final_review]).toarray()
    return model_file.predict(temp)


def model(review):
    x = predict_sentiment(review)
    x = x.astype('int')
    x1 = np.array([1])
    x2 = np.array([0])

    if np.array_equal(x, x1):
        return "This is a Moderate review."
    elif np.array_equal(x, x2):
        return "This is a Good review!"
    else:
        return "This is a Bad review "


def get_image():
    img = Image.open(r"C:\Users\shiva\ExcelR\Project2\deploy\ratings.PNG")
    return img


def main():
    st.title("Umbra Floating Book Shelf Reviews Classification")
    st.image(get_image())
    st.sidebar.image("https://m.media-amazon.com/images/I/61f1z4yxMUL._SL1500_.jpg")
    st.sidebar.image("https://m.media-amazon.com/images/I/915WuUux1EL._SX679_.jpg")
    st.sidebar.image("https://m.media-amazon.com/images/I/81SY42bP5gL._SL1500_.jpg")
    review = st.text_input("Please give review")

    results = ""
    if st.button("Click Here"):
        results = model(review)

    st.success(results)


# model("Probably great for smaller books")
#  model('Get the large set')
#  model('junk')
#  model('Missing hardware.')
#  model('Two Stars')
#  model('Not very stable')
#  model("Measure your books before buying this")
# model("Disappointed")
# model("great ")

if __name__ == '__main__':
    main()

