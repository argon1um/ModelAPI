"""
Routes and views for the flask application.
"""
from xml.dom.minidom import ReadOnlySequentialNamedNodeMap
from keras.src.utils import pad_sequences
from datetime import datetime
from flask import jsonify, render_template
from flask import request
from ModelApi import app
import tensorflow as tf
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
import pymorphy3
import inspect
import pandas as pd
import re
import nltk
import argparse
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords

model = tf.keras.models.load_model('model.h5')
tokenizer = Tokenizer()
ma = MorphAnalyzer()

def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols  
    text = " ".join(ma.parse((word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)

    return text


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных POST запроса
    data = request.json
    
    # Преобразование данных в формат, который ожидает модель
    cleaned_text = clean_text(data)
    cleaned_list = cleaned_text.split()
    tokenized_text = tokenizer.texts_to_sequences(cleaned_list)
    padded_text = pad_sequences(tokenized_text, maxlen=36, padding='post')
    prediction = model.predict(padded_text)
    # Предсказание с использованием модели
    tensor = prediction
    scalar = tensor.sum()  # Преобразует тензор в скаляр
    
    # Возвращение результата в формате JSON
    roundedscalar = round(scalar)
    if (roundedscalar<=1):
        scalar = 1
    else:
        scalar = 2
    return jsonify(scalar)
    

