#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import ast
from tqdm import tqdm

def get_model(n_inputs, n_outputs):
    """Define and compile the neural network model."""
    model = Sequential()
    model.add(Dense(100, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model

def main():
    # Load data
    df = pd.read_csv("./../data/processed/data_10.csv", nrows=3000)

    # Preprocess data
    df['Text'] = df['Text'].apply(lambda x: ast.literal_eval(x))
    df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))
    df['Text'] = df['Text'].apply(lambda x: ' '.join(map(str, x)))
    df['Tags'] = df['Tags'].apply(lambda x: ' '.join(map(str, x)))

    # Vectorize text and tags
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary='true', min_df=1)
    vectorizer.fit(df['Tags'])
    y = vectorizer.transform(df['Tags']).toarray()
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['Text']).toarray()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train the model
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    model = get_model(n_inputs, n_outputs)
    model.fit(X_train, y_train, verbose=0, epochs=100)

    # Predict and print classification report
    yhat = model.predict(X_test)
    yhat = yhat.round()
    print(classification_report(y_test, yhat))

if __name__ == "__main__":
    main()
