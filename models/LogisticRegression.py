#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
import ast
from tqdm import tqdm

def main():
    # Load data
    df = pd.read_csv("./../data/processed/data_10.csv")

    # Preprocess data
    df['Text'] = df['Text'].apply(lambda x: ast.literal_eval(x))
    df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))
    df['Text'] = df['Text'].apply(lambda x: ' '.join(map(str, x)))
    df['Tags'] = df['Tags'].apply(lambda x: ' '.join(map(str, x)))

    # Vectorize text and tags
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary='true', min_df=1)
    vectorizer.fit(df['Tags'])
    tags_dict = vectorizer.vocabulary_

    y = vectorizer.transform(df['Tags']).toarray()
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['Text']).toarray()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = MultiOutputClassifier(Perceptron()).fit(X_train, y_train)

    # Predict and print classification report
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
