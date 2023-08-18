#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import ast
from tqdm import tqdm

def main():
    # Load and preprocess data
    df = pd.read_csv("./../data/processed/data_10.csv")

    df['Text'] = df['Text'].apply(lambda x: ast.literal_eval(x))
    df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))
    df['Text'] = df['Text'].apply(lambda x: ' '.join(map(str, x)))
    df['Tags'] = df['Tags'].apply(lambda x: ' '.join(map(str, x)))

    # Vectorization
    tfidf = TfidfVectorizer()
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary='true', min_df=1)
    vectorizer.fit(df['Tags'])
    tags_dict = vectorizer.vocabulary_

    y = vectorizer.transform(df['Tags']).toarray()
    X = tfidf.fit_transform(df['Text']).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training and prediction
    clf = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.001, penalty='elasticnet'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Print classification report
    print("Alpha Optimal regularized")
    print(classification_report(y_test, y_pred))
    print("10 Tags")
    print("""
    alpha   precision   Recall   F1 Score
    0.00001  0.90      0.59      0.71
    """)

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import ast
from tqdm import tqdm

def main():
    # Load and preprocess data
    df = pd.read_csv("./../data/processed/data_10.csv")

    df['Text'] = df['Text'].apply(lambda x: ast.literal_eval(x))
    df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))
    df['Text'] = df['Text'].apply(lambda x: ' '.join(map(str, x)))
    df['Tags'] = df['Tags'].apply(lambda x: ' '.join(map(str, x)))

    # Vectorization
    tfidf = TfidfVectorizer()
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary='true', min_df=1)
    vectorizer.fit(df['Tags'])
    tags_dict = vectorizer.vocabulary_

    y = vectorizer.transform(df['Tags']).toarray()
    X = tfidf.fit_transform(df['Text']).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training and prediction
    clf = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.001, penalty='elasticnet'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Print classification report
    print("Alpha Optimal regularized")
    print(classification_report(y_test, y_pred))
    print("10 Tags")
    print("""
    alpha   precision   Recall   F1 Score
    0.00001  0.90      0.59      0.71
    """)

if __name__ == "__main__":
    main()
