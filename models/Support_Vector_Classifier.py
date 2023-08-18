#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

def main():
    # Read Data
    df = pd.read_csv('./../data/processed/data_10.csv', nrows=3000)

    # Preprocess data
    df.Text = df.Text.apply(lambda x: x.replace('\'', ''))
    df.Text = df.Text.apply(lambda x: x.strip('][').split(', '))
    df.Text = df.Text.apply(lambda x: ' '.join(x))
    df.Tags = df.Tags.apply(lambda word: re.findall(r"'([^']*)'", word))

    # TAG ANALYSIS
    tags = df.Tags
    tags_dict = {}
    for tag in tags:
        for l in tag:
            tags_dict[l] = tags_dict.get(l, 0) + 1

    sorted_tags = sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)
    n = 10
    sorted_tags_topn = [tag[0] for tag in sorted_tags[:n]]

    # CREATE A SUBSET
    setA = set(sorted_tags_topn)
    d = df.iloc[3:4, :]
    for index, row in df.iterrows():
        setB = set(row.Tags)
        check = len(setA & setB)
        if check:
            row.Tags = list(setA & setB)
            if check == 1 and np.random.binomial(1, .2, 1)[0] == 1:
                d = d.append(row, ignore_index=True)
            else:
                d = d.append(row, ignore_index=True)
        if d.shape[0] > 100000:
            break

    # Train the model
    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(d['Tags'])
    tfidf = TfidfVectorizer(analyzer='word', max_features=10000)
    X = tfidf.fit_transform(d.Text)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # HYPERPARAMETERS
    c_list = [.001, .01, .1, 1, 10, 100, 500, 1000]
    f_score_dict = {str(c): 0 for c in c_list}
    for hyperparameter in c_list:
        model = LinearSVC(C=hyperparameter, penalty='l1', dual=False, max_iter=2000)
        classifier = OneVsRestClassifier(model)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f_score_dict[str(hyperparameter)] = [
            report['micro avg']['precision'],
            report['micro avg']['recall'],
            report['micro avg']['f1-score']
        ]
        print(f"hyperparameter : {hyperparameter} is done")

    # Save Scores to CSV FILE
    new = pd.DataFrame.from_dict(f_score_dict).T
    new.columns = ['Precision', 'Recall', 'F1-Score']
    new.round(4).to_csv('10.csv', index=True, header=True)

    # KERNEL
    from sklearn import svm
    model = svm.SVC(kernel='rbf')
    clf = OneVsRestClassifier(model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # PLOTTING
    file_list = ['10.csv', '20.csv', '30.csv', '40.csv', '50.csv', '80.csv']
    hp = []
    prec = {}
    recall = {}
    f1 = {}
    numTags = []
    for i, file in enumerate(file_list):
        df = pd.read_csv(file)
        df.columns = ['Hyperparameter', 'Precision', 'Recall', 'F1score']
        numTags.append(file[0:2])
        prec[str(numTags[i])] = df.Precision.values
        recall[str(numTags[i])] = df.Recall.values
        f1[str(numTags[i])] = df.F1score.values
    hp = df.Hyperparameter
    df = pd.read_csv('100.csv')
    df.columns = ['Hyperparameter', 'Precision', 'Recall', 'F1score']
    numTags.append(100)
    prec[str(100)] = df.Precision.values
    recall[str(100)] = df.Recall.values
    f1[str(100)] = df.F1score.values

    for tagcount, f1scores in f1.items():
        plt.plot(np.log(hp.values), f1scores, marker='.', markersize=10)
    plt.legend(numTags, loc=4)
    plt.grid()
    plt.xlabel("log(Hyperparameter)")
    plt.ylabel("F1-Score")
    plt.title("Comparision of F1 Scores for different values of TagCount")
    plt.savefig("F1scores.png")

if __name__ == "__main__":
    main()
