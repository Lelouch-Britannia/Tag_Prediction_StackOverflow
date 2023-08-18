import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

    # Vectorize tags
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary='true', min_df=1)
    vectorizer.fit(df['Tags'])
    y = vectorizer.transform(df['Tags']).toarray()

    # Vectorize text
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['Text']).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = DecisionTreeClassifier(random_state=1, max_depth=1000).fit(X_train, y_train)

    # Predict and print classification report
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
