import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time

def train_svm():

    data=pd.read_csv("Breast Cancer Data.csv")
    df=data.drop("Unnamed: 32",axis=1)
    df=data.drop("id",axis=1)

    df.drop(columns=["Unnamed: 32"],inplace=True)
    X=df.drop(labels="diagnosis",axis=1)
    y=df["diagnosis"]
    X=X.values
    y=y.values

    from sklearn.preprocessing import LabelEncoder
    labelencoder_X_1 = LabelEncoder()
    y = labelencoder_X_1.fit_transform(y)

    global X_test, y_test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    from sklearn.preprocessing import StandardScaler
    global sc
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = SVC(probability=True)
    clf.fit(X_train, y_train)

    return clf

def test_svm(clf):
    output = clf.predict(X_test)
    acc = accuracy_score(y_test, output)
    print("The accuracy of testing data: ",acc)

def predict_svm(clf, inp):
    inp = sc.transform(inp)
    output = clf.predict(inp)
    acc = clf.predict_proba(inp)

    return output, acc