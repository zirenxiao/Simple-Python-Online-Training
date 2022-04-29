from sklearn.svm import SVC
import numpy as np
import pandas as pd
import pickle
import time


class Train():

    def __init__(self, fname):
        self.clf = SVC()
        self.fname = fname

    def train(self):
        df = pd.read_csv('static/data/' + self.fname)
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)

    def saveModel(self):
        t = time.strftime('%Y%m%d%H%M%S')
        new_fname = r'static/model/' + t + '-' + self.fname + '.model'
        f = open(new_fname, 'wb')
        pickle.dump(self.clf, f)
        return new_fname

    def loadModel(self, file):
        pickle.load(file)


class Test():

    def __init__(self, fname):
        f = open('static/model/' + fname, 'rb')
        self.clf = pickle.load(f)

    def predict(self, x):
        return self.clf.predict(x)
