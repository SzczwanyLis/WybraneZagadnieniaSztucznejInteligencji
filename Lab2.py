import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image

sciezka_train = '/Users/wafel/Downloads/OneDrive_2_5.02.2026/Lab2/MNIST_CSV/mnist_train.csv'
sciezka_test = '/Users/wafel/Downloads/OneDrive_2_5.02.2026/Lab2/MNIST_CSV/mnist_test.csv'

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.wagi = None
        self.bias = 0
        self.bledy = []

    def fit(self, X, y):
        self.wagi = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(self.n_iter):
            bledy_iter = 0
            for xi, cel in zip(X, y):
                update = self.eta * (cel - self.predict(xi))
                self.wagi += update * xi
                self.bias += update
                if update != 0:
                    bledy_iter += 1
            self.bledy.append(bledy_iter)
        return self

    def net_input(self, X):
        return np.dot(X, self.wagi) + self.bias

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

try:
    dane_train = pd.read_csv(sciezka_train, header=None)
    dane_test = pd.read_csv(sciezka_test, header=None)
except FileNotFoundError:
    print("Brak plikow")
    exit()

def ogarnij_dane(df, c1, c2):
    filtr = df[0].isin([c1, c2])
    temp = df[filtr]
    y = temp.iloc[:, 0].values
    X = temp.iloc[:, 1:].values
    y = np.where(y == c1, 0, 1)
    return X, y

c1, c2 = 0, 1
X_uczace, y_uczace = ogarnij_dane(dane_train, c1, c2)
X_testowe, y_testowe = ogarnij_dane(dane_test, c1, c2)

X_uczace = X_uczace / 255.0
X_testowe = X_testowe / 255.0

perc = Perceptron(eta=0.1, n_iter=20)
perc.fit(X_uczace, y_uczace)

plt.plot(range(1, len(perc.bledy) + 1), perc.bledy, marker='o')
plt.show()

y_pred = perc.predict(X_testowe)
zle = (y_testowe != y_pred).sum()
print(zle)

print(confusion_matrix(y_testowe, y_pred))
print(accuracy_score(y_testowe, y_pred))
print(precision_score(y_testowe, y_pred))
print(recall_score(y_testowe, y_pred))
print(f1_score(y_testowe, y_pred))

try:
    img = Image.open('moj_test.png').convert('L')
    img = img.resize((28, 28))
    arr = np.array(img)
    arr = 255 - arr
    vec = arr.flatten() / 255.0
    wynik = perc.predict(vec)
    print(c1 if wynik == 0 else c2)
except:
    pass