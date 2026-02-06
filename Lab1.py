import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Perceptron:
    def __init__(self, n, bias=True):
        self.w = np.random.randn(n)
        self.b = np.random.randn() if bias else 0

    def predict(self, x):
        y = np.dot(x, self.w) + self.b
        return np.where(y >= 0, 1, -1)

    def train(self, xx, d, eta=0.1, tol=0):
        while True:
            errors = 0
            for i in range(len(xx)):
                y = self.predict(xx[i])
                if y != d[i]:
                    self.w += eta * (d[i] - y) * xx[i]
                    self.b += eta * (d[i] - y)
                    errors += 1
            if errors <= tol:
                break

    def evaluate_test(self, xx, d):
        y_pred = self.predict(xx)
        error = np.mean(y_pred != d)
        return error, y_pred



path = "/Users/wafel/Downloads/OneDrive_2_11.10.2025/"

df2 = pd.read_csv(path + "2D.csv", skiprows=1, delimiter=';', names=['X1', 'X2', 'L'], decimal=',')
df3 = pd.read_csv(path + "3D.csv", skiprows=1, delimiter=';', names=['X1', 'X2', 'X3', 'L'], decimal=',')


def prepare_data(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = np.where(y == 0, -1, 1)  # klasy -1 i 1
    # podział 80% trening, 20% test
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(X))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]



X_train2, y_train2, X_test2, y_test2 = prepare_data(df2)
p2 = Perceptron(n=2)
p2.train(X_train2, y_train2, eta=0.1, tol=0)
err2, y_pred2 = p2.evaluate_test(X_test2, y_test2)
print(f"Błąd testowy (2D): {err2*100:.2f}%")


plt.figure()
plt.scatter(X_test2[y_pred2 == 1, 0], X_test2[y_pred2 == 1, 1], color='blue', label='Klasa +1')
plt.scatter(X_test2[y_pred2 == -1, 0], X_test2[y_pred2 == -1, 1], color='red', label='Klasa -1')

x_vals = np.linspace(min(X_test2[:, 0]), max(X_test2[:, 0]), 100)
y_vals = -(p2.w[0] * x_vals + p2.b) / p2.w[1]
plt.plot(x_vals, y_vals, 'k--', label='Granica decyzji')
plt.title("Perceptron - dane 2D")
plt.legend()
plt.show()



X_train3, y_train3, X_test3, y_test3 = prepare_data(df3)
p3 = Perceptron(n=3)
p3.train(X_train3, y_train3, eta=0.1, tol=0)
err3, y_pred3 = p3.evaluate_test(X_test3, y_test3)
print(f"Błąd testowy (3D): {err3*100:.2f}%")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test3[y_pred3 == 1, 0], X_test3[y_pred3 == 1, 1], X_test3[y_pred3 == 1, 2], c='black', label='Klasa +1')
ax.scatter(X_test3[y_pred3 == -1, 0], X_test3[y_pred3 == -1, 1], X_test3[y_pred3 == -1, 2], c='purple', label='Klasa -1')

# płaszczyzna separująca
xx, yy = np.meshgrid(np.linspace(min(X_test3[:, 0]), max(X_test3[:, 0]), 20),
                     np.linspace(min(X_test3[:, 1]), max(X_test3[:, 1]), 20))
zz = -(p3.w[0]*xx + p3.w[1]*yy + p3.b) / p3.w[2]
ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

ax.set_title("Perceptron - dane 3D")
ax.legend()
plt.show()
