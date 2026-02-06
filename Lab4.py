import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

sciezka_trening = '/Users/wafel/Downloads/OneDrive_2_5.02.2026/Lab2/MNIST_CSV/mnist_train.csv'
sciezka_test = '/Users/wafel/Downloads/OneDrive_2_5.02.2026/Lab2/MNIST_CSV/mnist_test.csv'


def ogarnij_dane(sciezka):
    try:
        tabela = pd.read_csv(sciezka, header=None)
        y = tabela.iloc[:, 0].values
        x = tabela.iloc[:, 1:].values
        
        x = x.astype('float32') / 255.0
        x = x.reshape(-1, 28, 28, 1)  
        
        y = tf.keras.utils.to_categorical(y, 10)
        return x, y
    except Exception as e:
        print(f"Nie pyklo wczytywanie {sciezka}: {e}")
        exit()

print("Wczytywanie danych...")
dane_x, dane_y = ogarnij_dane(sciezka_trening)
test_x, test_y = ogarnij_dane(sciezka_test)


kombajn = models.Sequential()
kombajn.add(layers.Conv2D(24, (3, 3), activation='relu', input_shape=(28, 28, 1)))
kombajn.add(layers.MaxPooling2D((2, 2)))
kombajn.add(layers.Conv2D(36, (3, 3), activation='relu'))
kombajn.add(layers.MaxPooling2D((2, 2)))
kombajn.add(layers.Flatten())
kombajn.add(layers.Dense(900, activation='relu'))
kombajn.add(layers.Dense(128, activation='relu'))
kombajn.add(layers.Dense(10, activation='softmax'))

kombajn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print("Lecimy z treningiem...")
historia = kombajn.fit(dane_x, dane_y, epochs=10, batch_size=64, validation_split=0.2)

strata, wynik = kombajn.evaluate(test_x, test_y)
print(f"Wynik na tescie: {wynik}")

plt.plot(historia.history['accuracy'], label='Trening')
plt.plot(historia.history['val_accuracy'], label='Walidacja')
plt.legend()
plt.show()