import numpy as np
import matplotlib.pyplot as plt

# --- Zadanie 1 ---

def funkcja_sigmoidalna(x, beta):
    return 1.0 / (1.0 + np.exp(-beta * x))

def pochodna_sigmoidalna(y, beta):
    return beta * y * (1 - y)

def funkcja_tanh(x, beta):
    return np.tanh(beta * x)

def pochodna_tanh(y, beta):
    return beta * (1 - y * y)

def oblicz_wyjscie_sieci(x, wagi_ukryta, wagi_wyjsciowa, beta):
    suma_ukryta = np.dot(wagi_ukryta, x)
    wyjscie_ukryta = funkcja_tanh(suma_ukryta, beta)
    
    wyjscie_ukryta_z_biasem = np.insert(wyjscie_ukryta, 0, 1)
    
    suma_wyjsciowa = np.dot(wagi_wyjsciowa, wyjscie_ukryta_z_biasem)
    y = funkcja_sigmoidalna(suma_wyjsciowa, beta)
    
    return y, wyjscie_ukryta_z_biasem

# --- Zadanie 2 ---

def czy_dobrze_sklasyfikowane(y, oczekiwane):
    if oczekiwane == 1 and y > 0.9:
        return True
    if oczekiwane == 0 and y < 0.1:
        return True
    return False

def inicjalizuj_wagi():
    w1 = np.random.uniform(-0.5, 0.5, (2, 3))
    w2 = np.random.uniform(-0.5, 0.5, (1, 3))
    return w1, w2

def trenuj_po_probce(dane_x, dane_y, wspolczynnik, beta, max_epok=100000):
    wagi_ukryta, wagi_wyjsciowa = inicjalizuj_wagi()
    historia_bledow = []
    
    for epoka in range(max_epok):
        suma_bledow = 0
        trafienia = 0
        
        for i in range(len(dane_x)):
            x = dane_x[i]
            cel = dane_y[i]
            
            y_out, v_ukryte = oblicz_wyjscie_sieci(x, wagi_ukryta, wagi_wyjsciowa, beta)
            y_wartosc = y_out[0]
            
            blad = y_wartosc - cel
            suma_bledow += blad ** 2
            
            if czy_dobrze_sklasyfikowane(y_wartosc, cel):
                trafienia += 1
            
            delta_wyjscie = blad * pochodna_sigmoidalna(y_wartosc, beta)
            gradient_w2 = delta_wyjscie * v_ukryte
            
            delta_ukryta = (delta_wyjscie * wagi_wyjsciowa[0, 1:]) * pochodna_tanh(v_ukryte[1:], beta)
            gradient_w1 = np.outer(delta_ukryta, x)
            
            wagi_wyjsciowa -= wspolczynnik * gradient_w2
            wagi_ukryta -= wspolczynnik * gradient_w1
            
        mse = 0.5 * suma_bledow / len(dane_x)
        historia_bledow.append(mse)
        
        if trafienia == len(dane_x):
            print(f"Koniec (próbka) w epoce: {epoka}")
            break
            
    return historia_bledow

def trenuj_po_epoce(dane_x, dane_y, wspolczynnik, beta, max_epok=100000):
    wagi_ukryta, wagi_wyjsciowa = inicjalizuj_wagi()
    historia_bledow = []
    
    for epoka in range(max_epok):
        suma_bledow = 0
        trafienia = 0
        
        grad_w1_suma = np.zeros_like(wagi_ukryta)
        grad_w2_suma = np.zeros_like(wagi_wyjsciowa)
        
        for i in range(len(dane_x)):
            x = dane_x[i]
            cel = dane_y[i]
            
            y_out, v_ukryte = oblicz_wyjscie_sieci(x, wagi_ukryta, wagi_wyjsciowa, beta)
            y_wartosc = y_out[0]
            
            blad = y_wartosc - cel
            suma_bledow += blad ** 2
            
            if czy_dobrze_sklasyfikowane(y_wartosc, cel):
                trafienia += 1
            
            delta_wyjscie = blad * pochodna_sigmoidalna(y_wartosc, beta)
            grad_w2_temp = delta_wyjscie * v_ukryte
            
            delta_ukryta = (delta_wyjscie * wagi_wyjsciowa[0, 1:]) * pochodna_tanh(v_ukryte[1:], beta)
            grad_w1_temp = np.outer(delta_ukryta, x)
            
            grad_w2_suma += grad_w2_temp
            grad_w1_suma += grad_w1_temp
            
        wagi_wyjsciowa -= wspolczynnik * grad_w2_suma
        wagi_ukryta -= wspolczynnik * grad_w1_suma
        
        mse = 0.5 * suma_bledow / len(dane_x)
        historia_bledow.append(mse)
        
        if trafienia == len(dane_x):
            print(f"Koniec (epoka) w epoce: {epoka}")
            break
            
    return historia_bledow

dane_xx = np.array([
    [1, -1, -1],
    [1, -1,  1],
    [1,  1, -1],
    [1,  1,  1]
])

dane_y = np.array([0, 1, 1, 0])

eta = 0.1
beta = 1.0

bledy_probka = trenuj_po_probce(dane_xx, dane_y, eta, beta)
bledy_epoka = trenuj_po_epoce(dane_xx, dane_y, eta, beta)

plt.figure(figsize=(10, 6))
plt.plot(bledy_probka, label='Metoda próbki')
plt.plot(bledy_epoka, label='Metoda epoki', linestyle='--')
plt.xlabel('Epoka')
plt.ylabel('Błąd MSE')
plt.legend()
plt.grid(True)
plt.show()