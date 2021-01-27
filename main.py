import numpy as np
import math
import keras

class Warstwa:
    def wyjscie(self, dane_wejsciowe):
        return dane_wejsciowe

    def wstecz(self, dane_wejsciowe, gradient):
        return gradient


class Relu(Warstwa):
    def __init__(self):
        self.min = 0
        self.max = 5

    def wyjscie(self, dane_wejsciowe):
        funkcja = np.minimum(np.maximum(self.min, dane_wejsciowe), self.max)
        return funkcja

    def wstecz(self, dane_wejsciowe, gradient):
        relu_grad_min = dane_wejsciowe > self.min
        relu_grad_max = dane_wejsciowe < self.max

        for idx, x in np.ndenumerate(relu_grad_min):
            gradient[idx] = gradient[idx] * relu_grad_min[idx] * relu_grad_max[idx]

        return gradient


class Ukryta(Warstwa):
    def __init__(self, liczba_wejsc, liczba_wyjsc, beta=0.2):
        self.wagi = np.random.uniform(-1 / (math.sqrt(liczba_wejsc)), 1 / math.sqrt(liczba_wejsc),
                                      (liczba_wejsc, liczba_wyjsc))
        self.bias = np.zeros(liczba_wyjsc)
        self.beta = beta

    def wyjscie(self, dane_wejsciowe):
        return np.dot(dane_wejsciowe, self.wagi) + self.bias  # wektor danych * macierz wag w postaci
        # [dane1,dane2]*[[waga1.1,waga2.1],[waga1.2],[waga2.2]]

    def wstecz(self, dane_wejsciowe, gradient):
        gradient_wstecz = np.dot(gradient, self.wagi.T)  # self.wagi.T ???!!!!
        gradient_wagi = np.dot(dane_wejsciowe.T, gradient)
        gradient_bias = gradient.mean(axis=0)
        self.wagi = self.wagi - self.beta * gradient_wagi
        self.bias = self.bias - self.beta * gradient_bias
        return gradient_wstecz


def pochodna_funkcji_straty(wyniki_sieci, poprawny_wynik):
    wyniki_macierz = np.zeros_like(wyniki_sieci)
    wyniki_macierz[np.arange(len(wyniki_sieci)), poprawny_wynik] = 1
    gradient = (-2 * (wyniki_macierz - wyniki_sieci))
    return gradient


def funkcja_straty(wyniki_sieci, poprawny_wynik):
    wyniki_macierz = np.zeros_like(wyniki_sieci)
    wyniki_macierz[np.arange(len(wyniki_sieci)), poprawny_wynik] = 1
    MSE = (wyniki_macierz - wyniki_sieci)
    for idx, x in np.ndenumerate(MSE):
        MSE[idx] = MSE[idx] * MSE[idx]
    return MSE

def wczytaj():
    (X_treningowe, Y_treningowe), (X_testowe, Y_testowe) = keras.datasets.mnist.load_data()
    X_treningowe = X_treningowe.astype(float) / float(255)
    X_testowe = X_testowe.astype(float) / float(255)
    X_treningowe = np.reshape(X_treningowe, (X_treningowe.shape[0], X_treningowe.shape[1] * X_treningowe.shape[2]))
    X_testowe = np.reshape(X_testowe, (X_testowe.shape[0], X_testowe.shape[1] * X_testowe.shape[2]))

    return X_treningowe, Y_treningowe, X_testowe, Y_testowe


X_treningowe, Y_treningowe, X_testowe, Y_testowe = wczytaj()

siec = []
siec.append(Ukryta(X_treningowe.shape[1], 300))
siec.append(Relu())
siec.append(Ukryta(300, 100))
siec.append(Relu())
siec.append(Ukryta(100, 10))


def przejdz_po_warstwach(siec, dane_wejsciowe):
    wyjscia = []
    wejscie_warstwa = dane_wejsciowe
    for warstwa in siec:
        wyjscia.append(warstwa.wyjscie(wejscie_warstwa))
        wejscie_warstwa = wyjscia[-1]

    return wyjscia


def przetestuj(network, X):
    wyjscia = przejdz_po_warstwach(network, X)
    przewidywania = wyjscia[-1]
    return przewidywania.argmax()


def trenuj(siec, dane_wejsciowe, poprawna_odpowiedz):
    wyjscia = przejdz_po_warstwach(siec, dane_wejsciowe)
    wejsia_warstw = [dane_wejsciowe] + wyjscia
    przewidywania = wyjscia[-1]

    pochodna_straty = pochodna_funkcji_straty(przewidywania, poprawna_odpowiedz)
    for idx in range(len(siec) - 1, -1, -1):
        pochodna_straty = siec[idx].wstecz(wejsia_warstw[idx], pochodna_straty)


r = 50000
for id_treningowe in range(r):
    trenuj(siec,np.matrix(X_treningowe[id_treningowe]),np.matrix(Y_treningowe[id_treningowe]))
    if id_treningowe == r/4:
        print("25%")
    if id_treningowe == 2*r/4:
        print("50%")
    if id_treningowe == 3*r/4:
        print("75%")
    if id_treningowe == 4*r/4 - 1:
        print("100%")


licznik = 0
for i in range(10000):
    wynik = przetestuj(siec,np.matrix(X_testowe[i]))
        #wyjscia = przejdz_po_warstwach(siec,np.matrix(X_treningowe[i]))
        #przewidywania = wyjscia[-1]
        #print(funkcja_straty(przewidywania,Y_treningowe[i]))
    if wynik == Y_testowe[i]:
        licznik += 1
print(licznik)
