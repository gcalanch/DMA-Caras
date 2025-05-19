#
# Version de chatGPT

# red_facial.py

### 1. Importaciones necesarias
import numpy as np
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os


### 2. Cargar y preprocesar imágenes (ya lo tenés implementado)
def cargar_imagenes_y_etiquetas(path):
    # devuelve X (imagenes preprocesadas) y y (etiquetas)
    pass


### 3. Reducción de dimensionalidad con Isomap
def aplicar_isomap(X, n_components=50, n_neighbors=5):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_reducido = isomap.fit_transform(X)
    return X_reducido, isomap


### 4. Inicialización de la red neuronal
def inicializar_pesos(n_entrada, n_oculta, n_salida):
    W1 = np.random.randn(n_oculta, n_entrada) * 0.01
    b1 = np.zeros((n_oculta, 1))
    W2 = np.random.randn(n_salida, n_oculta) * 0.01
    b2 = np.zeros((n_salida, 1))
    return W1, b1, W2, b2


### 5. Forward pass
def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / np.sum(e_z, axis=0, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


### 6. Cálculo de la función de pérdida (cross-entropy)
def calcular_loss(Y_hat, Y):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
    return loss


### 7. Backward pass y actualización de pesos
def backward(X, Y, Z1, A1, A2, W2):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (1 - np.tanh(Z1) ** 2)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2


def actualizar_pesos(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2


### 8. Entrenamiento
def entrenar(X, Y, n_oculta, lr, epochs):
    n_entrada = X.shape[0]
    n_salida = Y.shape[0]
    W1, b1, W2, b2 = inicializar_pesos(n_entrada, n_oculta, n_salida)

    for i in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        loss = calcular_loss(A2, Y)
        dW1, db1, dW2, db2 = backward(X, Y, Z1, A1, A2, W2)
        W1, b1, W2, b2 = actualizar_pesos(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")
    return W1, b1, W2, b2


### 9. Predicción y evaluación
def predecir(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=0)


def precision(Y_pred, Y_true):
    return np.mean(Y_pred == Y_true)


### 10. Loop principal con ajuste de hiperparámetros
def main():
    # Cargar datos
    X_raw, y_raw = cargar_imagenes_y_etiquetas("ruta/a/imagenes")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    Y_onehot = np.eye(len(le.classes_))[y_encoded].T

    # Dividir en entrenamiento y test
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X_raw, Y_onehot.T, test_size=0.2, random_state=42
    )

    # Isomap
    X_train_iso, isomap = aplicar_isomap(X_train_raw)
    X_test_iso = isomap.transform(X_test_raw)

    # Transponer para que columnas sean ejemplos
    X_train = X_train_iso.T
    Y_train = Y_train.T
    X_test = X_test_iso.T
    Y_test_labels = np.argmax(Y_test, axis=1)

    # Entrenar red
    W1, b1, W2, b2 = entrenar(X_train, Y_train, n_oculta=64, lr=0.05, epochs=1000)

    # Predecir y evaluar
    Y_pred = predecir(X_test, W1, b1, W2, b2)
    acc = precision(Y_pred, Y_test_labels)
    print(f"Precisión en test: {acc:.4f}")


if __name__ == "__main__":
    main()
