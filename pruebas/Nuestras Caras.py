"""
Este módulo implementa un sistema de reconocimiento facial utilizando redes neuronales y OpenCV.
"""

import os
import cv2
import numpy as np
import dlib
from sklearn.manifold import Isomap  # Importar Isomap
import joblib  # Importar joblib para guardar el modelo


def load_images(folder, model_path):
    """
    Carga y preprocesa imágenes desde una carpeta.

    Args:
        folder (str): Ruta a la carpeta que contiene las imágenes.
        model_path (str): Ruta al modelo de predicción de landmarks de dlib.

    Returns:
        tuple: Imágenes preprocesadas y sus etiquetas correspondientes.
    """
    images = []
    labels = []

    # Cargar el predictor de dlib
    predictor = dlib.shape_predictor(model_path)
    detector = dlib.get_frontal_face_detector()

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Redimensionar y normalizar
            img = cv2.resize(img, (64, 64)) / 255.0

            # Aplicar CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply((img * 255).astype(np.uint8))

            # Detectar caras y landmarks
            faces = detector(img)
            if faces:
                face_rect = faces[0]
                landmarks = predictor(img, face_rect)

            # Aplicar filtro de nitidez
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)

            # Agregar a las listas
            images.append(img.flatten())
            labels.append(0 if "personA" in filename else 1)  # Etiquetas binarias

    return np.array(images), np.array(labels)


class NeuralNetwork:
    """
    Implementa una red neuronal simple para clasificación binaria.
    """

    def __init__(self, layers, alpha=0.1):
        """
        Inicializa la red neuronal.

        Args:
            layers (list): Lista con el número de neuronas por capa.
            alpha (float): Tasa de aprendizaje.
        """
        self.weights = []
        self.layers = layers
        self.alpha = alpha

        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i] + 1, layers[i + 1]) * 0.1
            self.weights.append(weight)

    def sigmoid(self, x):
        """Calcula la función de activación sigmoide."""
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Realiza la propagación hacia adelante.

        Args:
            x (ndarray): Datos de entrada.

        Returns:
            ndarray: Salida de la red.
        """
        x = np.c_[x, np.ones((x.shape[0]))]  # Agregar bias
        for weight in self.weights[:-1]:
            x = self.sigmoid(np.dot(x, weight))
        return self.sigmoid(np.dot(x, self.weights[-1]))

    def backward(self, x, y, epochs=1000):
        """
        Realiza la propagación hacia atrás y actualiza los pesos.

        Args:
            x (ndarray): Datos de entrenamiento.
            y (ndarray): Etiquetas de entrenamiento.
            epochs (int): Número de épocas.
        """
        x = np.c_[x, np.ones((x.shape[0]))]

        for epoch in range(epochs):
            activations = [x]
            for weight in self.weights:
                x = self.sigmoid(np.dot(x, weight))
                activations.append(x)

            error = activations[-1] - y.reshape(-1, 1)
            deltas = [error * activations[-1] * (1 - activations[-1])]

            for i in reversed(range(len(self.weights) - 1)):
                delta = (
                    deltas[-1].dot(self.weights[i + 1].T)
                    * activations[i + 1]
                    * (1 - activations[i + 1])
                )
                deltas.append(delta)
            deltas.reverse()

            for i in range(len(self.weights)):
                self.weights[i] -= self.alpha * activations[i].T.dot(deltas[i])


# Cargar datos
model_path = "shape_predictor_68_face_landmarks.dat"
x_train, y_train = load_images("train_data", model_path)
x_test, y_test = load_images("test_data", model_path)

# Aplicar ISOMAP para reducción de dimensionalidad
n_components = 50  # Número de dimensiones reducidas
n_neighbors = 5  # Número de vecinos para ISOMAP

isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
x_train_reduced = isomap.fit_transform(x_train)

# Guardar el modelo ISOMAP entrenado
joblib.dump(isomap, "isomap_model.pkl")

# Cargar el modelo ISOMAP (opcional, si necesitas reutilizarlo)
# isomap = joblib.load("isomap_model.pkl")

x_test_reduced = isomap.transform(x_test)

# Configurar red (ej: 50-128-64-1, ajustado a las dimensiones reducidas)
nn = NeuralNetwork([n_components, 128, 64, 1])

# Entrenar
nn.backward(x_train_reduced, y_train, epochs=2000)

# Evaluar
predictions = (nn.forward(x_test_reduced) > 0.5).astype(int)
accuracy = np.mean(predictions.flatten() == y_test)
print(f"Precisión: {accuracy * 100:.2f}%")
