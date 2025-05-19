#
# Version de Gemini
#

import numpy as np
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------------------
# 1. Funciones de Activación
# ------------------------------------------------------------------------------


def sigmoid(x):
    """Función sigmoide."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivada(x):
    """Derivada de la función sigmoide."""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """Función ReLU (Rectified Linear Unit)."""
    return np.maximum(0, x)


def relu_derivada(x):
    """Derivada de la función ReLU."""
    return np.where(x > 0, 1, 0)


def softmax(z):
    """Función softmax para normalizar la salida."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ------------------------------------------------------------------------------
# 2. Capa de la Red Neuronal
# ------------------------------------------------------------------------------


class Capa:
    def __init__(
        self, num_entradas, num_neuronas, funcion_activacion, derivada_activacion
    ):
        self.num_entradas = num_entradas
        self.num_neuronas = num_neuronas
        self.funcion_activacion = funcion_activacion
        self.derivada_activacion = derivada_activacion
        self.pesos = (
            np.random.randn(num_entradas, num_neuronas) * 0.01
        )  # Inicialización pequeña
        self.bias = np.zeros((1, num_neuronas))
        self.activacion_entrada = None
        self.salida = None

    def forward(self, entrada):
        """Propagación hacia adelante."""
        self.activacion_entrada = entrada
        self.z = np.dot(entrada, self.pesos) + self.bias
        self.salida = self.funcion_activacion(self.z)
        return self.salida

    def backward(self, error_capa_siguiente, pesos_capa_siguiente):
        """Propagación hacia atrás y cálculo de gradientes."""
        self.error = np.dot(
            error_capa_siguiente, pesos_capa_siguiente.T
        ) * self.derivada_activacion(self.z)
        self.gradiente_pesos = np.dot(self.activacion_entrada.T, self.error)
        self.gradiente_bias = np.sum(self.error, axis=0, keepdims=True)
        return self.error, self.pesos

    def actualizar_parametros(self, tasa_aprendizaje):
        """Actualización de pesos y bias."""
        self.pesos -= tasa_aprendizaje * self.gradiente_pesos
        self.bias -= tasa_aprendizaje * self.gradiente_bias


# ------------------------------------------------------------------------------
# 3. Red Neuronal
# ------------------------------------------------------------------------------


class RedNeuronal:
    def __init__(self, capas):
        self.capas = capas

    def forward(self, entrada):
        """Propagación hacia adelante a través de todas las capas."""
        salida = entrada
        for capa in self.capas:
            salida = capa.forward(salida)
        return salida

    def backward(self, salida_deseada, salida_obtenida):
        """Propagación hacia atrás y cálculo de gradientes."""
        error = salida_obtenida - salida_deseada
        for i in reversed(range(len(self.capas))):
            capa_actual = self.capas[i]
            if i > 0:
                capa_anterior = self.capas[i - 1]
                error, pesos_anteriores = capa_actual.backward(
                    error,
                    (
                        self.capas[i + 1].pesos
                        if i < len(self.capas) - 1
                        else np.eye(capa_actual.num_neuronas)
                    ),
                )  # Identidad si es la última capa
            else:
                capa_actual.backward(
                    error, np.eye(capa_actual.num_neuronas)
                )  # No hay capa siguiente
            capa_actual.actualizar_parametros(self.tasa_aprendizaje)

    def entrenar(
        self, X_entrenamiento, y_entrenamiento_one_hot, epocas, tasa_aprendizaje
    ):
        """Entrenamiento de la red neuronal."""
        self.tasa_aprendizaje = tasa_aprendizaje
        num_ejemplos = X_entrenamiento.shape[0]
        for epoca in range(epocas):
            salidas = self.forward(X_entrenamiento)
            self.backward(y_entrenamiento_one_hot, salidas)
            perdida = self.calcular_perdida(y_entrenamiento_one_hot, salidas)
            if epoca % 10 == 0:
                print(f"Época {epoca}, Pérdida: {perdida:.4f}")

    def predecir(self, X):
        """Realiza predicciones."""
        salidas = self.forward(X)
        return np.argmax(salidas, axis=1)

    def calcular_perdida(self, y_verdadero, y_predicho):
        """Calcula la pérdida de entropía cruzada."""
        num_ejemplos = y_verdadero.shape[0]
        # Evitar logaritmo de cero
        y_predicho_clip = np.clip(y_predicho, 1e-15, 1 - 1e-15)
        perdida = -np.sum(y_verdadero * np.log(y_predicho_clip)) / num_ejemplos
        return perdida


# ------------------------------------------------------------------------------
# 4. Preprocesamiento con Isomap
# ------------------------------------------------------------------------------


def aplicar_isomap(X, n_vecinos, n_componentes):
    """Aplica Isomap para reducir la dimensionalidad."""
    iso = Isomap(
        n_neighbors=n_vecinos, n_components=n_componentes, n_jobs=-1
    )  # Usar todos los núcleos
    X_reducido = iso.fit_transform(X)
    return X_reducido


def one_hot_encode(y):
    """Codifica las etiquetas en formato one-hot."""
    num_clases = len(np.unique(y))
    num_ejemplos = len(y)
    encoding = np.zeros((num_ejemplos, num_clases))
    encoding[np.arange(num_ejemplos), y] = 1
    return encoding


# ------------------------------------------------------------------------------
# 5. Ajuste de Hiperparámetros
# ------------------------------------------------------------------------------


def evaluar_modelo(
    X, y, hiperparametros_isomap, hiperparametros_red, epocas, tasa_aprendizaje
):
    """Evalúa un modelo con un conjunto de hiperparámetros."""
    n_vecinos = hiperparametros_isomap["n_vecinos"]
    n_componentes = hiperparametros_isomap["n_componentes"]
    num_neuronas_capa_oculta = hiperparametros_red["num_neuronas_capa_oculta"]

    X_reducido = aplicar_isomap(X, n_vecinos, n_componentes)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X_reducido, y, test_size=0.2, random_state=42
    )
    y_entrenamiento_one_hot = one_hot_encode(y_entrenamiento)
    y_prueba_one_hot = one_hot_encode(y_prueba)

    num_caracteristicas_reducidas = X_entrenamiento.shape[1]
    num_clases = len(np.unique(y))

    capas = [
        Capa(
            num_caracteristicas_reducidas, num_neuronas_capa_oculta, relu, relu_derivada
        ),
        Capa(
            num_neuronas_capa_oculta, num_clases, softmax, lambda x: 1
        ),  # Softmax no tiene derivada directa aquí, se maneja en el backward
    ]
    red = RedNeuronal(capas)
    red.entrenar(X_entrenamiento, y_entrenamiento_one_hot, epocas, tasa_aprendizaje)
    predicciones = red.predecir(X_prueba)
    precision = accuracy_score(y_prueba, predicciones)
    return (
        precision,
        red,
    )  # Devolvemos la red entrenada para usar los mejores hiperparámetros


def busqueda_hiperparametros(
    X, y, rango_isomap, rango_red, epocas, tasa_aprendizaje, num_folds=3
):
    """Realiza una búsqueda simple de hiperparámetros usando validación cruzada."""
    mejor_precision = 0.0
    mejores_hiperparametros_isomap = None
    mejores_hiperparametros_red = None

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for n_vecinos in rango_isomap["n_vecinos"]:
        for n_componentes in rango_isomap["n_componentes"]:
            for num_neuronas_capa_oculta in rango_red["num_neuronas_capa_oculta"]:
                precisiones_fold = []
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    precision, _ = evaluar_modelo(
                        X_train,
                        y_train,
                        {"n_vecinos": n_vecinos, "n_componentes": n_componentes},
                        {"num_neuronas_capa_oculta": num_neuronas_capa_oculta},
                        epocas,
                        tasa_aprendizaje,
                    )
                    precisiones_fold.append(precision)

                precision_media = np.mean(precisiones_fold)
                print(
                    f"Isomap(vecinos={n_vecinos}, comp={n_componentes}), Red(neuronas={num_neuronas_capa_oculta}), Precisión CV: {precision_media:.4f}"
                )

                if precision_media > mejor_precision:
                    mejor_precision = precision_media
                    mejores_hiperparametros_isomap = {
                        "n_vecinos": n_vecinos,
                        "n_componentes": n_componentes,
                    }
                    mejores_hiperparametros_red = {
                        "num_neuronas_capa_oculta": num_neuronas_capa_oculta
                    }

    print("\nMejores Hiperparámetros Encontrados:")
    print(f"Isomap: {mejores_hiperparametros_isomap}")
    print(f"Red Neuronal: {mejores_hiperparametros_red}")
    return mejores_hiperparametros_isomap, mejores_hiperparametros_red


# ------------------------------------------------------------------------------
# 6. Entrenamiento Final y Evaluación
# ------------------------------------------------------------------------------


def entrenar_evaluar_final(
    X_entrenamiento_prep,
    y_entrenamiento,
    X_prueba_prep,
    y_prueba,
    mejores_hiperparametros_isomap,
    mejores_hiperparametros_red,
    epocas_final,
    tasa_aprendizaje_final,
):
    """Entrena el modelo final con los mejores hiperparámetros y lo evalúa."""
    X_entrenamiento_reducido = aplicar_isomap(
        X_entrenamiento_prep,
        mejores_hiperparametros_isomap["n_vecinos"],
        mejores_hiperparametros_isomap["n_componentes"],
    )
    X_prueba_reducido = aplicar_isomap(
        X_prueba_prep,
        mejores_hiperparametros_isomap["n_vecinos"],
        mejores_hiperparametros_isomap["n_componentes"],
    )

    y_entrenamiento_one_hot = one_hot_encode(y_entrenamiento)
    y_prueba_one_hot = one_hot_encode(y_prueba)

    num_caracteristicas_reducidas = X_entrenamiento_reducido.shape[1]
    num_clases = len(np.unique(y_entrenamiento))

    capas_final = [
        Capa(
            num_caracteristicas_reducidas,
            mejores_hiperparametros_red["num_neuronas_capa_oculta"],
            relu,
            relu_derivada,
        ),
        Capa(
            mejores_hiperparametros_red["num_neuronas_capa_oculta"],
            num_clases,
            softmax,
            lambda x: 1,
        ),
    ]
    red_final = RedNeuronal(capas_final)
    red_final.entrenar(
        X_entrenamiento_reducido,
        y_entrenamiento_one_hot,
        epocas_final,
        tasa_aprendizaje_final,
    )
    predicciones_prueba = red_final.predecir(X_prueba_reducido)
    precision_prueba = accuracy_score(y_prueba, predicciones_prueba)
    print(
        f"\nPrecisión del modelo final en el conjunto de prueba: {precision_prueba:.4f}"
    )

    return red_final


# ------------------------------------------------------------------------------
# 7. Carga y Preprocesamiento de Datos (Simulado)
# ------------------------------------------------------------------------------


def cargar_y_preprocesar_datos():
    """Simula la carga y preprocesamiento de datos."""
    # Aquí iría tu rutina real de carga y preprocesamiento
    # Por ejemplo, leer imágenes, normalizar píxeles, etc.
    np.random.seed(42)
    num_ejemplos = 100
    dimension_original = 50
    num_clases = 3
    X = np.random.rand(num_ejemplos, dimension_original)
    y = np.random.randint(0, num_clases, num_ejemplos)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_entrenamiento, y_entrenamiento, X_prueba, y_prueba


# ------------------------------------------------------------------------------
# 8. Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Cargar y preprocesar los datos
    X_entrenamiento_prep, y_entrenamiento, X_prueba_prep, y_prueba = (
        cargar_y_preprocesar_datos()
    )

    # Definir rangos de hiperparámetros para la búsqueda
    rango_isomap = {"n_vecinos": [5, 10, 15], "n_componentes": [10, 20, 30]}
    rango_red = {"num_neuronas_capa_oculta": [16, 32]}
    epocas_busqueda = 20
    tasa_aprendizaje_busqueda = 0.01

    # Buscar los mejores hiperparámetros usando validación cruzada
    mejores_isomap, mejores_red = busqueda_hiperparametros(
        X_entrenamiento_prep,
        y_entrenamiento,
        rango_isomap,
        rango_red,
        epocas_busqueda,
        tasa_aprendizaje_busqueda,
    )

    # Entrenar y evaluar el modelo final con los mejores hiperparámetros
    epocas_final = 50
    tasa_aprendizaje_final = 0.01
    modelo_final = entrenar_evaluar_final(
        X_entrenamiento_prep,
        y_entrenamiento,
        X_prueba_prep,
        y_prueba,
        mejores_isomap,
        mejores_red,
        epocas_final,
        tasa_aprendizaje_final,
    )

    # Puedes guardar el modelo_final si lo deseas
    # Ejemplo: np.save('modelo_pesos.npy', [capa.pesos for capa in modelo_final.capas])
    # Ejemplo: np.save('modelo_bias.npy', [capa.bias for capa in modelo_final.capas])
