#
# Version de Claude
#

# Importación de bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
import pickle
import time


# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ------------------------------------
# Suponiendo que ya tienes una función de preprocesamiento
def cargar_datos():
    """
    Carga las imágenes de entrenamiento y prueba.
    Retorna: X_train, y_train, X_test, y_test
    """
    # Incluir aquí tu código de preprocesamiento existente
    # Por ejemplo:
    # X_train, y_train = cargar_imagenes_entrenamiento()
    # X_test, y_test = cargar_imagenes_prueba()

    # Simulamos datos para el ejemplo
    X = np.random.rand(100, 2500)  # 100 imágenes de 50x50 píxeles
    y = np.random.randint(0, 10, 100)  # 10 identidades diferentes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_test, y_test


# 2. REDUCCIÓN DE DIMENSIONALIDAD CON ISOMAP
# -----------------------------------------
class ReduccionDimensionalidad:
    def __init__(self, n_componentes=50, n_vecinos=5):
        """
        Inicializa el reductor de dimensionalidad con Isomap

        Parámetros:
        - n_componentes: Número de dimensiones a reducir
        - n_vecinos: Número de vecinos para Isomap
        """
        self.n_componentes = n_componentes
        self.n_vecinos = n_vecinos
        self.isomap = Isomap(n_components=n_componentes, n_neighbors=n_vecinos)

    def ajustar(self, X):
        """Ajusta el modelo Isomap con los datos de entrada"""
        self.isomap.fit(X)

    def transformar(self, X):
        """Transforma los datos de entrada utilizando Isomap"""
        return self.isomap.transform(X)

    def ajustar_transformar(self, X):
        """Ajusta y transforma los datos en un solo paso"""
        return self.isomap.fit_transform(X)

    def guardar_modelo(self, ruta):
        """Guarda el modelo Isomap"""
        with open(ruta, "wb") as f:
            pickle.dump(self.isomap, f)

    def cargar_modelo(self, ruta):
        """Carga un modelo Isomap previamente guardado"""
        with open(ruta, "rb") as f:
            self.isomap = pickle.load(f)


# 3. RED NEURONAL DESDE CERO
# --------------------------
class RedNeuronalReconocimientoFacial:
    def __init__(self, tam_entrada, capas_ocultas, tam_salida):
        """
        Inicializa la red neuronal con arquitectura especificada

        Parámetros:
        - tam_entrada: Tamaño de la capa de entrada (después de Isomap)
        - capas_ocultas: Lista con el tamaño de cada capa oculta
        - tam_salida: Número de clases (personas) a identificar
        """
        self.tam_entrada = tam_entrada
        self.capas_ocultas = capas_ocultas
        self.tam_salida = tam_salida

        # Inicialización de pesos y sesgos
        self.pesos = []
        self.sesgos = []

        # Pesos entre capa de entrada y primera capa oculta
        self.pesos.append(np.random.randn(tam_entrada, capas_ocultas[0]) * 0.01)
        self.sesgos.append(np.zeros((1, capas_ocultas[0])))

        # Pesos entre capas ocultas
        for i in range(1, len(capas_ocultas)):
            self.pesos.append(
                np.random.randn(capas_ocultas[i - 1], capas_ocultas[i]) * 0.01
            )
            self.sesgos.append(np.zeros((1, capas_ocultas[i])))

        # Pesos entre última capa oculta y capa de salida
        self.pesos.append(np.random.randn(capas_ocultas[-1], tam_salida) * 0.01)
        self.sesgos.append(np.zeros((1, tam_salida)))

    def relu(self, Z):
        """Función de activación ReLU"""
        return np.maximum(0, Z)

    def derivada_relu(self, Z):
        """Derivada de la función ReLU"""
        return Z > 0

    def softmax(self, Z):
        """Función de activación Softmax para normalización de salida"""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def propagacion_adelante(self, X):
        """
        Realiza la propagación hacia adelante a través de la red

        Parámetros:
        - X: Datos de entrada [batch_size, tam_entrada]

        Retorna:
        - Activaciones de cada capa y las salidas previas a la activación
        """
        activaciones = [X]
        z_values = []

        # Propagación a través de capas ocultas con ReLU
        A = X
        for i in range(len(self.capas_ocultas)):
            Z = np.dot(A, self.pesos[i]) + self.sesgos[i]
            z_values.append(Z)
            A = self.relu(Z)
            activaciones.append(A)

        # Capa de salida con Softmax
        Z_out = np.dot(A, self.pesos[-1]) + self.sesgos[-1]
        z_values.append(Z_out)
        A_out = self.softmax(Z_out)
        activaciones.append(A_out)

        return activaciones, z_values

    def calcular_costo(self, y_pred, y_true):
        """
        Calcula la entropía cruzada como función de costo

        Parámetros:
        - y_pred: Predicciones de la red [batch_size, tam_salida]
        - y_true: Etiquetas reales en formato one-hot [batch_size, tam_salida]

        Retorna:
        - Costo promedio
        """
        m = y_true.shape[0]
        # Convertir etiquetas enteras a one-hot si es necesario
        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros((m, self.tam_salida))
            y_true_one_hot[np.arange(m), y_true] = 1
            y_true = y_true_one_hot

        # Evitar log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        costo = -np.sum(y_true * np.log(y_pred)) / m
        return costo

    def propagacion_atras(self, X, y, learning_rate=0.01):
        """
        Realiza la propagación hacia atrás (backpropagation) para ajustar pesos

        Parámetros:
        - X: Datos de entrada [batch_size, tam_entrada]
        - y: Etiquetas reales [batch_size]
        - learning_rate: Tasa de aprendizaje

        Retorna:
        - Costo del batch
        """
        m = X.shape[0]
        y_one_hot = np.zeros((m, self.tam_salida))
        y_one_hot[np.arange(m), y] = 1

        # Propagación hacia adelante
        activaciones, z_values = self.propagacion_adelante(X)

        # Calcular costo
        costo = self.calcular_costo(activaciones[-1], y_one_hot)

        # Inicializar gradientes
        dZ = activaciones[-1] - y_one_hot

        # Backpropagation
        for capa in range(len(self.pesos) - 1, -1, -1):
            # Calcular gradientes para pesos y sesgos
            dW = np.dot(activaciones[capa].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            # Actualizar pesos y sesgos
            self.pesos[capa] -= learning_rate * dW
            self.sesgos[capa] -= learning_rate * db

            # Calcular dZ para la capa anterior (si no es la primera capa)
            if capa > 0:
                dA = np.dot(dZ, self.pesos[capa].T)
                dZ = dA * self.derivada_relu(z_values[capa - 1])

        return costo

    def entrenar(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        early_stopping=True,
        paciencia=10,
    ):
        """
        Entrena la red neuronal usando mini-batch gradient descent

        Parámetros:
        - X: Datos de entrenamiento [n_samples, tam_entrada]
        - y: Etiquetas [n_samples]
        - X_val: Datos de validación (opcional)
        - y_val: Etiquetas de validación (opcional)
        - epochs: Número de épocas de entrenamiento
        - batch_size: Tamaño del mini-batch
        - learning_rate: Tasa de aprendizaje
        - early_stopping: Si se debe usar parada temprana
        - paciencia: Épocas de paciencia para parada temprana

        Retorna:
        - Historial de costos de entrenamiento y validación
        """
        n_samples = X.shape[0]
        historial_costos = []
        historial_costos_val = []

        mejor_costo_val = float("inf")
        contador_paciencia = 0

        for epoca in range(epochs):
            # Mezclar datos
            indices = np.random.permutation(n_samples)
            X_mezclado = X[indices]
            y_mezclado = y[indices]

            # Entrenamiento por mini-batches
            costo_epoca = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_mezclado[i : i + batch_size]
                y_batch = y_mezclado[i : i + batch_size]

                costo_batch = self.propagacion_atras(X_batch, y_batch, learning_rate)
                costo_epoca += costo_batch * (len(X_batch) / n_samples)

            historial_costos.append(costo_epoca)

            # Validación
            if X_val is not None and y_val is not None:
                pred_val, _ = self.propagacion_adelante(X_val)
                y_val_pred = pred_val[-1]
                costo_val = self.calcular_costo(y_val_pred, y_val)
                historial_costos_val.append(costo_val)

                # Early stopping
                if early_stopping:
                    if costo_val < mejor_costo_val:
                        mejor_costo_val = costo_val
                        contador_paciencia = 0
                    else:
                        contador_paciencia += 1

                    if contador_paciencia >= paciencia:
                        print(f"Parada temprana en época {epoca+1}")
                        break

                print(
                    f"Época {epoca+1}/{epochs} - Costo: {costo_epoca:.4f} - Costo Val: {costo_val:.4f}"
                )
            else:
                print(f"Época {epoca+1}/{epochs} - Costo: {costo_epoca:.4f}")

        return historial_costos, historial_costos_val

    def predecir(self, X):
        """
        Realiza predicciones para los datos de entrada

        Parámetros:
        - X: Datos de entrada [n_samples, tam_entrada]

        Retorna:
        - Predicciones de clase [n_samples]
        """
        activaciones, _ = self.propagacion_adelante(X)
        return np.argmax(activaciones[-1], axis=1)

    def evaluar(self, X, y):
        """
        Evalúa el rendimiento del modelo

        Parámetros:
        - X: Datos de evaluación [n_samples, tam_entrada]
        - y: Etiquetas reales [n_samples]

        Retorna:
        - Precisión del modelo
        """
        pred = self.predecir(X)
        precision = np.mean(pred == y)
        return precision

    def guardar_modelo(self, ruta):
        """Guarda los parámetros del modelo"""
        modelo = {
            "pesos": self.pesos,
            "sesgos": self.sesgos,
            "arquitectura": {
                "tam_entrada": self.tam_entrada,
                "capas_ocultas": self.capas_ocultas,
                "tam_salida": self.tam_salida,
            },
        }
        with open(ruta, "wb") as f:
            pickle.dump(modelo, f)

    def cargar_modelo(self, ruta):
        """Carga los parámetros del modelo"""
        with open(ruta, "rb") as f:
            modelo = pickle.load(f)

        self.tam_entrada = modelo["arquitectura"]["tam_entrada"]
        self.capas_ocultas = modelo["arquitectura"]["capas_ocultas"]
        self.tam_salida = modelo["arquitectura"]["tam_salida"]
        self.pesos = modelo["pesos"]
        self.sesgos = modelo["sesgos"]


# 4. OPTIMIZACIÓN DE HIPERPARÁMETROS
# ---------------------------------
def buscar_hiperparametros(X_train, y_train, X_val, y_val):
    """
    Realiza una búsqueda de hiperparámetros para Isomap y la red neuronal

    Parámetros:
    - X_train, y_train: Datos de entrenamiento
    - X_val, y_val: Datos de validación

    Retorna:
    - Mejores hiperparámetros encontrados
    """
    mejores_params = {
        "isomap_componentes": 0,
        "isomap_vecinos": 0,
        "capas_ocultas": [],
        "learning_rate": 0,
        "mejor_precision": 0,
    }

    # Parámetros a probar
    componentes_isomap = [30, 50, 70]
    vecinos_isomap = [5, 10, 15]
    arquitecturas = [[128, 64], [256, 128, 64], [512, 256, 128]]
    learning_rates = [0.01, 0.001, 0.0001]

    for n_comp in componentes_isomap:
        for n_vec in vecinos_isomap:
            print(f"\nProbando Isomap con {n_comp} componentes y {n_vec} vecinos")

            # Aplicar Isomap
            reductor = ReduccionDimensionalidad(n_componentes=n_comp, n_vecinos=n_vec)
            X_train_reducido = reductor.ajustar_transformar(X_train)
            X_val_reducido = reductor.transformar(X_val)

            # Número de clases únicas
            n_clases = len(np.unique(y_train))

            for arquitectura in arquitecturas:
                for lr in learning_rates:
                    print(f"Probando arquitectura {arquitectura} con lr={lr}")

                    # Crear y entrenar modelo
                    modelo = RedNeuronalReconocimientoFacial(
                        tam_entrada=n_comp,
                        capas_ocultas=arquitectura,
                        tam_salida=n_clases,
                    )

                    # Entrenamiento corto para optimización
                    modelo.entrenar(
                        X_train_reducido,
                        y_train,
                        X_val=X_val_reducido,
                        y_val=y_val,
                        epochs=30,
                        batch_size=32,
                        learning_rate=lr,
                        early_stopping=True,
                        paciencia=5,
                    )

                    # Evaluar modelo
                    precision = modelo.evaluar(X_val_reducido, y_val)
                    print(f"Precisión: {precision:.4f}")

                    # Actualizar mejores parámetros
                    if precision > mejores_params["mejor_precision"]:
                        mejores_params = {
                            "isomap_componentes": n_comp,
                            "isomap_vecinos": n_vec,
                            "capas_ocultas": arquitectura,
                            "learning_rate": lr,
                            "mejor_precision": precision,
                        }

    print("\nMejores hiperparámetros encontrados:")
    print(f"Componentes Isomap: {mejores_params['isomap_componentes']}")
    print(f"Vecinos Isomap: {mejores_params['isomap_vecinos']}")
    print(f"Arquitectura: {mejores_params['capas_ocultas']}")
    print(f"Learning rate: {mejores_params['learning_rate']}")
    print(f"Precisión: {mejores_params['mejor_precision']:.4f}")

    return mejores_params


# 5. VISUALIZACIÓN Y EVALUACIÓN
# ----------------------------
def visualizar_resultados(
    historial_costos, historial_costos_val=None, X_reducido=None, y=None
):
    """
    Visualiza los resultados del entrenamiento y los datos reducidos con Isomap

    Parámetros:
    - historial_costos: Historial de costos de entrenamiento
    - historial_costos_val: Historial de costos de validación (opcional)
    - X_reducido: Datos reducidos con Isomap (opcional para visualización)
    - y: Etiquetas correspondientes a X_reducido (opcional para visualización)
    """
    plt.figure(figsize=(12, 5))

    # Gráfica de costos
    plt.subplot(1, 2, 1)
    plt.plot(historial_costos, label="Entrenamiento")
    if historial_costos_val:
        plt.plot(historial_costos_val, label="Validación")
    plt.title("Evolución del Costo")
    plt.xlabel("Época")
    plt.ylabel("Costo")
    plt.legend()

    # Visualización de datos reducidos (si hay 2 o 3 componentes)
    if X_reducido is not None and y is not None:
        plt.subplot(1, 2, 2)
        if X_reducido.shape[1] == 2:
            # Para 2 componentes
            for clase in np.unique(y):
                idx = y == clase
                plt.scatter(
                    X_reducido[idx, 0], X_reducido[idx, 1], label=f"Clase {clase}"
                )
            plt.title("Visualización de Datos Reducidos (2D)")
            plt.legend()
        elif X_reducido.shape[1] >= 3:
            # Para 3 componentes
            from mpl_toolkits.mplot3d import Axes3D

            ax = plt.subplot(1, 2, 2, projection="3d")
            for clase in np.unique(y):
                idx = y == clase
                ax.scatter(
                    X_reducido[idx, 0],
                    X_reducido[idx, 1],
                    X_reducido[idx, 2],
                    label=f"Clase {clase}",
                )
            ax.set_title("Visualización de Datos Reducidos (3D)")
            plt.legend()

    plt.tight_layout()
    plt.show()


def matriz_confusion(y_true, y_pred, etiquetas=None):
    """
    Calcula y muestra la matriz de confusión

    Parámetros:
    - y_true: Etiquetas reales
    - y_pred: Predicciones del modelo
    - etiquetas: Nombres de las clases (opcional)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=etiquetas,
        yticklabels=etiquetas,
    )
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()


# 6. FUNCIÓN PRINCIPAL
# -------------------
def main():
    """Función principal para ejecutar todo el proceso"""
    # Cargar datos
    print("Cargando datos...")
    X_train, y_train, X_test, y_test = cargar_datos()

    # Dividir en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print(
        f"Datos cargados: {X_train.shape[0]} entrenamiento, {X_val.shape[0]} validación, {X_test.shape[0]} prueba"
    )

    # Optimización de hiperparámetros
    print("\nIniciando búsqueda de hiperparámetros...")
    start_time = time.time()
    mejores_params = buscar_hiperparametros(X_train, y_train, X_val, y_val)
    print(f"Tiempo de búsqueda: {(time.time() - start_time) / 60:.2f} minutos")

    # Aplicar Isomap con los mejores parámetros
    print("\nAplicando Isomap con los mejores parámetros...")
    reductor = ReduccionDimensionalidad(
        n_componentes=mejores_params["isomap_componentes"],
        n_vecinos=mejores_params["isomap_vecinos"],
    )
    X_train_reducido = reductor.ajustar_transformar(X_train)
    X_val_reducido = reductor.transformar(X_val)
    X_test_reducido = reductor.transformar(X_test)

    # Crear modelo con los mejores parámetros
    print("\nCreando modelo con los mejores parámetros...")
    n_clases = len(np.unique(y_train))
    modelo = RedNeuronalReconocimientoFacial(
        tam_entrada=mejores_params["isomap_componentes"],
        capas_ocultas=mejores_params["capas_ocultas"],
        tam_salida=n_clases,
    )

    # Entrenar modelo
    print("\nEntrenando modelo...")
    start_time = time.time()
    historial_costos, historial_costos_val = modelo.entrenar(
        X_train_reducido,
        y_train,
        X_val=X_val_reducido,
        y_val=y_val,
        epochs=100,
        batch_size=32,
        learning_rate=mejores_params["learning_rate"],
        early_stopping=True,
        paciencia=10,
    )
    print(f"Tiempo de entrenamiento: {(time.time() - start_time) / 60:.2f} minutos")

    # Evaluar en conjunto de prueba
    precision_test = modelo.evaluar(X_test_reducido, y_test)
    print(f"\nPrecisión en conjunto de prueba: {precision_test:.4f}")

    # Visualizar resultados
    print("\nVisualizando resultados...")
    visualizar_resultados(
        historial_costos, historial_costos_val, X_reducido=X_test_reducido, y=y_test
    )

    # Matriz de confusión
    y_pred = modelo.predecir(X_test_reducido)
    matriz_confusion(y_test, y_pred)

    # Guardar modelos
    print("\nGuardando modelos...")
    reductor.guardar_modelo("isomap_modelo.pkl")
    modelo.guardar_modelo("red_neuronal_modelo.pkl")
    print("Modelos guardados correctamente.")


if __name__ == "__main__":
    main()
