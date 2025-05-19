import itertools


def grid_search(
    x_train, y_train, x_test, y_test, layer_options, alpha_options, epochs=1000
):
    """
    Realiza una búsqueda en cuadrícula para encontrar los mejores hiperparámetros.

    Args:
        x_train (ndarray): Datos de entrenamiento.
        y_train (ndarray): Etiquetas de entrenamiento.
        x_test (ndarray): Datos de prueba.
        y_test (ndarray): Etiquetas de prueba.
        layer_options (list): Lista de configuraciones de capas.
        alpha_options (list): Lista de tasas de aprendizaje.
        epochs (int): Número de épocas para entrenar.

    Returns:
        dict: Mejor configuración de hiperparámetros y su rendimiento.
    """
    best_config = None
    best_error_rate = float("inf")

    # Generar todas las combinaciones de hiperparámetros
    for layers, alpha in itertools.product(layer_options, alpha_options):
        print(f"Probando configuración: layers={layers}, alpha={alpha}")
        nn = NeuralNetwork(layers, alpha)
        nn.backward(x_train, y_train, epochs)

        # Evaluar en el conjunto de prueba
        predictions = (nn.forward(x_test) > 0.5).astype(int)
        error_rate = 1 - np.mean(predictions.flatten() == y_test)

        print(f"Error rate: {error_rate:.4f}")
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_config = {"layers": layers, "alpha": alpha}

    return {"best_config": best_config, "best_error_rate": best_error_rate}


# Definir opciones de hiperparámetros
layer_options = [
    [4096, 128, 64, 1],  # Configuración original
    [4096, 256, 128, 1],  # Más neuronas
    [4096, 64, 32, 1],  # Menos neuronas
]
alpha_options = [0.01, 0.1, 0.5]  # Diferentes tasas de aprendizaje

# Realizar búsqueda en cuadrícula
result = grid_search(
    x_train, y_train, x_test, y_test, layer_options, alpha_options, epochs=1000
)

print(f"Mejor configuración: {result['best_config']}")
print(f"Menor error rate: {result['best_error_rate']:.4f}")
