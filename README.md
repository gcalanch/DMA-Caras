# DMA-Caras

El objetivo es entrenar un sistema de reconocimiento facial, donde el dataset son imagenes de las caras de alumnos de la cohorte 2024-2025. Luego de capturar las fotos iniciales de las caras, a los archivos se le realizó un preprocesamiento para mejorar el resultado del sistema. En preprosesamiento se recortan las fotos originales para eliminar elementos innecesarios, se balancea la iluminación para hacer los datos más consistentes.
 
Luego se aplicó reducción de dimensionalidad usando ISOMAP de modo que el modelo trabajara con datos más manejables, y se realizó un clustering para viualizar cómo se agrupaban las caras. En el clustering se observó que los grupos no estaban bien difereciados entre sí y luego al probar en diferentes entrenamientos de la red neuronal, se obtenian reausltados con grandes errores.
 
Para tener una mayor definicion entre los grupos de fotos, se desicidió aumentar la cantidad de datos con data augmentation, aplicando rotaciones, cambios de escala y variaciones de brillo para generar más ejemplos. Con los datos ampliados, se repitio el preprocesamiento, la reducción de dimensionalidad y el clustering. Esta vez, los grupos se distribuyeron de una manera más definida y el modelo mostró resultados con menos errores.
 
La cantidad y calidad de los datos hicieron diferencia en el rendimiento del modelo



La aplicacion caras esta dividida en dos partes: app (entrenamiento) y prod (produccion).

Dentro de *caras/app* estan los notebooks de entrenamiento del modelo. Se pueden ejecutar desde main.ipynb o en forma individual en el orden indicado:
notebooks = [
    "deteccion.ipynb",
    "aumentacion.ipynb",
    "deteccionaumentada.ipynb",
    "preprocesamiento.ipynb",
    "separacion.ipynb",
    "isomap.ipynb"
    ]

Dentro de *caras/prod* estan los notebooks de produccion, para la deteccion. Se pueden ejecutar desde produccion.ipynb o en forma individual en el orden indicado:

notebooks = [
  "deteccionprod.ipynb",
  "procesamientoprod.ipynb"
  ]

La estructura del proyecto es: 

├── caras
│   ├── app
│   │   ├── aumentacion.ipynb
│   │   ├── deteccionaumentadas.ipynb
│   │   ├── deteccion.ipynb
│   │   ├── main.ipynb
│   │   └── procesamiento.ipynb
│   └── prod
│       ├── deteccionprod.ipynb
│       ├── procesamientoprod.ipynb
│       └── produccion.ipynb
├── README.md





