# DMA-Caras

El objetivo es entrenar un sistema de reconocimiento facial, donde el dataset son imagenes de las caras de alumnos de la cohorte 2024-2025 MCD Austral. Luego de capturar las fotos iniciales de los alumnos,  se realizó una procesamiento de las imagenes para deteccion de rostros con tecnologia DNN (Deep Neural Network) con un modelo Basado en SSD + ResNet. A partir de imágenes a color, se convierte a gris después del recorte, redimensionado a (64x64) y se ajustan coordenadas para ampliar el rostro detectado.

Luego se aplicó reducción de dimensionalidad usando ISOMAP de modo que el modelo trabajara con datos más manejables, y se realizó un clustering para viualizar cómo se agrupaban las caras. En el clustering se observó que los grupos no estaban bien difereciados entre sí y luego al probar en diferentes entrenamientos de la red neuronal, se obtenian reausltados con grandes errores.
 
Para tener una mayor definicion entre los grupos de fotos, se desicidió aumentar la cantidad de datos con data augmentation, aplicando rotaciones, cambios de escala y variaciones de brillo para generar más ejemplos. Con los datos ampliados, se repitio el preprocesamiento, la reducción de dimensionalidad y el clustering. Esta vez, los grupos se distribuyeron de una manera más definida y el modelo mostró resultados con menos errores.

La aplicacion caras esta dividida en 4 partes: 
01 - Deteccion y procesamiento: Deteccion de rostros, recorte y normalizacion de los datos. 
02 - Reduccion y Division: Reuduccion de dimensionalidad, aumentacion de datos y division de los subconjuntos de training y Test 
03 - Entrenamiento: Entrenamiento de la red neuronal. 
04 - Clasificacion: Etapa que realiza la Clasificacion de nuevas imágenes utilizando los modelos generados en la etapa anterior. 






