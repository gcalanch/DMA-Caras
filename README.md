# DMA-Caras

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
