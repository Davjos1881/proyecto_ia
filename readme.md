# Priorización de siniestros viales en Cali: modelo final, análisis territorial y demo

# Integrantres

- Jose David Santa

- Brayan Stiven Tigreros

- Felipe Mosquera Murillo

Tercera entrega del proyecto de inteligencia artificial. Reúne el modelo final de clasificación de severidad (una red neuronal MLP), la regresión del conteo diario, el perfilamiento territorial de cruces con PCA y K-medias, y una demo en Streamlit que muestra los resultados en una interfaz sencilla.

## Contexto

El proyecto usa el historial de siniestros de tránsito de Cali (2016 a noviembre de 2024) para apoyar dos decisiones: a qué siniestro atender primero y en qué cruces concentrar recursos fijos. El análisis exploratorio y la comparación de modelos de clasificación se documentaron en entregas anteriores. Esta entrega parte de sus conclusiones.

La idea inicial era predecir cuándo ocurrirían más siniestros para repartir recursos según el calendario. El dataset no tiene hora del día y las pruebas mostraron que el calendario no mejora la predicción, así que el proyecto se organizó alrededor de preguntas que sí se pueden responder con los datos.

| Componente | Pregunta | Técnicas |
|---|---|---|
| Clasificación de severidad | ¿Qué tan grave es un siniestro según los vehículos involucrados? | Red neuronal MLP (Keras) |
| Notebook 02 | ¿El calendario permite predecir cuántos siniestros graves habrá por día? | Regresión lineal y SGDRegressor |
| Notebook 03 | ¿Hay cruces donde los siniestros graves se concentran y se repiten con los años? | Ranking de cruces, PCA y K-medias |
| Demo | Reúne los resultados anteriores en una interfaz | Streamlit |

## Estructura del repositorio

```
modelos/
├── MLP_Clasificacion_severidad_de_siniestros.ipynb
├── 02_regresion_conteo_diario.ipynb
├── 03_perfilamiento_territorial.ipynb
└── demo_siniestros/
    ├── app.py
    ├── requirements.txt
    ├── .streamlit/config.toml
    ├── modelo/
    │   ├── mlp_vehiculos.joblib
    │   └── columnas.json
    └── datos/
        ├── tasa_combinaciones.csv
        ├── cruces_perfil.csv
        └── comparacion_modelos.csv   (opcional)
```

## Datos

- Se parte de 371.003 registros. El 66% estaba marcado como "Negativo" (no confirmado como accidente) y se excluyó, por lo que quedaron 124.889 siniestros confirmados, ya limpios en el archivo `siniestralidad_clean.csv`.
- Variable objetivo binaria: grave si hubo lesionados o fallecidos (unos 50.860 casos) y no grave si fueron solo daños.
- Las direcciones vienen como cruces (por ejemplo, "CALLE 70 CON CARRERA 8"). Se normalizaron y se trató como el mismo lugar una dirección con las calles en distinto orden.
- Se excluyeron de los modelos las variables que se conocen después del evento (como fallecido foráneo y número de lesionados), el código del registro y el texto de la dirección, para evitar fuga de información.

Limitaciones de los datos que condicionaron el diseño:

- No hay hora del día ni coordenadas.
- Solo cerca del 37% de los siniestros de 2016 a 2022 tiene dirección válida, y ninguno entre 2018 y 2020.
- El 18,6% de los siniestros no tiene dato de tipo de vehículo. Se trata como "sin vehículo", que no es lo mismo que "no sabemos".
- Los siniestros de solo daños casi desaparecen del registro desde 2019, mientras los graves se mantienen estables desde 2020. Por eso el porcentaje de graves cambia mucho entre años.

## Clasificación de severidad: MLP en Keras

Notebook `MLP_Clasificacion_severidad_de_siniestros.ipynb`.

### Datos y variables

- Se eliminaron 5.404 registros sin dato de número de vehículos, por lo que el modelo usa 119.485 registros.
- 21 predictores: año, mes (codificado con seno y coseno), día de la semana (variables dummy), fin de semana, festivo, siete indicadores de tipo de vehículo (moto, motocarro, automóvil, camioneta o campero, bus, carga pesada y bicicleta), número de vehículos implicados y una bandera de "sin dato de vehículos".
- Los vacíos en los indicadores de vehículo se imputan como 0, y la bandera permite distinguirlos de un "sin vehículo" real.
- Los predictores se estandarizan con StandardScaler, ajustado solo con el conjunto de entrenamiento.

### Evaluación

Separación temporal fuera de muestra: entrenamiento con 2016 a 2021 (101.188 registros, 37,6% graves) y test con 2022 completo (9.817 registros, 44,9% graves).

### Arquitectura

| Capa | Neuronas | Activación |
|---|---|---|
| Densa 1 | 32 | ReLU |
| Densa 2 | 16 | ReLU |
| Salida | 2 | Softmax |

En total son 1.266 parámetros. Se compila con el optimizador Adam y pérdida de entropía cruzada categórica, y se entrena durante 30 épocas con lotes de 256. Se probó Dropout como regularización y se dejó como opción, porque con la cantidad de datos disponible el modelo no muestra un sobreajuste fuerte.

### Resultados

| Conjunto | Exactitud | Precisión (graves) | Recall (graves) | F1 (graves) |
|---|---|---|---|---|
| Entrenamiento 2016-2021 | 0,78 | 0,75 | 0,61 | 0,67 |
| Test 2022 | 0,76 | 0,70 | 0,80 | 0,75 |

Referencias en el test: predecir siempre "solo daños" daría una exactitud de 0,551, y predecir siempre "grave" daría un F1 de 0,62 para esa clase.

- El modelo supera con claridad las dos referencias simples, tanto en exactitud como en F1 de la clase grave.
- El recall de 0,80 en el test significa que detecta unos 8 de cada 10 siniestros graves. Es la métrica que más importa para un caso de emergencias, porque es preferible una falsa alarma a no detectar un siniestro grave. La precisión de 0,70 indica que cerca de 3 de cada 10 alertas serían falsas alarmas.
- Entre entrenamiento y test la exactitud baja apenas de 0,78 a 0,76. El recall y el F1 de graves suben en el test en parte porque la proporción de graves pasó de 37,6% a 44,9%, por lo que no deben leerse como una mejora real del modelo.
- Las curvas de pérdida y exactitud se estabilizan con las épocas.

## Notebook 02: regresión del conteo diario

Se evaluó si el calendario (día de la semana, mes, fin de semana y festivo) permite predecir los siniestros por día, con regresión lineal y SGDRegressor, frente al promedio del entrenamiento y al promedio móvil de 28 días.

- Con el total diario la prueba no es válida. El promedio pasa de cerca de 48 siniestros por día (2016 a 2021) a cerca de 27 en 2022 por el cambio en el registro, y ese salto domina el error.
- La prueba válida usa solo siniestros graves por día, sin el año como variable, con entrenamiento 2020 a 2022 y test 2023.

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Promedio del entrenamiento | 2,950 | 3,718 | -0,019 |
| Promedio móvil de 28 días | 2,913 | 3,725 | -0,023 |
| Regresión lineal (calendario) | 3,154 | 4,001 | -0,180 |
| SGDRegressor (calendario) | 3,160 | 4,012 | -0,187 |

El calendario empeora la predicción frente al simple promedio. Los festivos tienen cerca de 20% menos graves y el mes muestra diferencias, pero el día de la semana no, y esos efectos son pequeños frente al ruido diario. Regresión lineal y SGDRegressor llegan prácticamente a la misma solución.

## Notebook 03: perfilamiento territorial

Se trabajó con los siniestros con dirección válida de 2016-2017 y 2021-2022: 43.348 siniestros en 14.969 cruces distintos.

**Concentración y persistencia.** El 1% de los cruces concentra el 21,8% de los siniestros. Para medir si ese patrón se mantiene, se eligió una lista de los cruces con más graves en un periodo y se midió qué parte de los graves de un periodo posterior cae en ella.

| Prueba | 50 cruces | 100 cruces | 200 cruces |
|---|---|---|---|
| Lista de 2016-2017, graves de 2021-2022 | 10,6% | 16,3% | 24,0% |
| Lista de 2016-2022, graves de 2023-2024 | 5,3% | 7,5% | 11,3% |

Más del 75% de los graves ocurre fuera de la lista, y la captura baja a la mitad en 2023-2024.

**Perfil de los cruces.** Para 199 cruces se calcularon el volumen de siniestros, el porcentaje de graves, el de motos y el de carga pesada. Los porcentajes de vehículos se calculan solo con los siniestros que tienen dato de vehículos. La bicicleta se excluyó del clustering porque casi no aparece.

- **PCA** con dos componentes, que explica el 74,1% de la varianza. El primero contrapone gravedad y motos a carga pesada, y el segundo corresponde casi por completo al volumen de siniestros.
- **K-medias** con k = 3, elegido por la mejor silueta (0,288). La estructura es débil y los grupos se usan como segmentación descriptiva. La estabilidad frente a la inicialización es moderada (ARI 0,70).

| Grupo | Cruces | Siniestros por cruce | % graves | % motos | % carga pesada |
|---|---|---|---|---|---|
| Muchos siniestros | 35 | 103,7 | 47,1% | 45,7% | 16,4% |
| Más graves y más motos | 74 | 39,7 | 54,1% | 51,9% | 11,1% |
| Menos graves, más carga pesada | 90 | 49,7 | 33,4% | 32,5% | 18,6% |

## Demo (Streamlit)

La carpeta `demo_siniestros` contiene una aplicación local con tres pestañas.

| Pestaña | Qué muestra |
|---|---|
| Atender un reporte | Se eligen los vehículos involucrados y la app marca la prioridad (alta, media o baja), junto con el porcentaje histórico de graves para esa combinación y las combinaciones más frecuentes del historial |
| Dónde poner recursos | Un control para elegir 50, 100 o 200 cruces con el porcentaje de graves posteriores que cubren, y la tabla de cruces con su perfil, filtrable por grupo, calle y mínimo de graves |
| Evidencia y límites | Comparación de modelos, resultado del calendario y limitaciones del proyecto |

Se ejecuta con `streamlit run app.py`.

### Modelo de Clasificacíon MLP

La pestaña de reportes usa un MLP de scikit-learn con dos capas ocultas de 64 y 32 neuronas, entrenado solo con los siete indicadores de vehículo y con datos hasta 2020. Alcanza un ROC-AUC de 0,796 en el test de 2022. Se entrenó solo con vehículos para que la interfaz pida únicamente lo que importa para la decisión. El modelo se guarda con joblib junto con el orden de sus columnas, y una tabla con la tasa histórica de graves por combinación de vehículos acompaña cada predicción.

### Decisiones de diseño

- **Nivel de prioridad en lugar de probabilidad.** El modelo se entrenó con 36% de graves y se probó con 45%, así que su puntaje no es una probabilidad calibrada. La prioridad se define frente al promedio histórico, siendo alta si el puntaje lo supera por 0,10 o más, baja si queda 0,10 o más por debajo, y media en el resto. El 0,10 es un valor elegido para la demo y se puede ajustar en el código.
- **Dato histórico visible.** Junto al puntaje se muestra el porcentaje real de graves para esa combinación de vehículos y la cantidad de casos que lo respaldan, con un aviso cuando son pocos.
- **Grupos de cruces con nombre automático.** Los grupos del perfilamiento se nombran según sus promedios y no según el número que K-medias les asignó, para que la tabla sea correcta aunque cambie la numeración.
- **Advertencias en pantalla.** La demo recuerda que supone que el tipo de vehículos se conoce al recibir el reporte, y que "más motos" en un cruce indica participación de motos y no quién resultó herido.

## Herramientas

- **Lenguaje y entorno:** Python, Jupyter Notebook y Google Colab.
- **Datos y análisis:** pandas, NumPy y SciPy.
- **Modelos y métricas:** Keras con TensorFlow para la red neuronal. Scikit-learn para las métricas, LinearRegression, SGDRegressor, PCA, KMeans y el MLP de la demo.
- **Visualización:** Matplotlib y Seaborn.
- **Demo:** Streamlit y joblib.