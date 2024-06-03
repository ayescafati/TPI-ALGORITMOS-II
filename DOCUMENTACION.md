# Proyecto: Librería de Modelos Predictivos con Árboles de Decisión y Bosques Aleatorios

## Enunciado del Problema

El objetivo de este proyecto es desarrollar una librería que permita crear modelos predictivos mediante una versión simplificada del ensamble RandomForest. Se busca entrenar un modelo con un conjunto de datos etiquetados, utilizando técnicas de aprendizaje supervisado, para predecir la variable objetivo. Inicialmente, nos enfocaremos en problemas de clasificación, con la posibilidad de extender la funcionalidad a problemas de regresión en el futuro.

## Introducción

Este proyecto implementa un clasificador basado en árboles de decisión y un bosque aleatorio en Python. Utilizamos el algoritmo C4.5, una mejora del algoritmo ID3, para construir los árboles de decisión. Este clasificador puede manejar tanto atributos continuos como discretos y proporciona predicciones para un conjunto de datos de prueba.

## Árbol de Decisión

### Definición

Los árboles de decisión son modelos de aprendizaje automático utilizados para tareas de clasificación y regresión. Cada nodo interno del árbol representa una pregunta sobre una característica, cada rama representa el resultado de esa pregunta, y cada nodo hoja representa una clasificación o valor de predicción.

### Algoritmo ID3

El algoritmo ID3 es una metodología para construir árboles de decisión mediante los siguientes pasos:

1. **Selección del Atributo**: Utiliza la entropía y la ganancia de información para seleccionar el atributo que mejor clasifica los datos en cada nivel del árbol.
2. **División del Conjunto de Datos**: Divide el conjunto de datos en subconjuntos basados en los valores del atributo seleccionado.
3. **Construcción Recursiva del Árbol**: Repite los pasos anteriores hasta que se cumple un criterio de parada, como alcanzar nodos puros o una profundidad máxima del árbol.
4. **Poda del Árbol**: Elimina subárboles que no contribuyen significativamente a la precisión del modelo para evitar el sobreajuste.

### Algoritmo C4.5

El algoritmo C4.5 es una mejora del ID3 que introduce varias características adicionales, como el manejo de atributos continuos, la poda de árboles y el cálculo de la ganancia de información normalizada.

### Características Implementadas

1. **Atributos Continuos**: El algoritmo C4.5 puede manejar tanto atributos categóricos como continuos. Para los atributos continuos, se busca el umbral que maximiza la ganancia de información, dividiendo así el atributo en un conjunto de valores discretos.
    - **Ordenamiento de Valores**: Los valores únicos de un atributo continuo se ordenan en orden ascendente.
    - **Selección de Umbrales**: Se consideran los puntos medios entre cada par de valores consecutivos como posibles umbrales para dividir el atributo.
    - **Cálculo de la Ganancia de Información**: Para cada posible umbral, se calcula la ganancia de información resultante de dividir el conjunto de datos en dos grupos basados en ese umbral.
    - **Selección del Mejor Umbral**: El umbral que produce la mayor ganancia de información se selecciona como el umbral óptimo para dividir el atributo continuo.
    - **División del conjunto de datos**: Si la ganancia de información del atributo con el umbral óptimo es mayor que el resto de los atributos, se selecciona esta discretización como nodo divisor del árbol y continúa el proceso de construcción recursivo.
      
    Estas funcionalidades están implementadas en la clase `C45`.

2. **Criterio de División**: El criterio de división del C4.5 utiliza una métrica llamada Gain Ratio, que corrige la tendencia de la ganancia de información a favorecer atributos con muchos valores posibles.
    - **Cálculo del Split Info**: Mide la dispersión de los valores del atributo en el conjunto de datos.
    - **Cálculo del Gain Ratio**: Utiliza la ganancia de información y el split info para calcular el gain ratio.

    Estas funcionalidades están implementadas en la clase `C45`.

3. **Entropía**: Se calcula la entropía del conjunto de datos para medir la incertidumbre de la variable objetivo.

4. **Poda del Árbol**: Función para podar el árbol de decisión. Esta funcionalidad elimina subárboles que no contribuyen significativamente a la precisión del modelo, lo cual ayuda a reducir el sobreajuste (overfiting) del modelo..
5. **Manejo de Costos Asimétricos**: Permite incluir costos asimétricos asociados con diferentes tipos de errores de clasificación al evaluar la precisión del modelo.
6. **Clasificación con Datos Ponderados**: Permite clasificar conjuntos de datos que tienen diferentes pesos, lo que hace que podamos tratar de manera más efectiva los conjuntos de datos desbalanceados.

## Random Forest

### Definición

Random Forest es una técnica de ensamble que utiliza múltiples árboles de decisión entrenados sobre diferentes subconjuntos de datos y características. La predicción final se obtiene mediante la votación mayoritaria (para clasificación) o el promedio (para regresión) de las predicciones individuales de los árboles.

### Construcción del Bosque

1. **Bootstrapping**: Realiza un muestreo con reemplazo del conjunto de datos de entrenamiento para cada árbol.
2. **Selección Aleatoria de Características**: En cada división del árbol, se considera un subconjunto aleatorio de características.
3. **Combinación de Predicciones**: Combina las predicciones de todos los árboles del bosque para obtener la predicción final.

## Implementación

La implementación del proyecto se divide en varios módulos, cada uno con responsabilidades específicas:

- `main.py`: Script principal que carga los datos, entrena los modelos y realiza predicciones.
- `c45.py`: Implementa el algoritmo C4.5 para la construcción del árbol de decisión.
- `clasificadora_arbol_decision.py`: Define el clasificador de árbol de decisión utilizando el algoritmo C4.5.
- `impresora_arboles.py`: Proporciona funciones para imprimir árboles de decisión y bosques aleatorios.
- `lectora_datos.py`: Contiene una función para cargar datos desde un archivo CSV.
- `random_forest.py`: Implementa el bosque aleatorio, construyendo y combinando múltiples árboles de decisión.

## Conclusión

Este proyecto proporciona una implementación básica pero funcional de un clasificador basado en árboles de decisión y bosques aleatorios, utilizando el algoritmo C4.5. La modularidad del código permite futuras extensiones, como el soporte para problemas de regresión y la inclusión de más hiperparámetros configurables. Este enfoque asegura que la librería sea flexible y escalable, adaptándose a diversas necesidades de predicción en el campo del aprendizaje automático.

## Referencias

1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
2. Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning, 1(1), 81-106.
3. Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers.
4. https://uc-r.github.io/regression_trees
5. https://cienciadedatos.net/documentos/33_arboles_de_prediccion_bagging_random_forest_boosting#Ejemplo_regresi%C3%B3n
