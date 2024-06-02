# Proyecto: Librería de Modelos Predictivos con Árboles de Decisión y Bosques Aleatorios

## Enunciado del Problema

Se desea construir una librería que permita desarrollar modelos predictivos mediante una versión simplificada del ensamble RandomForest. El objetivo principal es entrenar un modelo con un conjunto de datos etiquetados, utilizando técnicas de aprendizaje supervisado, para luego predecir la variable objetivo. Nos enfocamos principalmente en problemas de clasificación, con posibilidad de extender la funcionalidad a problemas de regresión en el futuro.

## Introducción

Este proyecto implementa un clasificador basado en árboles de decisión y un bosque aleatorio en Python. Utilizamos el algoritmo C4.5, una mejora del algoritmo ID3, para construir los árboles de decisión. Este clasificador es capaz de manejar tanto atributos continuos como discretos y proporciona predicciones para un conjunto de datos de prueba.

## Árbol de Decisión

### Definición

Los árboles de decisión son modelos de aprendizaje automático utilizados para tareas de clasificación y regresión. Cada nodo interno del árbol representa una pregunta sobre una característica, cada rama representa el resultado de esa pregunta, y cada nodo hoja representa una clasificación o valor de predicción.

### Algoritmo ID3

El algoritmo ID3 es una metodología para construir árboles de decisión mediante los siguientes pasos:

1. **Selección del Atributo**: Utiliza la entropía y la ganancia de información para seleccionar el atributo que mejor clasifica los datos en cada nivel del árbol.
2. **División del Conjunto de Datos**: Divide el conjunto de datos en subconjuntos basados en los valores del atributo seleccionado.
3. **Construcción Recursiva del Árbol**: Repite los pasos anteriores hasta que se cumple un criterio de parada, como alcanzar nodos puros o una profundidad máxima del árbol.
4. **Podado del Árbol**: Elimina subárboles que no contribuyen significativamente a la precisión del modelo para evitar el sobreajuste.

### Algoritmo C4.5

El algoritmo C4.5 extiende ID3 e introduce varias mejoras:

- **Atributos Continuos**: Permite manejar atributos tanto categóricos como continuos, buscando el umbral que maximiza la ganancia de información.
- **Valores Faltantes**: Puede manejar datos incompletos mediante diversas estrategias de imputación.
- **Ratio de Ganancia**: Utiliza una métrica llamada Gain Ratio para corregir la tendencia de la ganancia de información a favorecer atributos con muchos valores posibles.
- **Podado Posterior**: Incluye un proceso de podado posterior para reducir la complejidad del modelo y evitar el sobreajuste.

## Random Forest

### Definición

Random Forest es una técnica de ensamble que utiliza múltiples árboles de decisión entrenados sobre diferentes subconjuntos de datos y características. La predicción final se obtiene mediante la votación mayoritaria (para clasificación) o el promedio (para regresión) de las predicciones individuales de los árboles.

### Construcción del Bosque

1. **Bootstrapping**: Realiza un muestreo con reemplazo del conjunto de datos de entrenamiento para cada árbol.
2. **Selección Aleatoria de Características**: En cada división del árbol, se considera un subconjunto aleatorio de características.
3. **Combinación de Predicciones**: Combina las predicciones de todos los árboles del bosque para obtener la predicción final.

## Implementación

La implementación del proyecto se divide en varios módulos, cada uno con responsabilidades específicas:

1. **`main.py`**: Script principal que carga los datos, entrena los modelos y realiza predicciones.
2. **`c45.py`**: Implementa el algoritmo C4.5 para la construcción del árbol de decisión.
3. **`clasificadora_arbol_decision.py`**: Define el clasificador de árbol de decisión utilizando el algoritmo C4.5.
4. **`impresora_arboles.py`**: Proporciona funciones para imprimir árboles de decisión y bosques aleatorios.
5. **`lectora_datos.py`**: Contiene una función para cargar datos desde un archivo CSV.
6. **`random_forest.py`**: Implementa el bosque aleatorio, construyendo y combinando múltiples árboles de decisión.

## Uso y Ejecución

Para ejecutar el proyecto, sigue estos pasos:

1. Clona el repositorio desde GitHub.
2. Instala las dependencias necesarias utilizando pip.
3. Ejecuta el script `main.py` con el conjunto de datos de entrada.

## Conclusión

Este proyecto proporciona una implementación básica pero funcional de un clasificador basado en árboles de decisión y bosques aleatorios, utilizando el algoritmo C4.5. La modularidad del código permite futuras extensiones, como el soporte para problemas de regresión y la inclusión de más hiperparámetros configurables. Este enfoque asegura que la librería sea flexible y escalable, adaptándose a diversas necesidades de predicción en el campo del aprendizaje automático.

## Referencias

1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
2. Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning, 1(1), 81-106.
3. Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers.
4. https://uc-r.github.io/regression_trees
5. https://cienciadedatos.net/documentos/33_arboles_de_prediccion_bagging_random_forest_boosting#Ejemplo_regresi%C3%B3n
