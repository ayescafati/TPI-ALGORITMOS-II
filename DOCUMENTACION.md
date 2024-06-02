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


## Algoritmo C4.5

### Características Implementadas

#### 1. Atributos Continuos
El algoritmo C4.5 puede manejar tanto atributos categóricos como continuos. Para los atributos continuos, se busca el umbral que maximiza la ganancia de información, dividiendo así el atributo en un conjunto de valores discretos.

- **Ordenamiento de Valores:** Los valores únicos de un atributo continuo se ordenan en orden ascendente.
- **Selección de Umbrales:** Se consideran los puntos medios entre cada par de valores consecutivos como posibles umbrales para dividir el atributo.
- **Cálculo de la Ganancia de Información:** Para cada posible umbral, se calcula la ganancia de información resultante de dividir el conjunto de datos en dos grupos basados en ese umbral.
- **Selección del Mejor Umbral:** El umbral que produce la mayor ganancia de información se selecciona como el umbral óptimo para dividir el atributo continuo.

Estas funcionalidades están implementadas en la clase `C45`.

#### 2. Gain Ratio
El criterio de división del C4.5 utiliza una métrica llamada Gain Ratio, que corrige la tendencia de la ganancia de información a favorecer atributos con muchos valores posibles.

- **Cálculo del Split Info:** Mide la dispersión de los valores del atributo en el conjunto de datos.
- **Cálculo del Gain Ratio:** Utiliza la ganancia de información y el split info para calcular el gain ratio.

Estas funcionalidades están implementadas en la clase `C45`.

#### 3. Entropía
Se calcula la entropía del conjunto de datos para medir la incertidumbre de la variable objetivo.

### Características No Implementadas

- **Valores Faltantes:** No hay una implementación explícita para manejar valores faltantes en los datos de entrenamiento.
- **Podado del Árbol:** El proceso de podado por reglas no está implementado.
- **Costos Asimétricos:** No se maneja explícitamente los costos asimétricos asociados con diferentes tipos de errores de clasificación.
- **Datos Ponderados:** No se muestra el uso de datos ponderados para tratar conjuntos de datos desbalanceados.

### Ejemplo de Uso

La clase `ClasificadorArbolDecision` utiliza la clase `C45` para construir y entrenar un árbol de decisión.

from c45 import C45
import pandas as pd

class ClasificadorArbolDecision:
    def __init__(self, datos, profundidad_max=None, min_instancias=1):
        if isinstance(datos, str):
            self.df = pd.read_csv(datos)
        elif isinstance(datos, pd.DataFrame):
            self.df = datos
        else:
            raise ValueError("Los datos deben ser una ruta de archivo CSV o un DataFrame de pandas.")
        
        self.atributo_objetivo = self.df.keys()[-1]
        self.nombres_atributos = list(self.df.keys())
        self.nombres_atributos.remove(self.atributo_objetivo)
        self.arbol = None
        self.profundidad_max = profundidad_max
        self.min_instancias = min_instancias
        self.c45 = C45(self.df.values.tolist())

    def ajustar(self):
        self.arbol = self.crearArbolDecisionDesde(self.df.values.tolist())

    def crearArbolDecisionDesde(self, filas, profundidad=0):
        if not filas:
            return None

        current_score = self.c45.entropia(filas)
        best_gain = 0.0
        best_attribute = None
        best_sets = None

        for col in range(len(filas[0]) - 1):
            column_values = set([row[col] for row in filas])
            for value in column_values:
                set1, set2 = self.c45.dividirConjunto(filas, col, value)

                if not set1 or not set2:
                    continue

                p = float(len(set1)) / len(filas)
                gain = current_score - p * self.c45.entropia(set1) - (1 - p) * self.c45.entropia(set2)

                if gain > best_gain:
                    best_gain = gain
                    best_attribute = (col, value)
                    best_sets = (set1, set2)

        if best_gain > 0 and (self.profundidad_max is None or profundidad < self.profundidad_max):
            true_branch = self.crearArbolDecisionDesde(best_sets[0], profundidad + 1)
            false_branch = self.crearArbolDecisionDesde(best_sets[1], profundidad + 1)
            return {"columna": best_attribute[0], "valor": best_attribute[1], "ramas": [true_branch, false_branch]}
        else:
            return {"hoja": self.c45.conteosUnicos(filas)}

    def predecir(self, datos_prueba):
        if isinstance(datos_prueba, str):
            df_nuevo = pd.read_csv(datos_prueba)
        elif isinstance(datos_prueba, pd.DataFrame):
            df_nuevo = datos_prueba
        else:
            raise ValueError("Los datos de prueba deben ser una ruta de archivo CSV o un DataFrame de pandas.")

        df_nuevo['Prediccion'] = df_nuevo.apply(self.clasificar, axis=1)
        return df_nuevo

    def imprimir_arbol(self):
        def print_tree(tree, indent=''):
            if "hoja" in tree:
                print(f"{indent}{tree['hoja']}")
            else:
                col_name = self.nombres_atributos[tree['columna']]
                print(f"{indent}{col_name} = {tree['valor']}")
                print(f"{indent}--> True:")
                print_tree(tree['ramas'][0], indent + '  ')
                print(f"{indent}--> False:")
                print_tree(tree['ramas'][1], indent + '  ')

        print("El árbol de decisión resultante es:")
        ImpresoraArboles.imprimir_arbol_decision(self.arbol)

    def obtener_clase_mayoritaria(self, df):
        return df[self.atributo_objetivo].mode()[0]

    def clasificar(self, instancia, arbol=None, defecto=None):
        if arbol es None:
            arbol = self.arbol

        if "hoja" en arbol:
            return max(arbol["hoja"], key=arbol["hoja"].get)

        valor = instancia[arbol["columna"]]
        if valor == arbol["valor"]:
            return self.clasificar(instancia, arbol["ramas"][0])
        else:
            return self.clasificar(instancia, arbol["ramas"][1])

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
