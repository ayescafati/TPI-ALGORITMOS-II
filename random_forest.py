# En este archivo tenemos la clase RandomForests que se encarga de crear y ajustar un 
# conjunto de arboles de decisión. Esto lo hace a través de la función ajustar(), que 
# crea un conjunto de arboles de decisión y los ajusta a un subconjunto de datos. Y también
# tiene la función predecir(), que se encarga de predecir un conjunto de datos de prueba
# usando los arboles de decisión creados. Además, tiene la función imprimir_arboles_forest()
# que imprime los arboles de decisión creados.

import random
import pandas as pd
from collections import Counter
from clasificadora_arbol_decision import ClasificadorArbolDecision
from impresora_arboles import ImpresoraArboles


class RandomForests:
    def __init__(self, numero_arboles=10, profundidad_max=None, min_instancias=1):
        self.numero_arboles = numero_arboles
        self.profundidad_max = profundidad_max
        self.min_instancias = min_instancias
        self.arboles = []

    def ajustar(self, datos):
        if isinstance(datos, list):
            datos = pd.DataFrame(datos)
            
        for _ in range(self.numero_arboles):
            # Crear un subconjunto de datos con reemplazo
            subconjunto_indices = random.choices(datos.index, k=len(datos))
            subconjunto = datos.iloc[subconjunto_indices]

            # Crear un clasificador de árbol de decisión para el subconjunto
            arbol = ClasificadorArbolDecision(subconjunto, 
                                              profundidad_max=self.profundidad_max, 
                                              min_instancias=self.min_instancias)
            arbol.ajustar()
            self.arboles.append(arbol)

    def predecir(self, datos_prueba):
        if isinstance(datos_prueba, list):
            datos_prueba = pd.DataFrame(datos_prueba)
            
        predicciones = []

        for arbol in self.arboles:
            predicciones_arbol = arbol.predecir(datos_prueba)['Prediccion']
            predicciones.append(predicciones_arbol)

        # Transponer la lista de listas para hacer el voto mayoritario
        predicciones = list(zip(*predicciones))
        predicciones_finales = [max(set(p), key=p.count) for p in predicciones]
        
        return predicciones_finales

    def imprimir_arboles_forest(self):
        impresora = ImpresoraArboles()
        print("\nÁrboles del Random Forest:")
        for i, arbol in enumerate(self.arboles, start=1):
            print(f"\nÁrbol {i}:")
            impresora.imprimir_arbol_decision(arbol.arbol)



'''


class RandomForests:
    def __init__(self, numero_arboles=10, profundidad_max=None, min_instancias=1):
        self.numero_arboles = numero_arboles
        self.profundidad_max = profundidad_max
        self.min_instancias = min_instancias
        self.arboles = []

    def ajustar(self, datos):
        if isinstance(datos, list):
            datos = pd.DataFrame(datos)

        num_caracteristicas = datos.shape[1] - 1  # Asumiendo que la ultima columna es la de la etiqueta
        num_caracteristicas_seleccionadas = int(num_caracteristicas ** 0.5)

        for _ in range(self.numero_arboles):
            # Crear un subconjunto de datos con reemplazo
            subconjunto_indices = random.choices(datos.index, k = len(datos))
            subconjunto = datos.iloc[subconjunto_indices]

            # Seleccionamos, aleatoriamente, un subconjunto de características
            caracteristicas_seleccionadas = random.sample(list(datos.columns[:-1]), num_caracteristicas_seleccionadas)
            subconjunto = subconjunto[caracteristicas_seleccionadas + [datos.columns[-1]]]

            arbol = ClasificadorArbolDecision(subconjunto, 
                                              profundidad_max=self.profundidad_max, 
                                              min_instancias=self.min_instancias) # con esto creamos un clasificador de árbol de decisión para el subconjunto
            arbol.ajustar()
            self.arboles.append(arbol)

    def predecir(self, datos_prueba):
        if isinstance(datos_prueba, list):
            datos_prueba = pd.DataFrame(datos_prueba)
            
        predicciones = []

        for arbol in self.arboles:
            predicciones_arbol = arbol.predecir(datos_prueba)['Prediccion']
            predicciones.append(predicciones_arbol)

        # Transponer la lista de listas para hacer el voto mayoritario
        predicciones = list(zip(*predicciones))
        predicciones_finales = [max(set(p), key=p.count) for p in predicciones]
        
        return predicciones_finales

    def imprimir_arboles_forest(self):
        impresora = ImpresoraArboles()
        print("\nÁrboles del Random Forest:")
        for i, arbol in enumerate(self.arboles, start=1):
            print(f"\nÁrbol {i}:")
            impresora.imprimir_arbol_decision(arbol.arbol)




'''