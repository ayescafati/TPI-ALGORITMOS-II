# En este archivo tenemos la clase RandomForests que se encarga de crear y ajustar un 
# conjunto de arboles de decisión. Esto lo hace a través de la función ajustar(), que 
# crea un conjunto de arboles de decisión y los ajusta a un subconjunto de datos. Y también
# tiene la función predecir(), que se encarga de predecir un conjunto de datos de prueba
# usando los arboles de decisión creados. Además, tiene la función imprimir_arboles_forest()
# que imprime los arboles de decisión creados.
# Falta la poda!!!!!!!!! (IMPORTANTE)

import random
import pandas as pd
from collections import Counter
from clasificadora_arbol_decision import ClasificadorArbolDecision
from impresora_arboles import ImpresoraArboles

class RandomForests:
    def __init__(self, numero_arboles = 10,  max_atributos = None): # agregamos el parametro max_atributos para poder setear la cant de atributos a usar
        self.numero_arboles = numero_arboles
        self.arboles = [ ]
        self.max_atributos = max_atributos

    def ajustar(self, datos):
        for i in range(self.numero_arboles):
            subconjunto_indices = [random.randint(0, len(datos) - 1) for _ in range(len(datos))]
            subconjunto = datos.iloc[subconjunto_indices]
            nombres_atributos = list(subconjunto.columns)
            nombres_atributos.remove(subconjunto.columns[-1])
            if self.max_atributos:
                nombres_atributos = random.sample(nombres_atributos, self.max_atributos)
            arbol = ClasificadorArbolDecision(subconjunto[nombres_atributos + [subconjunto.columns[-1]]])
            arbol.ajustar()
            self.arboles.append(arbol)
        
  #  def ajustar(self, datos):     for i in range(self.numero_arboles):
  #          subconjunto_indices = [random.randint(0, len(datos) - 1) for _ in range(len(datos))]
  #          subconjunto = [datos[j] for j in subconjunto_indices]
  #          nombres_atributos = list(datos.columns)
  #          nombres_atributos.remove(datos.columns[-1])
  #          if self.max_atributos:
  #              nombres_atributos = random.sample(nombres_atributos, self.max_atributos)
  #          subconjunto = [[fila[k] for k in nombres_atributos] + [fila[-1]] for fila in subconjunto]
  #          arbol = ClasificadorArbolDecision(subconjunto)
  #          arbol.ajustar()
  #          self.arboles.append(arbol)


    def predecir(self, datos_prueba):
        predicciones = []
        for arbol in self.arboles:
            predicciones.append(arbol.predecir(datos_prueba)['Prediccion'])

        predicciones_transpuestas = list(zip(*predicciones))
        predicciones_finales = [Counter(prediccion).most_common(1)[0][0] for prediccion in predicciones_transpuestas]
        return predicciones_finales


    def imprimir_arboles_forest(self):
        impresora = ImpresoraArboles()
        print("\nÁrboles del Random Forest:")
        for i, arbol in enumerate(self.arboles, start=1):
            print(f"\nÁrbol {i}:")
            impresora.imprimir_arbol_decision(arbol.arbol)