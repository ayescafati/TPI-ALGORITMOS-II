
import math
from collections import Counter
from typing import List, Any, Optional

class C45:
    def __init__(self, filas: List[List[Any]]):
        self.filas = filas

    @staticmethod
    def dividirConjunto(filas: List[List[Any]], columna: int, valor: Any):
        lista1 = [fila for fila in filas if fila[columna] == valor]
        lista2 = [fila for fila in filas if fila[columna] != valor]
        return lista1, lista2

    def conteosUnicos(self, filas: List[List[Any]]):
        return dict(Counter(fila[-1] for fila in filas))

    def entropia(self, filas: List[List[Any]]):
        resultados = self.conteosUnicos(filas)
        ent = 0.0
        total_filas = len(filas)
        for r in resultados.values():
            p = r / total_filas
            ent -= p * math.log2(p) if p > 0 else 0
        return ent

    def mejorUmbral(self, atributo: int, datos_entrenamiento: List[List[Any]]) -> Optional[float]:
        valores = sorted(set(fila[atributo] for fila in datos_entrenamiento))
        mejor_ganancia = 0
        mejor_umbral = None
        for i in range(1, len(valores)):
            umbral = (valores[i - 1] + valores[i]) / 2
            conjunto_izquierdo, conjunto_derecho = [], []
            for fila in datos_entrenamiento:
                if fila[atributo] <= umbral:
                    conjunto_izquierdo.append(fila)
                else:
                    conjunto_derecho.append(fila)
            if len(conjunto_izquierdo) == 0 or len(conjunto_derecho) == 0:
                continue
            ganancia = self.entropia(datos_entrenamiento) - (len(conjunto_izquierdo) / len(datos_entrenamiento)) * self.entropia(conjunto_izquierdo) - (len(conjunto_derecho) / len(datos_entrenamiento)) * self.entropia(conjunto_derecho)
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_umbral = umbral
        return mejor_umbral

    def gainRatio(self, atributo: int, datos_entrenamiento: List[List[Any]]) -> float:
        ganancia = self.ganancia_informacion(atributo, datos_entrenamiento)
        split_info = self.calcularSplitInfo(atributo, datos_entrenamiento)
        return ganancia / split_info if split_info != 0 else 0

    def calcularSplitInfo(self, atributo: int, datos_entrenamiento: List[List[Any]]) -> float:
        valores_atributo = set(fila[atributo] for fila in datos_entrenamiento)
        total_filas = len(datos_entrenamiento)
        split_info = 0
        for valor in valores_atributo:
            p = sum(1 for fila in datos_entrenamiento if fila[atributo] == valor) / total_filas
            split_info -= p * math.log2(p) if p > 0 else 0
        return split_info

    def ganancia_informacion(self, atributo: int, datos_entrenamiento: List[List[Any]]) -> float:
        entropia_total = self.entropia(datos_entrenamiento)
        valores_atributo = set(fila[atributo] for fila in datos_entrenamiento)
        entropia_atributo = 0
        for valor in valores_atributo:
            subset = [fila for fila in datos_entrenamiento if fila[atributo] == valor]
            p = len(subset) / len(datos_entrenamiento)
            entropia_atributo += p * self.entropia(subset)
        return entropia_total - entropia_atributo
