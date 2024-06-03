
import math
from collections import Counter
from typing import List, Any, Optional


class DecisionNode:
    def __init__(self, columna: int = -1, valor: Any = None, resultados: Optional[dict] = None, verdadero: Optional['DecisionNode'] = None, falso: Optional['DecisionNode'] = None):
        self.columna = columna
        self.valor = valor
        self.resultados = resultados
        self.verdadero = verdadero
        self.falso = falso

class C45:
    def __init__(self, filas: List[List[Any]]) -> None:
        self.filas = filas

    @staticmethod
    def dividirConjunto(filas: List[List[Any]], columna: int, valor: Any) -> List[List[Any]]:
        lista1 = [fila for fila in filas if fila[columna] == valor]
        lista2 = [fila for fila in filas if fila[columna] != valor]
        return lista1, lista2

    def conteosUnicos(self, filas: List[List[Any]]) -> dict:
        return dict(Counter(fila[-1] for fila in filas))

    def entropia(self, filas: List[List[Any]]) -> float:
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

    def gananciaInformacion(self, atributo: int, datos_entrenamiento: List[List[Any]]) -> float:
        entropia_total = self.entropia(datos_entrenamiento)
        valores_atributo = set(fila[atributo] for fila in datos_entrenamiento)
        entropia_atributo = 0
        for valor in valores_atributo:
            subset = [fila for fila in datos_entrenamiento if fila[atributo] == valor]
            p = len(subset) / len(datos_entrenamiento)
            entropia_atributo += p * self.entropia(subset)
        return entropia_total - entropia_atributo



# PODADO DE ARBOLES DE DECISION #
    

    def podarArbol(self, arbol: dict, datos_validacion: List[List[Any]]) -> dict:
        if not isinstance(arbol, dict):
            return arbol

        atributo = list(arbol.keys())[0]
        subarboles = arbol[atributo]
        atributo_valor = datos_validacion[0][atributo]

        if atributo_valor not in subarboles:
            return arbol

        subarbol = subarboles[atributo_valor]
        subarbol_poda = self.podarArbol(subarbol, datos_validacion[1:])

        if self.evaluarPrecision(subarbol_poda, datos_validacion) >= self.evaluarPrecision(arbol, datos_validacion):
            return subarbol_poda
        else:
            return arbol

    def evaluarPrecision(self, arbol: dict, datos_validacion: List[List[Any]], costos: Optional[dict] = None) -> float:
        total = len(datos_validacion)
        correctos = 0
        for fila in datos_validacion:
            prediccion = self.clasificarFila(arbol, fila)
            if prediccion == fila[-1]:
                correctos += 1
            else:
                if costos:
                    costo_error = costos.get((fila[-1], prediccion), 1)  # Obtener el costo asociado al error de clasificación
                    correctos += 1 / costo_error
        return correctos / total

    def clasificarFila(self, arbol: dict, fila: List[Any]) -> Any:
        if not isinstance(arbol, dict):
            return arbol
        atributo = list(arbol.keys())[0]
        subarboles = arbol[atributo]
        atributo_valor = fila[atributo]
        if atributo_valor not in subarboles:
            return None
        subarbol = subarboles[atributo_valor]
        return self.clasificarFila(subarbol, fila)

    @staticmethod
    def dividirConjuntoConPesos(filas: List[List[Any]], columna: int, valor: Any, pesos: List[float]) -> List[List[List[Any]]]:
        conjunto_verdadero = []
        conjunto_falso = []
        pesos_verdadero = []
        pesos_falso = []

        for i, fila in enumerate(filas):
            if fila[columna] == valor:
                conjunto_verdadero.append(fila)
                pesos_verdadero.append(pesos[i])
            else:
                conjunto_falso.append(fila)
                pesos_falso.append(pesos[i])

        return conjunto_verdadero, conjunto_falso, pesos_verdadero, pesos_falso
    


    # Inicio Manejar valores faltantes #

    def manejar_valores_faltantes(self, filas: List[List[Any]], estrategia: str = 'comun') -> List[List[Any]]:
        if estrategia == 'comun':
            for col in range(len(filas[0]) - 1):
                valores = [fila[col] for fila in filas if fila[col] is not None]
                valor_comun = max(set(valores), key=valores.count)
                for fila in filas:
                    if fila[col] is None:
                        fila[col] = valor_comun
        elif estrategia == 'comun_clase':
            clases = self.conteosUnicos(filas).keys()
            for clase in clases:
                filas_clase = [fila for fila in filas if fila[-1] == clase]
                for col in range(len(filas[0]) - 1):
                    valores = [fila[col] for fila in filas_clase if fila[col] is not None]
                    if valores:
                        valor_comun = max(set(valores), key=valores.count)
                        for fila in filas_clase:
                            if fila[col] is None:
                                fila[col] = valor_comun
        elif estrategia == 'probabilidad':
            for col in range(len(filas[0]) - 1):
                valores = [fila[col] for fila in filas if fila[col] is not None]
                conteo_valores = Counter(valores)
                total = sum(conteo_valores.values())
                probabilidades = {k: v / total for k, v in conteo_valores.items()}
                for fila in filas:
                    if fila[col] is None:
                        fila[col] = max(probabilidades, key=probabilidades.get)
        return filas

    def entrenar(self, estrategia: str = 'comun') -> None:
        self.filas = self.manejar_valores_faltantes(self.filas, estrategia)

    def clasificar(self, observacion: List[Any], arbol: Optional[DecisionNode] = None) -> dict:
        if arbol is None:
            arbol = self.arbol

        if arbol.resultados is not None:
            return arbol.resultados

        v = observacion[arbol.columna]
        rama = None
        if v is None:
            # Asignar la probabilidad basada en frecuencias observadas en el nodo actual
             rama = arbol.verdadero if len(arbol.verdadero.resultados) > len(arbol.falso.resultados) else arbol.falso
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= arbol.valor:
                     rama = arbol.verdadero
                else:
                     rama = arbol.falso
            else:
                if v == arbol.valor:
                     rama = arbol.verdadero
                else:
                     rama = arbol.falso
        return self.clasificar(observacion,  rama)
    
    # Fin Manejar valores faltantes #
