import pandas as pd
from c45 import C45
from impresora_arboles import ImpresoraArboles

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
        self.arbol = self.crearArbolDecision(self.df.values.tolist())

    def crearArbolDecision(self, filas, profundidad=0):
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
            true_branch = self.crearArbolDecision(best_sets[0], profundidad + 1)
            false_branch = self.crearArbolDecision(best_sets[1], profundidad + 1)
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
        if arbol is None:
            arbol = self.arbol

        if "hoja" in arbol:
            return max(arbol["hoja"], key=arbol["hoja"].get)

        valor = instancia[arbol["columna"]]
        if valor == arbol["valor"]:
            return self.clasificar(instancia, arbol["ramas"][0])
        else:
            return self.clasificar(instancia, arbol["ramas"][1])

    # Métodos adicionales para podado de árboles y costos asimétricos

    def podarArbol(self, arbol, datos_validacion):
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

    def evaluarPrecision(self, arbol, datos_validacion, costos=None):
        total = len(datos_validacion)
        correctos = 0
        for fila in datos_validacion:
            prediccion = self.clasificar(fila, arbol)
            if prediccion == fila[-1]:
                correctos += 1
            elif costos:
                costo_error = costos.get((fila[-1], prediccion), 1)
                correctos += 1 / costo_error
        return correctos / total
