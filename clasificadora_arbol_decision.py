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

    def ajustar(self, estrategia ='comun'): # Llama a manejar_valores_faltantes y despues construye el arbol de decision
        self.c45.filas = self.c45.manejar_valores_faltantes(self.c45.filas, estrategia) 
        self.arbol = self.crearArbolDecision(self.c45.filas)

    def crearArbolDecision(self, filas, profundidad = 0) -> dict :
        """
        Construye un árbol de decisión recursivamente a partir del conjunto de datos dado.

        Parametros:
            - filas: Lista de filas que representan el conjunto de datos.
            - profundidad: Profundidad actual del árbol (por defecto 0).

        Devuelve:
            - Diccionario representando el nodo del árbol de decisión.

        Objetivo: La función de este método es dividir el conjunto de datos en subgrupos más homogéneos basados en la variable objetivo. 
        Para cada atributo en el conjunto de datos, se calcula la "entropía antes", que representa la incertidumbre general antes de la división, 
        y la "entropía después", que mide la incertidumbre en los subgrupos resultantes tras la división. 
        La "ganancia de información" se obtiene restando la entropía después de la división de la entropía antes de la misma. 
        El objetivo es encontrar la división que maximice esta ganancia, lo que indica una mayor reducción en la incertidumbre o entropía de los subgrupos resultantes. 
        Este proceso se repite de manera recursiva para construir el árbol de decisiones, donde en cada paso se selecciona el atributo que proporciona
        la mayor ganancia de información.

        """
        # Si no hay filas, se retorna None
        if not filas:
            return None

        # Calcula la entropía antes de la división
        entropia_antes = self.c45.entropia(filas)

        # Inicializa variables para almacenar la mejor ganancia de información y la mejor división
        mejor_ganancia = 0.0
        mejor_atributo = None
        mejores_conjuntos = None

        # Itera sobre cada columna del conjunto de datos
        for col in range(len(filas[0]) - 1):
            # Obtiene los valores únicos en la columna
            valores_columna = set([fila[col] for fila in filas])
            # Itera sobre cada valor único en la columna
            for valor in valores_columna:
                # Divide el conjunto de datos en dos según el valor actual
                conjunto1, conjunto2 = self.c45.dividirConjunto(filas, col, valor)

                # Si uno de los conjuntos está vacío, se omite
                if not conjunto1 or not conjunto2:
                    continue

                # Calcula la entropía después de la división
                p = float(len(conjunto1)) / len(filas)
                entropia_despues = p * self.c45.entropia(conjunto1) + (1 - p) * self.c45.entropia(conjunto2)
                
                # Calcula la ganancia de información
                ganancia = entropia_antes - entropia_despues

                # Actualiza la mejor ganancia y la mejor división si corresponde
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_atributo = (col, valor)
                    mejores_conjuntos = (conjunto1, conjunto2)

        # Si la ganancia de información es mayor que cero y no se alcanzó la profundidad máxima,
        # se construyen las ramas del árbol recursivamente.
        if mejor_ganancia > 0 and (self.profundidad_max is None or profundidad < self.profundidad_max):
            rama_verdadera = self.crearArbolDecision(mejores_conjuntos[0], profundidad + 1)
            rama_falsa = self.crearArbolDecision(mejores_conjuntos[1], profundidad + 1)
            return {"columna": mejor_atributo[0], "valor": mejor_atributo[1], "ramas": [rama_verdadera, rama_falsa]}
        # Si no se cumplen las condiciones anteriores, se retorna una hoja del árbol.
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

    def clasificar(self, instancia, arbol=None):
        if arbol is None:
            arbol = self.arbol

        if "hoja" in arbol:
            return max(arbol["hoja"], key=arbol["hoja"].get)

        valor = instancia[arbol["columna"]]
        if valor == arbol["valor"]:
            return self.clasificar(instancia, arbol["ramas"][0])
        else:
            return self.clasificar(instancia, arbol["ramas"][1])
