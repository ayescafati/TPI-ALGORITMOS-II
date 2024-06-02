class ImpresoraArboles:

    @staticmethod
    def imprimir_arbol_decision(arbol, prefijo='', nivel=0):
        """
        Imprime un árbol de decisión de forma recursiva con formato indentado.

        Args:
            arbol (dict): Árbol de decisión representado como un diccionario.
            prefijo (str): Prefijo para la impresión del nodo.
            nivel (int): Nivel actual en la jerarquía del árbol.
        """
        if isinstance(arbol, dict):
            if 'hoja' in arbol:
                print('  ' * nivel + prefijo + f"{arbol['hoja']}")
            else:
                print('  ' * nivel + prefijo + f"{arbol['columna']}: {arbol['valor']}")
                for rama in arbol['ramas']:
                    ImpresoraArboles.imprimir_arbol_decision(rama, prefijo, nivel + 1)
    @staticmethod
    def imprimir_bosque_aleatorio(bosque):
        """
        Imprime los árboles en el bosque aleatorio.

        Args:
            bosque (list): Lista de árboles de decisión.
        """
        print("\nÁrboles del Bosque Aleatorio:")
        for i, arbol in enumerate(bosque, start=1):
            print(f"\nÁrbol {i}:")
            ImpresoraArboles.imprimir_arbol_decision(arbol.arbol)