class ImpresoraArboles:

    @staticmethod
    def imprimir_arbol_decision(arbol: any, prefijo = '', nivel = 0) -> None:
        """
        Imprime un árbol de decisión de forma recursiva con formato indentado.
        """
        if isinstance(arbol, dict):
            if 'hoja' in arbol:
                print('  ' * nivel + prefijo + f"{arbol['hoja']}")
            else:
                print('  ' * nivel + prefijo + f"{arbol['columna']}: {arbol['valor']}")
                for rama in arbol['ramas']:
                    ImpresoraArboles.imprimir_arbol_decision(rama, prefijo, nivel + 1)

    @staticmethod
    def imprimir_bosque_aleatorio(bosque: list) -> None:
        """
        Imprime los árboles en el bosque aleatorio.
        """
        print("\nÁrboles del Bosque Aleatorio:")
        for i, arbol in enumerate(bosque, start=1):
            print(f"\nÁrbol {i}:")
            ImpresoraArboles.imprimir_arbol_decision(arbol.arbol)