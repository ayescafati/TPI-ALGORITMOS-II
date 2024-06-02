
from clasificadora_arbol_decision import ClasificadorArbolDecision
from random_forest import RandomForests
from lectora_datos import cargar_datos
from impresora_arboles import ImpresoraArboles
import pandas as pd



def cargar_datos(ruta):
    return pd.read_csv(ruta)

if __name__ == "__main__":
    ruta_datos_entrenamiento = "/Users/ayelenscafati/Desktop/LCD/2024/Algoritmos II/GuiaArboles/ElRandomForest/playgolf_data.csv"
    ruta_datos_prueba = "/Users/ayelenscafati/Desktop/LCD/2024/Algoritmos II/GuiaArboles/ElRandomForest/playgolf_test.csv"

    # Cargar datos
    datos_entrenamiento = cargar_datos(ruta_datos_entrenamiento)
    datos_prueba = cargar_datos(ruta_datos_prueba)

    # Crear y ajustar el árbol de decisión
    dt = ClasificadorArbolDecision(datos_entrenamiento)
    dt.ajustar()

    print("\nEl árbol de decisión resultante es:")
    ImpresoraArboles.imprimir_arbol_decision(dt.arbol)

    resultados_arbol = dt.predecir(datos_prueba)
    print("\nPredicciones del conjunto de prueba usando el árbol de decisión:")
    print(resultados_arbol)

    # Crear y ajustar Random Forest
    rf = RandomForests(numero_arboles=5)
    rf.ajustar(datos_entrenamiento)  # Usar el mismo conjunto de datos de entrenamiento que el árbol de decisión

    # Imprimir los árboles del Random Forest
    ImpresoraArboles.imprimir_bosque_aleatorio(rf.arboles)

    # Predicciones del conjunto de prueba usando Random Forests
    resultados_rf = rf.predecir(datos_prueba)  # Obtener predicciones del conjunto de prueba usando Random Forests
    
    predicciones_rf_df = datos_prueba.copy()  # Crea una copia del DataFrame de datos de prueba
    predicciones_rf_df['Prediccion'] = resultados_rf  # Agrega la columna de predicciones
    print("\nPredicciones del conjunto de prueba usando Random Forests:")
    print(predicciones_rf_df)