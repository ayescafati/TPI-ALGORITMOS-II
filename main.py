from clasificadora_arbol_decision import ClasificadorArbolDecision
from random_forest import RandomForests
from lectora_datos import cargar_datos
from impresora_arboles import ImpresoraArboles
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score


def cargar_datos(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta)

if __name__ == "__main__":

    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    ruta_datos_entrenamiento = os.path.join(directorio_actual, "playgolf_data.csv")
    ruta_datos_prueba = os.path.join(directorio_actual, "playgolf_test.csv")

    # Cargar datos
    datos_entrenamiento = cargar_datos(ruta_datos_entrenamiento)
    datos_prueba = cargar_datos(ruta_datos_prueba)

    # Crear y ajustar el árbol de decisión usando C4.5
    clasificador = ClasificadorArbolDecision(datos_entrenamiento)
    clasificador.ajustar(estrategia = 'comun')  # Ajustar con la estrategia deseada ('comun', 'comun_clase' o 'probabilidad')

    # Imprimir el árbol de decisión resultante
  #  print("\nEl árbol de decisión resultante es:")
    clasificador.imprimir_arbol()

    # Predecir con el árbol de decisión
    resultados_arbol = clasificador.predecir(datos_prueba)
    print("\nPredicciones del conjunto de prueba usando el árbol de decisión:")
    print(resultados_arbol)

    # Crear y ajustar Random Forest
    rf = RandomForests(numero_arboles=5)
    rf.ajustar(datos_entrenamiento)  # Pasar DataFrame en lugar de lista

    # Imprimir los árboles del Random Forest
    ImpresoraArboles.imprimir_bosque_aleatorio(rf.arboles)

    # Predicciones del conjunto de prueba usando Random Forests
    resultados_rf = rf.predecir(datos_prueba)
    
    predicciones_rf_df = datos_prueba.copy()
    predicciones_rf_df['Prediccion'] = resultados_rf
    print("\nPredicciones del conjunto de prueba usando Random Forests:")
    print(predicciones_rf_df)



    # F1 score
    datos_prueba_subset = datos_prueba.iloc[:13] # me quedo con las primeras 13 filas
    # Calcula las métricas de evaluación solo para líneas 1 a 13 del dataset de prueba
    precision = precision_score(datos_prueba_subset['JugarGolf'], resultados_rf[:13], average='macro')
    recall = recall_score(datos_prueba_subset['JugarGolf'], resultados_rf[:13], average='macro')
    f1 = f1_score(datos_prueba_subset['JugarGolf'], resultados_rf[:13], average='macro')

    # Imprime las métricas de evaluación
    print("\nMétricas de evaluación del modelo Random Forests (solo para líneas 1 a 13 del dataset de prueba):")
    print("Precisión:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)