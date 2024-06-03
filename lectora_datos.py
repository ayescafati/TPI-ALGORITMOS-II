
# A continuación tenemos una simple y confiable función para leer archivos CSV con pandas.

def cargar_datos(ruta):
    import pandas as pd
    return pd.read_csv(ruta)
