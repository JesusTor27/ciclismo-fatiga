import joblib
import pandas as pd

MODELO_PATH = "modelo_ciclismo.pkl"

def interpretar_fatiga(valor):
    if valor <= 20:
        return "Muy baja - Sin fatiga significativa"
    elif valor <= 40:
        return "Baja - Esfuerzo leve"
    elif valor <= 60:
        return "Media - Fatiga moderada"
    elif valor <= 80:
        return "Alta - Fatiga evidente"
    else:
        return "Muy alta - Fatiga extrema"

def predecir(frecuencia, potencia, cadencia, tiempo, temperatura, pendiente, velocidad):
    
   
    modelo = joblib.load(MODELO_PATH)

    datos = pd.DataFrame([[frecuencia, potencia, cadencia, tiempo,
                           temperatura, pendiente, velocidad]],
        columns=["frecuencia_cardiaca", "potencia", "cadencia", "tiempo",
                 "temperatura", "pendiente", "velocidad"]
    )

    pred = modelo.predict(datos)
    valor = round(pred[0], 2)

    return valor, interpretar_fatiga(valor)
