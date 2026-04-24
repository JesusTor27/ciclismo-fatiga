import streamlit as st
import os
import train
import predict

st.title("Predicción de Fatiga en Ciclismo")

# BOTÓN REINICIAR
if st.button("Reiniciar modelo"):
    if os.path.exists("modelo_ciclismo.pkl"):
        os.remove("modelo_ciclismo.pkl")
        st.success("Modelo eliminado, puedes volver a entrenar")
    else:
        st.warning("No hay modelo para eliminar")

# BOTÓN ENTRENAR
if st.button("Entrenar modelo"):
    if os.path.exists("modelo_ciclismo.pkl"):
        st.warning("El modelo ya fue entrenado")

    else:
        mse, mae, r2 = train.entrenar()
        st.success("Modelo entrenado correctamente")

        # mostrar métricas
        st.subheader("Métricas del modelo")
        st.write("MSE:", mse)
        st.write("MAE:", mae)
        st.write("R2:", r2)

st.write("---")

# INPUTS
frecuencia = st.number_input("Frecuencia cardiaca")
potencia = st.number_input("Potencia")
cadencia = st.number_input("Cadencia")
tiempo = st.number_input("Tiempo")
temperatura = st.number_input("Temperatura")
pendiente = st.number_input("Pendiente")
velocidad = st.number_input("Velocidad")

# BOTÓN PREDECIR
if st.button("Predecir"):
    valor, interpretacion = predict.predecir(
        frecuencia, potencia, cadencia, tiempo,
        temperatura, pendiente, velocidad
    )

    st.subheader("Resultado")
    st.write("Fatiga estimada:", valor)
    st.write("Interpretación:", interpretacion)
    # mostrar métricas
    st.subheader("Métricas del modelo")
    st.write("MSE:", mse)
    st.write("MAE:", mae)
    st.write("R2:", r2)
