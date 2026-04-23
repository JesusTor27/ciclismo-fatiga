import streamlit as st
import subprocess
import os
import predict

st.title("Predicción de Fatiga en Ciclismo")

# BOTÓN ENTRENAR
if st.button("Entrenar modelo"):
    if os.path.exists("modelo_ciclismo.pkl"):
        st.warning("El modelo ya fue entrenado")
    else:
        subprocess.run(["python", "train.py"])
        st.success("Modelo entrenado correctamente")

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