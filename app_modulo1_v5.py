import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from fpdf import FPDF
import tempfile
import os
from datetime import datetime

# Función para procesar archivo subido
def procesar_archivo(uploaded_file):
    datos_completos = pd.read_csv(uploaded_file)
    datos_personales = datos_completos.iloc[0, :].to_frame().T
    datos_senales = datos_completos.iloc[1:, :].astype(float)
    return datos_personales, datos_senales

# Función para analizar señal
def analizar_senal(datos_senales, eje='Aceleracion X'):
    señal = datos_senales[eje].values
    tiempo = datos_senales['Tiempo'].values
    fs = 1000  # Frecuencia de muestreo
    N = len(señal)
    T = 1.0 / fs
    yf = fft(señal)
    xf = fftfreq(N, T)[:N//2]
    amplitudes = 2.0/N * np.abs(yf[0:N//2])
    frec_dominante = xf[np.argmax(amplitudes)]
    varianza = np.var(señal)
    rms = np.sqrt(np.mean(señal**2))
    desplazamiento_max = np.max(np.abs(señal)) * 0.1  # cm (escala estimada)
    return frec_dominante, varianza, rms, desplazamiento_max, xf, amplitudes

# Función para diagnóstico simple
def diagnostico_simple(frecs_dom):
    if all(4 <= f <= 6 for f in frecs_dom):
        return "Parkinson"
    elif all(6 <= f <= 12 for f in frecs_dom):
        return "Temblor esencial"
    elif all(f < 4 for f in frecs_dom):
        return "Normal"
    else:
        return "Indeterminado"

# Función para generar PDF
def generar_pdf(datos_personales, resultados, diagnostico):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Informe de análisis de temblores", ln=True, align='C')
    pdf.ln(10)

    # Datos personales
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Datos del paciente:", ln=True)
    pdf.set_font("Arial", size=12)
    for col in datos_personales.columns:
        pdf.cell(200, 10, txt=f"{col}: {datos_personales[col].values[0]}", ln=True)

    # Resultados por eje
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Resultados del análisis:", ln=True)
    pdf.set_font("Arial", size=12)
    for eje, valores in resultados.items():
        pdf.cell(200, 10, txt=f"{eje}", ln=True)
        pdf.cell(200, 10, txt=f"Frecuencia dominante: {valores['frecuencia']:.2f} Hz", ln=True)
        pdf.cell(200, 10, txt=f"Varianza: {valores['varianza']:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"RMS: {valores['rms']:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Desplazamiento máx: {valores['desplazamiento']:.2f} cm", ln=True)
        pdf.ln(5)

    # Diagnóstico
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Diagnóstico sugerido:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=diagnostico, ln=True)

    # Guardar archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        return tmpfile.name

# Interfaz Streamlit
st.title("Análisis de temblores - Parkinson vs Temblor esencial")

archivo = st.file_uploader("Subí el archivo CSV con los datos del paciente", type=["csv"])

if archivo is not None:
    datos_personales, datos_senales = procesar_archivo(archivo)
    st.subheader("Datos personales del paciente:")
    st.write(datos_personales)

    resultados = {}
    frecs_dom = []

    for eje in ['Aceleracion X', 'Aceleracion Y', 'Aceleracion Z']:
        st.subheader(f"Análisis del eje: {eje}")
        f_dom, var, rms, desplazamiento, xf, amplitudes = analizar_senal(datos_senales, eje)
        resultados[eje] = {
            'frecuencia': f_dom,
            'varianza': var,
            'rms': rms,
            'desplazamiento': desplazamiento
        }
        frecs_dom.append(f_dom)

        fig, ax = plt.subplots()
        ax.plot(xf, amplitudes)
        ax.set_title(f"FFT del eje {eje}")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud")
        st.pyplot(fig)

    diagnostico = diagnostico_simple(frecs_dom)
    st.subheader("Diagnóstico sugerido:")
    st.write(diagnostico)

    # Generar PDF
    pdf_path = generar_pdf(datos_personales, resultados, diagnostico)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Descargar informe PDF",
            data=f,
            file_name=f"{datos_personales['Nombre'].values[0]}_informe.pdf",
            mime="application/pdf"
        )
