
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from fpdf import FPDF
import tempfile
import os
from datetime import datetime

def procesar_archivo(uploaded_file):
    datos_completos = pd.read_csv(uploaded_file)
    datos_personales = datos_completos.iloc[0, :].to_frame().T
    datos_personales.reset_index(drop=True, inplace=True)
    mediciones = datos_completos.iloc[1:, :]
    mediciones = mediciones[['Acel_X', 'Acel_Y', 'Acel_Z', 'GiroX', 'GiroY', 'GiroZ']]
    mediciones = mediciones.apply(pd.to_numeric, errors='coerce')
    return datos_personales, mediciones

@st.cache_data
def calcular_parametros(test_name, mediciones):
    sample_rate = 200
    n = len(mediciones)
    resultados = []
    imagenes = []
    for eje in ['Acel_X', 'Acel_Y', 'Acel_Z']:
        señal = mediciones[eje].dropna().values
        fft_valores = fft(señal)
        frecuencias = fftfreq(n, d=1/sample_rate)
        amplitudes = np.abs(fft_valores)

        frecuencias_positivas = frecuencias[:n // 2]
        amplitudes_positivas = amplitudes[:n // 2]
        filtro = (frecuencias_positivas >= 1) & (frecuencias_positivas <= 14)
        frecuencias_filtradas = frecuencias_positivas[filtro]
        amplitudes_filtradas = amplitudes_positivas[filtro]

        frecuencia_dominante = frecuencias_filtradas[np.argmax(amplitudes_filtradas)]
        amplitudes_cm = (amplitudes_filtradas / ((2 * np.pi * frecuencias_filtradas) ** 2)) * 100
        amplitudes_cm = np.nan_to_num(amplitudes_cm)

        varianza = np.var(señal)
        rms = np.sqrt(np.mean(señal**2))
        max_desplazamiento = np.max(amplitudes_cm)

        resultados.append([test_name, eje, frecuencia_dominante, varianza, rms, max_desplazamiento])

        fig, ax = plt.subplots()
        ax.plot(frecuencias_filtradas, amplitudes_cm)
        ax.set_title(f"FFT {test_name} - {eje}")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud estimada (cm)")
        ax.grid(True)

        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(img_path)
        plt.close(fig)
        imagenes.append((test_name, eje, img_path))

    return resultados, imagenes

def diagnostico_global(resultados):
    posibles_diagnosticos = []
    comparativo = []

    for fila in resultados:
        test, eje, freq, _, _, max_disp = fila

        if 4 <= freq <= 6 and max_disp > 1:
            diag = "Frecuencia compatible con temblor Parkinsoniano (4–6 Hz) y amplitud significativa."
            posibles_diagnosticos.append("Parkinson")
        elif 6 < freq <= 12 and max_disp <= 1.5:
            diag = "Frecuencia compatible con Temblor Esencial (6–12 Hz) y amplitud moderada o baja."
            posibles_diagnosticos.append("Temblor Esencial")
        elif freq < 4:
            diag = "Frecuencia baja (<4 Hz), fuera del rango típico de temblores patológicos. Podría ser fisiológico o artefacto."
        else:
            diag = "Frecuencia fuera de rangos típicos o amplitud no significativa. Compatible con patrón normal."

        comparativo.append([test, eje, diag])

    if posibles_diagnosticos.count("Parkinson") > posibles_diagnosticos.count("Temblor Esencial"):
        global_diag = "Compatibilidad predominante con temblor Parkinsoniano."
    elif posibles_diagnosticos.count("Temblor Esencial") > posibles_diagnosticos.count("Parkinson"):
        global_diag = "Compatibilidad predominante con Temblor Esencial."
    else:
        global_diag = "Sin evidencia clara de patrón patológico. Compatible con normalidad o patrón mixto."

    return comparativo, global_diag

def generar_pdf(datos_personales, resultados, comparativo, global_diag, imagenes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Informe de Diagnóstico de Temblor", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Fecha del informe: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "Información del Paciente:", ln=True)
    pdf.set_font("Arial", '', 10)
    for col in datos_personales.columns:
        val = datos_personales[col].values[0] if not pd.isna(datos_personales[col].values[0]) else "Sin info"
        pdf.cell(0, 8, f"{col}: {val}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "Resultados de Análisis:", ln=True)
    pdf.set_font("Arial", '', 9)
    pdf.multi_cell(0, 8, "Se analizan los ejes de aceleración en cada prueba (reposo, postural y acción). Se reportan frecuencia dominante, varianza [(m/s²)²], RMS [m/s²] y desplazamiento estimado [cm].")
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 8)
    pdf.cell(30, 10, "Test", 1)
    pdf.cell(20, 10, "Eje", 1)
    pdf.cell(40, 10, "Freq. Dom (Hz)", 1)
    pdf.cell(30, 10, "Varianza [(m/s²)²]", 1)
    pdf.cell(25, 10, "RMS [m/s²]", 1)
    pdf.cell(45, 10, "Max. Desplaz. [cm]", 1)
    pdf.ln()

    pdf.set_font("Arial", '', 8)
    for r in resultados:
        pdf.cell(30, 10, r[0], 1)
        pdf.cell(20, 10, r[1], 1)
        pdf.cell(40, 10, f"{r[2]:.2f}", 1)
        pdf.cell(30, 10, f"{r[3]:.2f}", 1)
        pdf.cell(25, 10, f"{r[4]:.2f}", 1)
        pdf.cell(45, 10, f"{r[5]:.2f}", 1)
        pdf.ln()

    pdf.ln(5)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "Diagnóstico Comparativo por Eje:", ln=True)
    pdf.set_font("Arial", '', 9)
    for fila in comparativo:
        pdf.cell(0, 8, f"{fila[0]} - {fila[1]}: {fila[2]}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(0, 10, f"Diagnóstico Global: {global_diag}", ln=True)
    pdf.set_text_color(0, 0, 0)

    for test, eje, img in imagenes:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, f"Análisis Espectral - {test} - {eje}", ln=True)
        pdf.set_font("Arial", '', 9)
        pdf.multi_cell(0, 8, "Transformada rápida de Fourier (FFT): amplitud estimada del temblor en función de la frecuencia.")
        pdf.image(img, x=10, y=40, w=180)

    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf.output(temp_path)
    return temp_path
