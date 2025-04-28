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

# Función para procesar archivos
def procesar_archivo(uploaded_file):
    datos_completos = pd.read_csv(uploaded_file)
    datos_personales = datos_completos.iloc[0, :].to_frame().T
    datos_personales.reset_index(drop=True, inplace=True)
    mediciones = datos_completos.iloc[1:, :]
    mediciones = mediciones[['Acel_X', 'Acel_Y', 'Acel_Z', 'GiroX', 'GiroY', 'GiroZ']]
    mediciones = mediciones.apply(pd.to_numeric, errors='coerce')
    return datos_personales, mediciones

# Función para calcular parámetros de la resultante
@st.cache_data
def calcular_parametros_resultante(test_name, mediciones):
    sample_rate = 200
    n = len(mediciones)

    # Resultantes
    acel_resultante = np.sqrt(mediciones['Acel_X']**2 + mediciones['Acel_Y']**2 + mediciones['Acel_Z']**2).dropna().values
    giro_resultante = np.sqrt(mediciones['GiroX']**2 + mediciones['GiroY']**2 + mediciones['GiroZ']**2).dropna().values

    resultados = []
    imagenes = []

    for tipo, señal in [('Aceleracion', acel_resultante), ('Giroscopio', giro_resultante)]:
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

        resultados.append([test_name, tipo, frecuencia_dominante, varianza, rms, max_desplazamiento])

        # Guardar gráfico
        fig, ax = plt.subplots()
        ax.plot(frecuencias_filtradas, amplitudes_cm)
        ax.set_title(f"FFT {test_name} - {tipo}")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud (cm)")
        ax.grid(True)

        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(img_path)
        plt.close(fig)
        imagenes.append((test_name, tipo, img_path))

    return resultados, imagenes

# Diagnóstico
def diagnostico_global(resultados):
    posibles_diagnosticos = []
    comparativo = []
    for fila in resultados:
        test, tipo, freq, _, _, max_disp = fila
        if 4 <= freq <= 6 and max_disp > 0.5:
            diag = "Posible Parkinson"
            posibles_diagnosticos.append("Parkinson")
        elif 6 < freq <= 12 and max_disp <= 1.5:
            diag = "Posible Temblor Esencial"
            posibles_diagnosticos.append("Temblor Esencial")
        else:
            diag = "Normal"
        comparativo.append([test, tipo, diag])

    if posibles_diagnosticos.count("Parkinson") > posibles_diagnosticos.count("Temblor Esencial"):
        return comparativo, "Posible Parkinson"
    elif posibles_diagnosticos.count("Temblor Esencial") > posibles_diagnosticos.count("Parkinson"):
        return comparativo, "Posible Temblor Esencial"
    else:
        return comparativo, "Normal"

# Generador de PDF mejorado
def generar_pdf(datos_personales, resultados, comparativo, global_diag, imagenes, incluir_graficos=True):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Informe de Diagnóstico de Temblor", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", '', 10)
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(0, 10, f"Fecha del informe: {fecha}", ln=True)
    pdf.ln(5)

    # Info del paciente
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Información del Paciente:", ln=True)
    pdf.set_font("Arial", '', 10)
    for col in datos_personales.columns:
        val = datos_personales[col].values[0]
        if pd.notna(val) and str(val).strip().lower() not in ["nan", "sin info", "none", ""]:
            pdf.cell(0, 8, f"{col}: {val}", ln=True)

    pdf.ln(5)

    # Resultados
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Resultados del Análisis:", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(30, 8, "Test", 1)
    pdf.cell(30, 8, "Tipo", 1)
    pdf.cell(35, 8, "Frecuencia (Hz)", 1)
    pdf.cell(30, 8, "Varianza", 1)
    pdf.cell(25, 8, "RMS", 1)
    pdf.cell(40, 8, "Amplitud Max (cm)", 1)
    pdf.ln()
    pdf.set_font("Arial", '', 9)

    for r in resultados:
        pdf.cell(30, 8, r[0], 1)
        pdf.cell(30, 8, r[1], 1)
        pdf.cell(35, 8, f"{r[2]:.2f}", 1)
        pdf.cell(30, 8, f"{r[3]:.2f}", 1)
        pdf.cell(25, 8, f"{r[4]:.2f}", 1)
        pdf.cell(40, 8, f"{r[5]:.2f}", 1)
        pdf.ln()

    pdf.ln(5)

    # Conclusión
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Conclusión Preliminar:", ln=True)
    pdf.set_font("Arial", '', 10)

    if global_diag == "Posible Parkinson":
        texto_diag = "El análisis muestra frecuencias entre 4 y 6 Hz con amplitudes moderadas, compatible con temblor Parkinsoniano."
    elif global_diag == "Posible Temblor Esencial":
        texto_diag = "El análisis presenta frecuencias de 6-12 Hz con amplitudes bajas, compatible con Temblor Esencial."
    else:
        texto_diag = "No se identificaron patrones compatibles con temblores patológicos claros."

    pdf.multi_cell(0, 8, texto_diag)

    pdf.ln(5)

    # Diagnóstico Comparativo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Detalles del Diagnóstico Comparativo:", ln=True)
    pdf.set_font("Arial", '', 10)
    for fila in comparativo:
        pdf.cell(0, 8, f"{fila[0]} - {fila[1]}: {fila[2]}", ln=True)

    pdf.cell(0, 10, f"\nDiagnóstico Global Sugerido: {global_diag}", ln=True)

    # Gráficos si se desea
    if incluir_graficos:
        for test, tipo, img in imagenes:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"Gráfico - {test} - {tipo}", ln=True)
            pdf.image(img, x=10, y=25, w=190)

    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf.output(temp_path)
    return temp_path

# --- Streamlit App ---

st.title("🧠 Análisis de Temblor - Test Clínico Mejorado")
st.markdown("Cargá los tres archivos CSV correspondientes a Reposo, Postural y Acción para comenzar el análisis.")

with st.form("upload_form"):
    archivo_reposo = st.file_uploader("Archivo Reposo", type="csv")
    archivo_postural = st.file_uploader("Archivo Postural", type="csv")
    archivo_accion = st.file_uploader("Archivo Acción", type="csv")
    opcion_graficos = st.radio("¿Incluir gráficos de resultantes en el PDF?", ["Sí", "No"])
    submitted = st.form_submit_button("Comenzar Análisis")

if submitted and archivo_reposo and archivo_postural and archivo_accion:
    st.success("Archivos cargados correctamente. Procesando...")

    datos_personales, mediciones_reposo = procesar_archivo(archivo_reposo)
    _, mediciones_postural = procesar_archivo(archivo_postural)
    _, mediciones_accion = procesar_archivo(archivo_accion)

    resultados = []
    imagenes = []
    for nombre, datos in zip(["Reposo", "Postural", "Acción"], [mediciones_reposo, mediciones_postural, mediciones_accion]):
        res, imgs = calcular_parametros_resultante(nombre, datos)
        resultados.extend(res)
        imagenes.extend(imgs)

    comparativo, global_diag = diagnostico_global(resultados)

    incluir = True if opcion_graficos == "Sí" else False
    path_pdf = generar_pdf(datos_personales, resultados, comparativo, global_diag, imagenes, incluir_graficos=incluir)

    st.success("Análisis finalizado.")

    nombre_raw = datos_personales.iloc[0, 0] if not datos_personales.empty else ""
    nombre_paciente = str(nombre_raw).strip()

    if not nombre_paciente or nombre_paciente.lower() in ["nan", "none"]:
        nombre_paciente = "Paciente"

    nombre_pdf = f"Informe_Temblor_{nombre_paciente.replace(' ', '_')}.pdf"

    with open(path_pdf, "rb") as f:
        st.download_button("📄 Descargar Informe PDF", f, file_name=nombre_pdf)
