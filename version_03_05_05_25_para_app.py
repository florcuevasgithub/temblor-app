# -*- coding: utf-8 -*-
# app_temblor.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft, fftfreq
from fpdf import FPDF
from datetime import datetime, timedelta
import os

# --------- Funciones auxiliares ----------

def filtrar_temblor(signal, fs=200):
    b, a = butter(N=4, Wn=[3, 12], btype='bandpass', fs=fs)
    return filtfilt(b, a, signal)

def analizar_temblor_por_ventanas(signal, eje, fs=200, ventana_seg=2):
    signal = signal.dropna().to_numpy()
    signal_filtrada = filtrar_temblor(signal, fs)

    tamaño_ventana = int(fs * ventana_seg)
    num_ventanas = len(signal_filtrada) // tamaño_ventana
    resultados = []

    for i in range(num_ventanas):
        segmento = signal_filtrada[i*tamaño_ventana:(i+1)*tamaño_ventana]
        if len(segmento) < tamaño_ventana:
            continue

        f, Pxx = welch(segmento, fs=fs, nperseg=tamaño_ventana)
        freq_dominante = f[np.argmax(Pxx)]

        if eje in ['Acel_X', 'Acel_Y', 'Acel_Z']:
            segmento = segmento * 9.81  # g a m/s²

        varianza = np.var(segmento)
        rms = np.sqrt(np.mean(segmento**2))
        amplitud = np.max(segmento) - np.min(segmento)

        resultados.append({
            'Frecuencia Dominante (Hz)': freq_dominante,
            'Varianza (m2/s4)': varianza,
            'RMS (m/s2)': rms,
            'Amplitud Temblor (g)': amplitud
        })

    return pd.DataFrame(resultados)

def diagnosticar(df):
    def max_amp(df, test): return df[df['Test'] == test]['Amplitud Temblor (cm)'].max()
    def mean_freq(df, test): return df[df['Test'] == test]['Frecuencia Dominante (Hz)'].mean()

    if max_amp(df, 'Reposo') > 0.3 and 3 <= mean_freq(df, 'Reposo') <= 7:
        return "Probable Parkinson"
    elif (max_amp(df, 'Postural') > 0.3 or max_amp(df, 'Acción') > 0.3) and (8 <= mean_freq(df, 'Postural') <= 10 or 8 <= mean_freq(df, 'Acción') <= 10):
        return "Probable Temblor Esencial"
    else:
        return "Temblor dentro de parámetros normales"

def crear_grafico(df, nombre):
    resumen = df.groupby('Test')[['Frecuencia Dominante (Hz)', 'Varianza (m2/s4)', 'RMS (m/s2)', 'Amplitud Temblor (cm)']].mean()
    resumen.plot(kind='bar', figsize=(10, 6))
    plt.title(f"Resumen de Medidas - {nombre}")
    plt.ylabel("Valor")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("grafico_resumen.png")
    plt.close()

def generar_pdf(nombre_paciente, apellido_paciente, edad, sexo, diag_clinico, mano, dedo, diagnostico_auto, df):
    import unicodedata
    def limpiar_texto_para_pdf(texto):
        return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    fecha_hora = (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Informe de Análisis de Temblor", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Nombre: {nombre_paciente}", ln=True)
    pdf.cell(200, 10, f"Apellido: {apellido_paciente}", ln=True)
    pdf.cell(200, 10, f"Edad: {edad}", ln=True)
    pdf.cell(200, 10, f"Sexo: {sexo}", ln=True)
    pdf.cell(200, 10, f"Diagnóstico clínico: {diag_clinico}", ln=True)
    pdf.cell(200, 10, f"Mano: {mano}", ln=True)
    pdf.cell(200, 10, f"Dedo: {dedo}", ln=True)
    pdf.cell(200, 10, f"Fecha y hora: {fecha_hora}", ln=True)

    pdf.ln(5)
    pdf.image("grafico_resumen.png", x=10, w=180)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(30, 10, "Test", 1)
    pdf.cell(25, 10, "Eje", 1)
    pdf.cell(40, 10, "Frecuencia (Hz)", 1)
    pdf.cell(25, 10, "Varianza", 1)
    pdf.cell(25, 10, "RMS", 1)
    pdf.cell(45, 10, "Amplitud (cm)", 1)
    pdf.set_font("Arial", size=9)
    pdf.ln(10)

    for _, row in df.iterrows():
        pdf.cell(30, 10, row['Test'], 1)
        pdf.cell(25, 10, row['Eje'], 1)
        pdf.cell(40, 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
        pdf.cell(25, 10, f"{row['Varianza (m2/s4)']:.4f}", 1)
        pdf.cell(25, 10, f"{row['RMS (m/s2)']:.4f}", 1)
        pdf.cell(45, 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
        pdf.ln(10)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Interpretación clínica:", ln=True)
    pdf.set_font("Arial", size=11)
    texto_limpio = limpiar_texto_para_pdf.multi_cell(0, 8, """
Este informe analiza tres tipos de temblores: en reposo, postural y de acción.

Los valores de referencia considerados son:
  Para las frecuencias (Hz):
- Temblor Parkinsoniano: 3-6 Hz en reposo.
- Temblor Esencial: 8-10 Hz en acción o postura.

  Para las amplitudes 
- Mayores a 0.3 cm pueden ser clínicamente relevantes.

  Para la varianza (m2/s4) :
Representa la dispersión del temblor, en el contexto de temblores:
- Normal/sano: muy baja, puede estar entre 0.001 – 0.1 m2/s4.
- Temblor leve: entre 0.1 – 0.5 m2/s4.
- Temblor patológico (PK o TE): suele superar 1.0 m2/s4, llegando hasta 5–10 m2/s4 en casos severos.

  Para el RMS (m/s2):
- Normal/sano: menor a 0.5 m/s2.
- PK leve: entre 0.5 y 1.5 m/s2.
- TE o PK severo: puede llegar a 2–3 m/s2 o más.

La clasificación automática es orientativa y debe ser evaluada por un profesional.
""")
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"Diagnóstico automático: {diagnostico_auto}", ln=True)

    filename = f"{nombre_paciente}_informe_temblor.pdf"
    pdf.output(filename)
    return filename

# ----------- INTERFAZ STREAMLIT ------------

st.title("🧠 Análisis de Temblor")
st.write("Sube los tres archivos CSV correspondientes a las pruebas de Reposo, Postural y Acción.")

uploaded_files = {}
for test_name in ["Reposo", "Postural", "Acción"]:
    uploaded_files[test_name] = st.file_uploader(f"Archivo para test de {test_name}", type=["csv"], key=test_name)

if all(uploaded_files.values()):
    if st.button("Iniciar análisis"):
        mediciones_tests = {}
        datos_personales = None

        for test, file in uploaded_files.items():
            df = pd.read_csv(file)
            datos_personales = df.iloc[0].to_frame().T
            mediciones = df.iloc[1:][['Acel_X', 'Acel_Y', 'Acel_Z', 'GiroX', 'GiroY', 'GiroZ']]
            mediciones = mediciones.apply(pd.to_numeric, errors='coerce')
            mediciones_tests[test] = mediciones

        nombre = datos_personales.iloc[0].get("Nombre", "No especificado")
        apellido = datos_personales.iloc[0].get("Apellido", "No especificado")
        edad = datos_personales.iloc[0].get("Edad", "No especificado")
        sexo = datos_personales.iloc[0].get("Sexo", "No especificado")
        diag_clinico = datos_personales.iloc[0].get("Diagnostico", "No disponible")
        mano = datos_personales.iloc[0].get("Mano", "No disponible")
        dedo = datos_personales.iloc[0].get("Dedo", "No disponible")

        resultados_globales = []

        for test, datos in mediciones_tests.items():
            for eje in ['Acel_X', 'Acel_Y', 'Acel_Z', 'GiroX', 'GiroY', 'GiroZ']:
                df_ventana = analizar_temblor_por_ventanas(datos[eje], eje=eje, fs=50)
                if not df_ventana.empty:
                    prom = df_ventana.mean(numeric_only=True)
                    freq = prom['Frecuencia Dominante (Hz)']
                    amp_g = prom['Amplitud Temblor (g)']
                    amp_cm = (amp_g * 981) / ((2 * np.pi * freq) ** 2) if freq > 0 else 0.0

                    resultados_globales.append({
                        'Test': test,
                        'Eje': eje,
                        'Frecuencia Dominante (Hz)': round(freq, 2),
                        'Varianza (m2/s4)': round(prom['Varianza (m2/s4)'], 4),
                        'RMS (m/s2)': round(prom['RMS (m/s2)'], 4),
                        'Amplitud Temblor (cm)': round(amp_cm, 2)
                    })

        df_resultados = pd.DataFrame(resultados_globales)
        st.dataframe(df_resultados)

        diag_auto = diagnosticar(df_resultados)
        st.success(f"Diagnóstico automático: {diag_auto}")

        crear_grafico(df_resultados, nombre)

        archivo_pdf = generar_pdf(nombre, apellido, edad, sexo, diag_clinico, mano, dedo, diag_auto, df_resultados)

        with open(archivo_pdf, "rb") as f:
            st.download_button("📄 Descargar informe PDF", f, file_name=archivo_pdf)
