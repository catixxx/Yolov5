import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de la página
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Fondo tipo cielo con nubes (usando HTML + CSS)
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #aee1f9, #ffffff);
    background-attachment: fixed;
    background-size: cover;
    position: relative;
}

/* efecto de nubes suaves */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.8) 0%, transparent 60%),
                      radial-gradient(circle at 80% 30%, rgba(255,255,255,0.7) 0%, transparent 70%),
                      radial-gradient(circle at 40% 80%, rgba(255,255,255,0.8) 0%, transparent 60%);
    background-repeat: no-repeat;
    z-index: -1;
}

h1 {
    color: #0077cc !important; /* Azul cielo */
    text-align: center;
    font-weight: 800;
}

h2, h3, h4 {
    color: #006699;
}

div.stButton > button:first-child {
    background-color: #4fc3f7;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6em 1.2em;
    font-size: 16px;
    transition: 0.3s;
}
div.stButton > button:first-child:hover {
    background-color: #29b6f6;
    transform: scale(1.05);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            model = yolov5.load(model_path)
            return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instalar versiones compatibles:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        """)
        return None

# Título principal
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Bienvenido a la herramienta de **detección de objetos** con YOLOv5.  
Captura una imagen con tu cámara y deja que el modelo encuentre lo que ve.
""")

# Cargar modelo YOLOv5
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    # Barra lateral
    st.sidebar.title("⚙️ Parámetros de configuración")
    with st.sidebar:
        st.subheader('Ajustes del modelo')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles")

    # Contenedor principal
    main_container = st.container()
    with main_container:
        st.subheader("📸 Captura desde la cámara")
        picture = st.camera_input("Toma una foto", key="camera")

        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("🔎 Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            # Mostrar resultados
            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🖼️ Imagen con detecciones")
                    results.render()
                    st.image(cv2_img, channels='BGR', use_container_width=True)

                with col2:
                    st.subheader("📊 Resultados de detección")
                    label_names = model.names
                    category_count = {}
                    for category in categories:
                        idx = int(category.item())
                        category_count[idx] = category_count.get(idx, 0) + 1

                    data = []
                    for cat, count in category_count.items():
                        label = label_names[cat]
                        confidence = scores[categories == cat].mean().item()
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })

                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos.")
                        st.caption("Prueba bajando el umbral de confianza.")
            except Exception as e:
                st.error(f"Error al procesar resultados: {str(e)}")
else:
    st.error("No se pudo cargar el modelo. Verifica dependencias.")

# Pie de página
st.markdown("---")
st.caption("☁️ **Aplicación desarrollada con Streamlit y YOLOv5 — Inspirada en el cielo de Medellín.**")

