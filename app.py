import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform
import json
from streamlit_lottie import st_lottie

# Función para cargar el json del Lottie
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Cargar animación
lottie_hola = load_lottiefile("hola.json")

# Info del sistema
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")

image = Image.open('OIG5.jpg')
st.image(image, width=350)

with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Normalizar
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)
    print(prediction)

    # 👇 AQUÍ defines la clase "camilo"
    if prediction[0][0] > 0.5:
        st.header('Camilo detectado 😎 Probabilidad: ' + str(prediction[0][0]))

        # Mostrar animación Lottie
        st_lottie(lottie_hola, height=300, key="camilo")

    elif prediction[0][1] > 0.5:
        st.header('Arriba, con Probabilidad: ' + str(prediction[0][1]))
