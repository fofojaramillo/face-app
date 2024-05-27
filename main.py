import streamlit as st
import numpy as np
from PIL import Image
import json
from tensorflow.keras.models import model_from_json
import cv2


# Función para cargar el modelo de reconocimiento facial
@st.cache_resource
def load_face_model(model_json_path, weights_path):
    # Cargar el modelo desde un archivo JSON
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # Cargar los pesos del modelo
    model.load_weights(weights_path)
    return model

class SimpleFaceRecognition:
    def __init__(self, model_json_path, weights_path, database):
        self.model = load_face_model(model_json_path, weights_path)
        self.database = database
        self.preprocess_database()
        self.threshold = 1.0  # Nuevo umbral de distancia mínima

    def preprocess_image(self, image):
        # Normalizar y expandir dimensiones de la imagen
        img_resized = np.around(np.array(image) / 255.0, decimals=12)
        x_train = np.expand_dims(img_resized, axis=0)
        embedding = self.model.predict_on_batch(x_train)
        return embedding / np.linalg.norm(embedding, ord=2)

    def preprocess_database(self):
        # Preprocesar todas las imágenes en la base de datos
        for name, enc in self.database.items():
            self.database[name] = np.array(enc)

    def verify_identity(self, image):
        # Verificar identidad usando la función que implementaste anteriormente
        encoding = self.preprocess_image(image)
        min_dist = float('inf')
        min_name = None
        for name, enc in self.database.items():
            dist = np.linalg.norm(encoding - enc)
            if dist < min_dist:
                min_dist = dist
                min_name = name
        st.write(f'Distancia mínima: {min_dist}')  # Agregar un registro de la distancia mínima
        if min_dist < self.threshold:  # Comparar con el nuevo umbral
            return min_name
        else:
            return None

# Función para cargar la base de datos desde un archivo JSON
def load_database(database_path):
    with open(database_path, 'r') as f:
        database = json.load(f)
    return database

# Crear una aplicación Streamlit
def main():
    st.title('Reconocimiento Facial')

    captured_image = st.file_uploader('Cargar una imagen', type=['jpg', 'jpeg', 'png'])

    if captured_image:
        img = Image.open(captured_image)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img_np = np.array(img)

        # Redimensionar la imagen cargada al tamaño esperado por el modelo (160x160)
        img_resized = cv2.resize(img_np, (160, 160))

        # Mostrar la imagen redimensionada
        st.image(img_resized, caption='Imagen cargada y redimensionada', use_column_width=True)

        if st.button('Verificar Identidad'):
            with st.spinner('Verificando...'):
                identity = face_recognition.verify_identity(img_resized)
                if identity:
                    st.success(f'Bienvenido, {identity}!')
                else:
                    st.error('Lo siento, no se puede verificar la identidad.')

# Cargar la base de datos desde el archivo JSON
database = load_database('database.json')

# Crear una instancia de SimpleFaceRecognition con el modelo y la base de datos
face_recognition = SimpleFaceRecognition('model/model.json', 'model/model.h5', database)

# Ejecutar la aplicación
if __name__ == '__main__':
    main()